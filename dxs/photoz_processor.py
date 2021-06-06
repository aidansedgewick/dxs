import logging
import os
import time
import yaml
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from dustmaps import sfd
from easyquery import Query
from eazy.photoz import PhotoZ
from eazy.param import EazyParam
from eazy.utils import path_to_eazy_data

from dxs.utils.phot import vega_to_ab, ab_to_flux, apply_extinction

from dxs import paths

logger = logging.getLogger("photoz_processor")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

class PhotozProcessor:
    def __init__(
        self, 
        input_catalog_path, 
        aperture, 
        external_catalog_apertures, 
        output_dir=None,
        Nobj=None,
        basename="photoz",
        n_cpus=1,
    ):
        """
        Parameters
        ----------
        input_catalog_path
        aperture
            str describing which aperture to use for DXS data eg. "auto", "aper_30"
        external_catalog_apertures
            dict[str] eg {"panstarrs": "aper_30", "swire": "aper_30"}
        output_dir
            path to directory where data are to be saved
        """

        self.input_catalog_path = Path(input_catalog_path)
        print(f"read {self.input_catalog_path}")
        self.catalog_stem = self.input_catalog_path.name.split(".")[0]

        if output_dir is None:
            output_dir = input_catalog_path.parent / "photoz"
        self.output_dir = Path(output_dir).absolute()
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.basename = basename

        self.Nobj = Nobj

        self.translate_file_path = self.output_dir / f"{self.basename}.translate"

        output_name = f"{self.input_catalog_path.stem}.{self.basename}.cat.fits"
        self.photoz_input_path = self.output_dir / output_name
        self.eazy_parameters_path = self.output_dir / f"{self.basename}.param"

        self.aperture = aperture
        self.external_catalog_apertures = external_catalog_apertures
        self.n_cpus = n_cpus

    def prepare_input_catalog(
        self,
        convert_from_vega=None,
        convert_magnitudes=True,
        magnitudes=True,
        flux_format="{band}_mag_{aperture}", 
        fluxerr_format="{band}_magerr_{aperture}",
        query=None,
    ):

        full_catalog = Table.read(self.input_catalog_path)
        if "id" not in full_catalog.columns:
            full_catalog["id"] = np.arange(len(full_catalog))
        if "z_spec" not in full_catalog.columns:
            full_catalog["z_spec"] = np.zeros(len(full_catalog))
        coords = SkyCoord(ra=full_catalog["ra"], dec=full_catalog["dec"], unit="deg")
        sfdq = sfd.SFDQuery()
        ebv = sfdq(coords)

        if convert_from_vega is None:
            convert_from_vega = []
        
        d = {"dxs": self.aperture, **self.external_catalog_apertures}
        translate_dict = {}
        for survey, aper in d.items():
            if survey == "dxs":
                bands = ["J", "H", "K"]
                eazy_codes = survey_config["eazy_filter_codes"]
            else:
                bands = survey_config[survey].get("bands")
                eazy_codes = survey_config[survey].get("eazy_filter_codes")
            for band in bands:
                data_col = flux_format.format(band=band, aperture=aper)
                err_col = fluxerr_format.format(band=band, aperture=aper)
                if data_col not in full_catalog.columns:
                    logger.info(f"missing {data_col}")
                    continue
                if magnitudes:
                    if data_col in convert_from_vega:
                        logger.info(f"convert {data_col} to AB")
                        full_catalog[data_col] = vega_to_ab(full_catalog[data_col], band=band)
                    if survey_config["reddening_coeffs"].get(band, None) is not None:
                        full_catalog[data_col] = apply_extinction(
                            full_catalog[data_col], ebv, band=band
                        )
                    else:
                        print(f"no reddening coeff for {band}")
                if convert_magnitudes:
                    flux_col = f"{band}_flux"
                    fluxerr_col = f"{band}_fluxerr"
                    mask = full_catalog[data_col] < 30.
                    flux = np.full(len(full_catalog), -99.)
                    flux[ mask ] = ab_to_flux(
                        full_catalog[data_col][ mask ], band
                    ) * 1e6 # uJansky
                    full_catalog[flux_col] = flux

                    fluxerr = np.full(len(full_catalog), -99.)
                    snr = full_catalog[err_col] * np.log(10.) / 2.5
                    fluxerr[ mask ] = flux[ mask ] * snr[ mask ]
                    full_catalog[fluxerr_col] = fluxerr
                    
                else:
                    flux_col = data_col
                    fluxerr_col = err_col
                code = eazy_codes[band]
                translate_dict[flux_col] = f"F{code}"
                translate_dict[fluxerr_col] = f"E{code}"

        with open(self.translate_file_path, "w+") as f:
            lines = []
            for k, v in translate_dict.items():
                if k in full_catalog.columns:
                    lines.append(f"{k} {v}\n")
            f.writelines(lines)
            logger.info(f"write translate_file {self.translate_file_path}")

        if query is not None:
            catalog = Query(*query).filter(full_catalog)
        else:
            catalog = full_catalog

        if self.Nobj is not None:
            logger.info("selecting first {self.Nobj} objects, AFTER query...")        
        catalog.write(self.photoz_input_path, overwrite=True)
        

    def prepare_eazy_parameters(self, paths_to_modify=None, **kwargs):
        param = modified_eazy_parameters(
            paths_to_modify = paths_to_modify,
            catalog_file=self.photoz_input_path, 
            main_output_file=self.output_dir / self.basename,
            **kwargs
        )
        param.write(self.eazy_parameters_path)
        
    def initialize_eazy(self, ):
        self.phz = PhotoZ(
            param_file=str(self.eazy_parameters_path), 
            translate_file=str(self.translate_file_path), 
            n_proc=self.n_cpus,
        )

    def run(self):
        t1 = time.time()
        self.phz.fit_parallel()
        t2 = time.time()
        dt = t2 - t1
        rate = self.phz.NOBJ / dt 
        print(f"fit {self.phz.NOBJ} parallel in {dt:.2f} s (={rate:.2e}  obj / s)")
        t1 = time.time()
        zout, hdu = self.phz.standard_output(
            prior=True, beta_prior=True, save_fits=True, n_proc=self.n_cpus,
        )
        t2 = time.time()
        dt = t2 - t1
        print(f"write stdout in {t2-t1}")

    def make_plots(self, N=None):
        plot_dir = self.output_dir / "SED_plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        matplotlib.use('Agg')
        if N is None:
            N = self.phz.NOBJ

        for ii in range(N):
            print(f"plot {ii}")
            fig, params = self.phz.show_fit(ii, id_is_idx=True)
            fig_name = plot_dir / f"{self.catalog_stem}_{ii:06d}.png"
            fig.savefig(fig_name)
            plt.close()
                

def modified_eazy_parameters(
    paths_to_modify=None, **kwargs
):
    eazy_data_path = Path(path_to_eazy_data())
    paths_to_modify = paths_to_modify or []
    param = EazyParam()
    param["FILTERS_RES"] = eazy_data_path / f"filters" / param["FILTERS_RES"]
    print(kwargs)
    for key, value in kwargs.items():
        #if key.upper() not in param:
        #    print(f"WARNING: unknown key {key.upper()}")
        if isinstance(value, Path):
            value = str(value)
        param[key.upper()] = value
    for path in paths_to_modify:
        print(path, param[path.upper()])
        param[path.upper()] = eazy_data_path / param[path.upper()]

    return param


if __name__ == "__main__":
    #pzp = PhotozProcessor(
    #    paths.catalogs_path / "EN00/EN00_panstarrs_swire.cat.fits",
    #    "aper_30", {"panstarrs": "aper_30", "swire": "aper_30"},
    #)
    parser = ArgumentParser()
    parser.add_argument("catalog_path")
    parser.add_argument("--Nobj", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--K-cut", type=float, default=30.0)
    args = parser.parse_args()

    input_path = Path(args.catalog_path).absolute()
    output_dir = args.output_dir
    N = args.Nobj

    pzp = PhotozProcessor(
        input_path, "aper_30", {"panstarrs": "aper_30", "swire": "aper_30"},
        output_dir=args.output_dir,
        n_cpus=3
    )

    pzp.prepare_input_catalog(
        convert_from_vega=["J_mag_aper_30", "K_mag_aper_30"],
        query=(
            f"(J_mag_aper_30 - 0.938) - (K_mag_aper_30 - 1.900) > 1.0", 
            f"i_mag_aper_30 - K_mag_aper_30 > 2.45", "i_mag_aper_30 < 25.0",
            f"K_mag_auto + 1.900 < {args.K_cut}",
        )
    )
    pzp.prepare_eazy_parameters(
        paths_to_modify=[
            "prior_file", "templates_file", "wavelength_file", "temp_err_file",
        ],
        z_max=6.0, 
        prior_filter="K_flux",
        prior_file="templates/prior_K_TAO.dat",
    )
    pzp.initialize_eazy()
    pzp.run()
    pzp.make_plots(N=10000)


            
