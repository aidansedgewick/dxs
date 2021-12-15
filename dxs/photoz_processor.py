import logging
import os
import shutil
import time
import tqdm
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

from stilts_wrapper import Stilts

from dxs.utils.misc import print_header
from dxs.utils.phot import vega_to_ab, ab_to_flux, apply_extinction

from dxs import paths

logger = logging.getLogger("photoz")

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

class PhotozProcessor:
    def __init__(
        self, 
        input_catalog_path, 
        #aperture, 
        catalog_apertures, 
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

        #self.aperture = aperture
        #self.external_catalog_apertures = external_catalog_apertures
    
        self.aperture = catalog_apertures.get("dxs", None)
        if self.aperture is None:
            self.aperture = catalog_apertures.get("uds", "NO_AP")
        self.catalog_apertures = catalog_apertures
        self.n_cpus = n_cpus

    def prepare_input_catalog(
        self,
        convert_from_vega=None,
        convert_magnitudes=True,
        magnitudes=True,
        flux_format="{band}_mag_{aperture}", 
        fluxerr_format="{band}_magerr_{aperture}",
        skip_bands=None,
        query=None,
    ):
        """
        convert_from_vega
           a list of bands to apply AB/Vega transform, defined in survey_config.
        convert_magnitudes
            if True, convert from (AB) magnitude to flux in uJy.
        magnitudes
            if True (default), assume input data is in magnutudes
            
     
        """
        print_header("prepare input catalog")
        full_catalog = Table.read(self.input_catalog_path)
        if "id" not in full_catalog.columns:
            full_catalog["id"] = np.arange(len(full_catalog))
        if "z_spec" not in full_catalog.columns:
            full_catalog["z_spec"] = np.zeros(len(full_catalog))
        coords = SkyCoord(ra=full_catalog["ra"], dec=full_catalog["dec"], unit="deg")
        sfdq = sfd.SFDQuery()
        ebv = sfdq(coords)

        skip_bands = skip_bands or []

        if convert_from_vega is None:
            convert_from_vega = []
        for data_col in convert_from_vega:
            if data_col not in full_catalog.columns:
                continue
            band = data_col.split("_")[0]
            logger.info(f"convert {data_col} from VEGA with band={band}")
            full_catalog[data_col] = vega_to_ab(full_catalog[data_col], band=band)
        
        translate_dict = {}

        for x, aper in self.catalog_apertures.items():
            if x == "dxs":
                bands = ["J", "H", "K"]
                eazy_codes = survey_config[x].get("eazy_filter_codes")
                
            elif x in survey_config["external_catalogs"]:
                bands = survey_config[x].get("bands")
                if bands is None:
                    raise ValueError(f"no bands in survey_config for {x}")
                eazy_codes =  survey_config[x].get("eazy_filter_codes", None)
                if eazy_codes is None:                    
                    raise ValueError(f"no eazy_filter_codes in survey_config for {x}")
            else:
                bands = [x] # assume that this is one band? how to get 
                raise ValueError(
                    r"for now give eg \{'panstarrs': 'aper_30'}, not indiv bands."
                )

            for band in bands:
                if band in skip_bands:
                    continue
                data_col = flux_format.format(band=band, aperture=aper)
                err_col = fluxerr_format.format(band=band, aperture=aper)
                if data_col not in full_catalog.columns:
                    logger.info(f"missing {data_col}")
                    continue
                print(f"use {data_col}")
                if magnitudes:
                    #if data_col in convert_from_vega:
                    #    logger.info(f"convert {data_col} to AB")
                    #    full_catalog[data_col] = vega_to_ab(full_catalog[data_col], band=band)
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
            catalog = catalog[:self.Nobj]
        logger.info(f"catalog length {len(catalog)}\n")
        catalog.write(self.photoz_input_path, overwrite=True)
        

    def prepare_eazy_parameters(
        self, paths_to_modify=None, modify_templates_file=True, **kwargs
    ):
        print_header("prep param file")
        param = modified_eazy_parameters(
            paths_to_modify=paths_to_modify,
            catalog_file=self.photoz_input_path, 
            main_output_file=self.output_dir / self.basename,
            **kwargs
        )
        if modify_templates_file:
            original_templates_file = Path(param["TEMPLATES_FILE"])
            modified_templates_file = self.output_dir / original_templates_file.name

            mod_templates_file = get_modified_templates_file(original_templates_file)
            with open(modified_templates_file, "w+") as f:
                
                f.writelines(mod_templates_file)
            param["TEMPLATES_FILE"] = str(modified_templates_file)

            original_param_fits = original_templates_file.with_suffix(".param.fits")
            new_param_fits = self.output_dir / original_param_fits.name                                    
            if original_param_fits.exists():
                print(f"copying {original_param_fits}")
                shutil.copy2(original_param_fits, new_param_fits)
            else:
                logger.warn("no param.fits!!")

        param.write(self.eazy_parameters_path)
        
    def initialize_eazy(self, param_path=None):
        print_header("init EAZY")
        param_path = param_path or self.eazy_parameters_path
        self.phz = PhotoZ(
            param_file=str(param_path), 
            translate_file=str(self.translate_file_path), 
            n_proc=self.n_cpus,
        )

    def fit_catalog(self, **kwargs):
        print_header("fit catalog")
        t1 = time.time()
        self.phz.fit_catalog(n_proc=self.n_cpus, timeout=7200, **kwargs)
        t2 = time.time()
        dt = t2 - t1
        rate = self.phz.NOBJ / dt
        logger.info(f"finish catalog fit")
        logger.info(f"fit {self.phz.NOBJ} obj in {dt:.2f}s (={rate:.2e} obj/s)")
        print()

    def standard_output(self, auto_save_fits=False, par_timeout=14400, **kwargs):
        print_header("standard output")

        
        t1 = time.time()
        
        extra_rf_filters = kwargs.pop("extra_rf_filters", [])
        absmag_filters = kwargs.pop("absmag_filters", [])
        assert "absmag_filters" not in kwargs

        print("standard output kwargs are")
        for k, v in kwargs.items():
            print(f"{k} : {v}")
        
        zout, hdu = self.phz.standard_output(
            prior=True, 
            beta_prior=True, 
            save_fits=auto_save_fits, 
            n_proc=self.n_cpus,
            par_skip=100,
            extra_rf_filters=[],
            absmag_filters=[],
            par_timeout=par_timeout,
            **kwargs
        )
        
        # Basically copy-paste from sps_parameters
        # do absmag by hand.
        if len(absmag_filters) > 0:
            print('Abs Mag filters', absmag_filters)
            absm = self.phz.abs_mag(
                f_numbers=absmag_filters, 
                cosmology=None, 
                rest_kwargs={
                    'percentiles': [2.5,16,50,84,97.5], 
                    'pad_width': 0.5, 
                    'max_err': 0.5, 
                    'verbose': False,
                    'simple': False,
                    'par_timeout': par_timeout,
                    'n_proc': self.n_cpus,
                }
            )
            for c in absm.colnames:
                zout[c] = absm[c]
            
            for key in absm.meta:
                zout.meta[key] = absm.meta[key]

        logger.info("done STDOUT")
        if not auto_save_fits:
            eazy_data_path = Path(path_to_eazy_data())
            for key, value in zout.meta.items():
                if isinstance(value, str) and len(value) > 50:
                    print(f"change {value}")
                    zout.meta[key] = value[-50:]
                    print("NEW", zout.meta[key])
            self.zout_path = Path(self.phz.param.params['MAIN_OUTPUT_FILE'] + ".zout.fits")
            if self.zout_path.exists():
                os.remove(self.zout_path)
                
            zout.write(self.zout_path, format='fits')
        t2 = time.time()
        dt = t2 - t1
        logger.info(f"write stdout in {t2-t1}")

    def match_result(self, output_path=None):
        print_header("match result")
        if output_path is None:
            stem = self.input_catalog_path.name.split(".")[0]
            output_path = (
                self.input_catalog_path.parent / 
                f"{stem}_photoz_{self.aperture}.cat.fits"
            )
        
        tab = Table.read(self.zout_path)
        tab.rename_column("ra", "ra_pz")
        tab.rename_column("dec", "dec_pz")
        temp_path = paths.scratch_stilts_path / output_path.name
        tab.write(temp_path, overwrite=True)
        #shutil.copy2(self.zout_path, temp_path)
        try:
            stilts = Stilts.tskymatch2(
                in1=self.input_catalog_path, 
                in2=temp_path,
                out=output_path,
                ra1="ra", dec1="dec",
                ra2="ra_pz", dec2="dec_pz",
                error=1.0,
                join="all1",
                all_formats="fits"
            )
            stilts.run()
        except Exception as e:
            print(e)
            print("match failed!")

    def make_plots(self, N=None):
        plot_dir = self.output_dir / "SED_plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        matplotlib.use('Agg')
        if N is None:
            N_plots = self.phz.NOBJ
        else:
            N_plots = min(N, self.phz.NOBJ)

        print_header(f"make {N_plots} plots")
        for ii in tqdm.tqdm(range(N_plots)):
            #print(f"plot {ii}")
            fig, params = self.phz.show_fit(ii, id_is_idx=True, add_label=False)
            axes = fig.axes
            txt = (
                f"z={self.phz.zbest[ii]:.2f}\n"
                f"ID={self.phz.OBJID[ii]}\n" 
                f"chi2={self.phz.chi2_best[ii]:.2f}\n"
                f"mag={self.phz.prior_mag_cat[ii]:.2f}"
            )
            axes[0].text(
                0.95, 0.95, txt, ha="right", va="top", fontsize=10, 
                transform=axes[0].transAxes, 
                bbox=dict(facecolor='w', alpha=0.5), zorder=10
                
            )

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

def get_modified_templates_file(templates_file):
    modified_file = []
    with open(templates_file) as f:
        for ii, line in enumerate(f):
            if line.startswith("#"):
                modified_file.append(line)
                continue
            lspl = line.split()
            old_path = lspl[1]
            new_path = str(Path(path_to_eazy_data()) / lspl[1])
            lspl[1] = new_path
            if ii == 0:
                print(f"modify file paths, eg. \n   {old_path} --> \n   {new_path}")
            modified_file.append(' '.join(lspl) + "\n")
    return modified_file
   


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

    #pzp.prepare_input_catalog(
        #convert_from_vega=["J_mag_aper_30", "K_mag_aper_30"],
        #query=(
        #    #f"(J_mag_aper_30 - 0.938) - (K_mag_aper_30 - 1.900) > 1.0", 
        #    #f"i_mag_aper_30 - K_mag_aper_30 > 2.45", "i_mag_aper_30 < 25.0",
        #    #f"K_mag_auto + 1.900 < {args.K_cut}",
        #)
    #)
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


            
