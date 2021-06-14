from argparse import ArgumentParser
from pathlib import Path

from eazy.utils import path_to_eazy_data

from dxs import PhotozProcessor

from dxs import paths


field_choices = ["SA", "EN", "LH", "XM"]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--field", choices=field_choices)
    parser.add_argument("--input-catalog", default=None)
    parser.add_argument("--aper", default="aper_30")
    parser.add_argument("--optical", default="panstarrs")
    parser.add_argument("--optical-aper", default="aper_30")
    parser.add_argument("--mir", default="swire")
    parser.add_argument("--mir-aper", default="aper_30")
    parser.add_argument("--n-cpus", default=3, type=int)
    parser.add_argument("--K-cut", default=30.0, type=float)

    args = parser.parse_args()

    input_catalog = args.input_catalog
    if input_catalog is None:
        suffix = f"_{args.optical}_{args.mir}"
        input_catalog = paths.get_catalog_path(args.field, 00, "", suffix=suffix)
        output_dir = paths.get_catalog_dir(args.field, 00, "") / f"photoz{suffix}_{args.aper}"
    else:
        input_catalog = Path(input_catalog)
        stem = input_catalog.name.split(".")[0]
        output_dir = input_catalog.parent / f"{stem}_photoz_{args.aper}"

    pzp = PhotozProcessor(
        input_catalog,
        args.aper, {args.optical: args.optical_aper, args.mir: args.mir_aper},
        output_dir=output_dir,
        n_cpus=args.n_cpus,
    )
    
    bands_to_convert = [f"{b}_mag_aper_30" for b in "JHK"] + [f"{b}_mag_auto" for b in "JHK"]
        
    for b in "JHK":
        col = f"{b}_mag_{args.aper}" 
        if col not in bands_to_convert:
            bands_to_convert.append(col)

    galaxy_cut = 1.0 + 0.938 - 1.900
    # J_vega-K_vega > 1.0 >>> (J_AB-0.938) - (K_AB-1.9) > 1.0 >>> J_AB - K_AB > 1.0 + 0.938 - 1.9
    pzp.prepare_input_catalog(
        convert_from_vega=bands_to_convert,
        query=(
            f"(J_mag_aper_30) - (K_mag_aper_30) > {galaxy_cut}", 
            f"K_mag_auto < {args.K_cut}",
        )        
    )
    pzp.prepare_eazy_parameters(
        paths_to_modify=[
            "prior_file", "templates_file", "wavelength_file", "temp_err_file",
        ],
        z_max=6.0, 
        prior_filter="265",
        prior_abzp=23.9,
        prior_file="templates/prior_K_extend.dat",
        templates_file = (
            Path(path_to_eazy_data()) / "templates/fsps_full/tweak_fsps_QSF_12_v3.param"
        ),
    )
    pzp.initialize_eazy()
    pzp.run()
    pzp.match_result()

    pzp.make_plots(N=3000)
