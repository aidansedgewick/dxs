from argparse import ArgumentParser

from dxs import PhotozProcessor

from dxs import paths


field_choices = ["SA", "EN", "LH", "XM"]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("field", choices=field_choices)
    parser.add_argument("--optical", default="panstarrs")
    parser.add_argument("--optical-aper", default="aper_30")
    parser.add_argument("--mir", default="swire")
    parser.add_argument("--mir-aper", default="aper_30")
    parser.add_argument("--n-cpus", default=3, type=int)
    parser.add_argument("--K-cut", default=30.0, type=float)

    args = parser.parse_args()


    suffix = f"_{args.optical}_{args.mir}"
    input_catalog = paths.get_catalog_path(args.field, 00, "", suffix=suffix)

    output_dir = paths.get_catalog_dir(args.field, 00, "") / f"photoz{suffix}"

    pzp = PhotozProcessor(
        paths.catalogs_path / "EN00/EN00_panstarrs_swire.cat.fits",
        "aper_30", {args.optical: args.optical_aper, args.mir: args.mir_aper},
        output_dir=output_dir
    )
    pzp.prepare_input_catalog(
        convert_from_vega=["J_mag_aper_30", "K_mag_aper_30"],
        query=(
            "(J_mag_aper_30 - 0.938) - (K_mag_aper_30 - 1.900) > 1.0", 
            f"K_mag_auto + 1.900 < {args.K_cut}"
        less)        
    )
    pzp.prepare_eazy_parameters(
        paths_to_modify=[
            "prior_file", "templates_file", "wavelength_file", "temp_err_file",
        ],
        z_max=6.0, 
        prior_filter="K_flux",
        prior_file="templates/prior_K_TAO.dat",
    )
    pzp.initialize_eazy(n_cpus=args.n_cpus)
    pzp.run()
