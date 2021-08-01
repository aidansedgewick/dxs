import copy
import yaml
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scipy.interpolate import interp1d

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling import models
from astropy.table import Table
from astropy.wcs import WCS


from photutils.datasets import make_gaussian_sources_image
from regions import read_ds9
from reproject.mosaicking import find_optimal_celestial_wcs

from dxs import CatalogExtractor, Stilts, merge_catalogs
from dxs.utils.image import uniform_sphere, objects_in_coverage
from dxs.utils.misc import create_file_backups, calc_mids, tile_parser

from dxs import paths

survey_config_path = paths.config_path / "survey_config.yaml"
with open(survey_config_path, "r") as f:
    survey_config = yaml.load(f, Loader=yaml.FullLoader)

analysis_config_path = paths.config_path / "analysis_config.yaml"
with open(analysis_config_path, "r") as f:
    analysis_config = yaml.load(f, Loader=yaml.FullLoader)

parser = ArgumentParser()
parser.add_argument("--gen-stars", default=False, action="store_true")
parser.add_argument("--force-extract", default=False, action="store_true")
parser.add_argument("--map-plot", default=True, action="store_true")
parser.add_argument("--tiles", default=None)

args = parser.parse_args()

if args.gen_stars:
    args.force_extract = True

fields = ["SA", "LH"]

for field in fields:

    if args.tiles is not None:
        tiles = tile_parser(args.tiles)
    else:
        tiles = [x for x in range(1, survey_config["tiles_per_field"][field]+1)]
    print("tiles", tiles)

    band = "K"

    field_limits = survey_config["field_limits"][field]

    print(field_limits)

    density = int(1e4)


    dM = 0.25
    mag_bins = np.arange(19.0, 28. + dM, dM)
    mag_mids = calc_mids(mag_bins)


    field_input_stars_path = paths.temp_swarp_path / f"{field}_input_stars.cat.fits"
    if not field_input_stars_path.exists() or args.gen_stars:
        print("generate new randoms")
        randoms = uniform_sphere(field_limits["ra"], field_limits["dec"], density=density)
        mags = np.random.uniform(20, 25, len(randoms)) - 1.900

        field_input_stars = Table({"ra": randoms[:,0], "dec": randoms[:,1], "input_mag": mags})
        print(field_input_stars)

        field_input_stars.write(field_input_stars_path, overwrite=True)

        args.force_extract = True
    else:
        print("reading existing randoms")
        field_input_stars = Table.read(field_input_stars_path)

    mosaic_headers = []
    mosaic_regions = []
    joined_catalog_paths = []

    #matplotlib.use('Agg')

    fig, ax = plt.subplots()
    for tile in tiles:
        input_stars = copy.deepcopy(field_input_stars)

        mosaic_path = paths.get_mosaic_path(field, tile, band)
        segmentation_path = paths.get_mosaic_path(field, tile, band, extension=".seg.fits")

        print(mosaic_path)
        print(segmentation_path)

        spec = (field, tile, band)

        imgout_name = paths.get_mosaic_stem(*spec, suffix="_fake_stars") + ".fits"
        imgout_path = paths.temp_swarp_path / imgout_name
        input_stars_name = paths.get_catalog_stem(*spec, suffix="_input_stars") + ".cat.fits"
        input_stars_path = paths.temp_swarp_path / input_stars_name
        extracted_cat_name = paths.get_catalog_stem(*spec, suffix="_extracted") + ".cat.fits"
        extracted_cat_path = paths.temp_swarp_path / extracted_cat_name
        joined_cat_name = paths.get_catalog_stem(*spec, suffix="_merged") + ".cat.fits"
        joined_cat_path = paths.temp_swarp_path / joined_cat_name

        joined_catalog_paths.append(joined_cat_path)

        with fits.open(segmentation_path) as s:
            swcs = WCS(s[0].header)
            seg_map = s[0].data

            mosaic_headers.append((seg_map, swcs))
            
            region_path = paths.get_mosaic_path(
                field, tile, band, extension=".reg"
            )
            region = read_ds9(region_path)
            assert len(region) == 1
            mosaic_regions.append(region[0])
            footprint = swcs.calc_footprint()

        print("not force", args.force_extract, "exists", extracted_cat_path.exists())
        print("or", (not args.force_extract) or (extracted_cat_path.exists()))
        if (not args.force_extract):
            if (extracted_cat_path.exists()):
                continue

        print("modify mosaic")

        ylim, xlim = seg_map.shape
        randoms = np.column_stack([input_stars["ra"], input_stars["dec"]])

        randoms_pix = swcs.wcs_world2pix(randoms, 0).astype(int)
        input_stars["xcoord"] = randoms_pix[:,0]
        input_stars["ycoord"] = randoms_pix[:,1]

        ymask = (0 < input_stars["ycoord"]) & (input_stars["ycoord"] < ylim)
        xmask = (0 < input_stars["xcoord"]) & (input_stars["xcoord"] < xlim)

        input_stars = input_stars[ ymask & xmask ]
        
        seg_mask = seg_map[input_stars["ycoord"], input_stars["xcoord"]] == 0

        input_stars = input_stars[ seg_mask ]
        print(f"keep {len(input_stars)}")
        input_stars.write(input_stars_path, overwrite=True)
        print(f"injected stars at {input_stars_path}")

        with fits.open(mosaic_path) as f:
            fwcs = WCS(f[0].header)        
            data = f[0].data

            gain = f[0].header["GAIN"]
            zpt = f[0].header["MAGZPT"]
            seeing = f[0].header["SEEING"]

            flux = (10**(0.4 * (zpt - input_stars["input_mag"]))) #/ gain

            n_sources = len(input_stars)
            source_table = Table()
            #source_table["flux"] = flux
            source_table["y_mean"] = input_stars["ycoord"]
            source_table["x_mean"] = input_stars["xcoord"]

            fwhm = (seeing / 3600.) / f[0].header["CD2_2"]
            stddev = fwhm / 2.355

            xstd = np.full(n_sources, stddev)
            ystd = np.full(n_sources, stddev)
            
            source_table["x_stddev"] = xstd
            source_table["y_stddev"] = ystd
            source_table["amplitude"] = flux / (2. * np.pi * xstd * ystd)

            model_list = [models.Gaussian2D(**dict(row)) for row in source_table]

            source_img = np.zeros(data.shape)
            for ii, model in tqdm(enumerate(model_list)):
                model.bounding_box = (
                    (model.y_mean-30, model.y_mean+30), (model.x_mean-30, model.x_mean+30)
                )
                model.render(source_img)

            mod_data = data + source_img

            hdu = fits.PrimaryHDU(data=mod_data, header=f[0].header)

            print(f"write to {imgout_path}")
            hdu.writeto(imgout_path, overwrite=True)
        
        extractor = CatalogExtractor(
            imgout_path, catalog_path=extracted_cat_path, use_weight=None
        )
        extractor.extract()
        print(f"extracted at {extracted_cat_path}")

        stilts = Stilts.tskymatch2_fits(
            input_stars_path, extracted_cat_path, joined_cat_path, 
            find="best", join="all1", error=1.0,
            ra1="ra", dec1="dec", ra2="X_WORLD", dec2="Y_WORLD",
        )
        stilts.run()
        print(f"merged at {joined_cat_path}")

    for tile, joined_cat_path in zip(tiles, joined_catalog_paths):
        joined = Table.read(joined_cat_path)

        joined["MAG_AUTO"][ ~np.isfinite(joined["MAG_AUTO"]) ] = 99.0
        joined["input_mag"] = joined["input_mag"] + 1.900

        detected = joined[ joined["MAG_AUTO"] < 50. ]

        denom_hist, _ = np.histogram(joined["input_mag"], bins=mag_bins)
        numer_hist, _ = np.histogram(detected["input_mag"], bins=mag_bins)

        mask = denom_hist > 0
        denom_hist = denom_hist[ mask ]
        numer_hist = numer_hist[ mask ]

        mag_mids_relevant = mag_mids[ mask ]

        ratio = numer_hist / denom_hist

        if tile >10:
            ls="--"
        else:
            ls="-"

        ax.plot(mag_mids_relevant, ratio, label=tile, ls=ls)
        
    ax.legend(ncol=3)
    ax.axhline(0.9, color="k", ls=":")
    ax.axvline(22.7, color="k", ls=":")
    ax.set_ylabel("fraction recovered", fontsize=12)
    ax.set_xlabel("$K$", fontsize=12)


    """
    if args.map_plot:

        from astropy.wcs.utils import (
            celestial_frame_to_wcs, pixel_to_skycoord, proj_plane_pixel_scales,
            skycoord_to_pixel, wcs_to_celestial_frame
        )

        resolution = 10. * u.arcminute

        projection="TAN"

        output_wcs, reproject_shape = find_optimal_celestial_wcs(
            mosaic_headers, resolution=resolution
        )


        print(output_wcs, reproject_shape)


        corners = []
        references = []
        #resolutions = []

        for array, wcs in mosaic_headers:

            frame = wcs_to_celestial_frame(wcs)

            ny, nx = array.shape
            xc = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
            yc = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])

            corners.append(pixel_to_skycoord(xc, yc, wcs, origin=0).icrs.frame)

            xp, yp = wcs.wcs.crpix
            references.append(pixel_to_skycoord(xp, yp, wcs, origin=1).icrs.frame)

        corners = SkyCoord(corners)
        references = SkyCoord(references)
        
        reference = SkyCoord(references.data.mean(), frame=references.frame)
        reference = reference.transform_to(frame)

        cdelt = resolution.to(u.deg).value
        

        wcs_final = celestial_frame_to_wcs(frame, projection=projection)

        rep = reference.represent_as('unitspherical')
        wcs_final.wcs.crval = rep.lon.degree, rep.lat.degree
        wcs_final.wcs.cdelt = -cdelt, cdelt
        
        xp, yp = skycoord_to_pixel(corners, wcs_final, origin=1)

        xmin = xp.min() - 1
        xmax = xp.max() + 1
        ymin = yp.min() - 1
        ymax = yp.max() + 1

        wcs_final.wcs.crpix = (1 - xmin) + 0.5, (1 - ymin) + 0.5

        # Return the final image shape too
        naxis1 = int(np.ceil(xmax - xmin))
        naxis2 = int(np.ceil(ymax - ymin))

        shape = (naxis2, naxis1)

        print(wcs_final, shape)

        merged_catalog_name = paths.get_catalog_stem(
            field, 00, band, suffix="_joined"
        ) + ".cat.fits"
        merged_catalog_path = paths.temp_swarp_path / merged_catalog_name
        if not merged_catalog_path.exists():
            merge_catalogs(
                joined_catalog_paths, mosaic_regions, merged_catalog_path,
                id_col="NUMBER", ra_col="ra", dec_col="dec", snr_col="input_mag"
            )

        catalog = Table.read(merged_catalog_path)
        catalog["MAG_AUTO"][ ~np.isfinite(catalog["MAG_AUTO"]) ] = 99.0
        catalog["input_mag"] = catalog["input_mag"] + 1.900

        depth_map = np.zeros(shape)

        coords = np.column_stack([catalog["ra"], catalog["dec"]])
        #pix = skycoord_to_pixel(catalog["ra"].data, catalog["dec"].data, wcs_final)
        pix = wcs_final.wcs_world2pix(coords, 0).astype(int)

        print(np.min(pix[:,0]), np.max(pix[:,0]))
        print(np.min(pix[:,1]), np.max(pix[:,1]))

        print(depth_map.shape)
            
        catalog["ypix"] = pix[:, 1]
        catalog["xpix"] = pix[:, 0]

        fig2, ax2 = plt.subplots()

        dm = 0.01

        grouped = catalog.group_by(["ypix", "xpix"])
        for idx, area_input in zip(grouped.groups.keys, grouped.groups):

            if len(area_input) < 100:
                continue
            area_detected = area_input[ area_input["MAG_AUTO"] < 50. ]
            
            numer_hist, _ = np.histogram(area_detected["input_mag"], bins=mag_bins)
            denom_hist, _ = np.histogram(area_input["input_mag"], bins=mag_bins)

            mask = (denom_hist > 0)

            denom_hist = denom_hist[ mask ]
            numer_hist = numer_hist[ mask ]
            mag_mids_relevant = mag_mids[ mask ]

            mag_grid = np.arange(
                mag_mids_relevant.min(), mag_mids_relevant.max(), dm
            )

            
            ratio = numer_hist / denom_hist

            spline = interp1d(mag_mids_relevant, ratio)

            evals = spline(mag_grid)
            depth_idx = np.arange(len(evals))[evals < 0.9][0]
            depth = mag_grid[depth_idx]

            depth_map[idx["ypix"], idx["xpix"]] = depth

            ax2.plot(mag_mids_relevant, ratio)

        
        fig3, ax3 = plt.subplots()
        ax3.imshow(depth_map, vmin=20., vmax=24.)"""

plt.show()

#fig2.savefig(paths.base_dir/"completeness_curves.png")
#fig3.savefig(paths.base_dir/"depth_map.png")





















    
