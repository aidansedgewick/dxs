import numpy as np
import pandas as pd

from dxs import paths

def make_all_files():
    
    ###====================== KIM+2010 GAL, ERO, DRG ===========================###

    #MNRAS 410 241 (2011)

    K_bins = np.arange(15, 20.5, 0.5) + 1.900 # CONVERT TO AB!!
    log_gal = np.array(
        [2.165, 2.465, 2.742, 2.973, 3.174, 3.393, 3.568, 3.759, 3.900, 4.017, 4.101]
    )
    log_ero396 = np.array(
        [-np.inf,-np.inf,-np.inf,0.652, 1.690, 2.333, 2.815, 3.092, 3.203, 3.146, 2.678]
    )
    log_ero45 = np.array(
        [-np.inf,-np.inf,-np.inf,0.213, 0.991, 1.725, 2.374, 2.747, 2.875, 2.716, 1.815]
    )
    log_drg = np.array(
        [-np.inf,-np.inf,-np.inf,0.416, 0.814, 1.185, 1.761, 2.306, 2.665, 2.561, np.nan]
    )

    kim11_dict = {
        "Kmag": K_bins , 
        "galaxies": 10**log_gal, 
        "ero396_ctio": 10**log_ero396, 
        "ero450_ctio": 10**log_ero45, 
        "drg": 10**log_drg,
    }

    df = pd.DataFrame(kim11_dict)
    outpath = paths.input_data_path / "plotting/kim11_number_counts.csv"
    df.to_csv(outpath, index=False, float_format="%.2f")

    try:
        print_path = outpath.relative_to(paths.Path.cwd())
    except:
        print_path = outpath
    print(f"save to {print_path}")
    #for col in df.columns:
    #    if col == "K":
    #        continue
    #    df[col] = df[col].map(lambda x: f"{x:.3e}")

    # csv_path = 

    ###====================== KIM+2014 GAL, ERO ===========================###

    K_bins = np.arange(17.25, 23.25, 0.5)
    ero245_hsc = np.array(
        [0., 0., 4.16, 16.53, 88.01, 299.5, 718.03, 1260.93, 1653.75, 1684.22, 1384.71, 833.89]
    )
    ero245_ps = np.array(
        [0., 0., 0., 12.96, 87.83, 337.42, 746.86, 1032.96, 765.25, 266.72, 30.37, 0.]
    )
    ero295_hsc = np.array(
        [0., 0., 0., 0., 16.84, 82.92, 241.5, 495.5, 774.27, 773.32, 576.87, 280.45]
    )
    ero295_ps = np.array(
        [0., 0., 0., 0., 17.5, 96.03, 277.76, 371.9, 176.39, 14.17, 0., 0.]
    )
    galaxies = np.array(
        [0., 395.2, 764.91, 1267.09, 2018.8, 3093.7, 4386.01, 6098.18, 7998.12, 10287.64, 12005.77, 12466.94]
    )

    kim14_dict = {
        "Kmag": K_bins, 
        "ero245_hsc": ero245_hsc, 
        "ero245_ps": ero245_ps, 
        "ero295_hsc": ero295_hsc, 
        "ero295_ps": ero295_ps, 
        "galaxies": galaxies,
    }
    df = pd.DataFrame(kim14_dict)
    outpath = paths.input_data_path / "plotting/kim14_number_counts.csv"
    df.to_csv(outpath, index=False, float_format="%.2f")
    try:
        print_path = outpath.relative_to(paths.Path.cwd())
    except Exception as e:
        print_path = outpath
    print(f"save to {print_path}")

    ###====================== KAJISAWA+2006 DRG ===========================###

    #kajisawa drg counts
    #https://academic.oup.com/pasj/article/58/6/951/1446014

    K_mags = np.arange(18.75, 23.25, 0.5) + 1.900 ## CONVERT TO AB!!!
    density_per_arcmin_per_mag = np.array(
        [0.16, 0.25, 0.33, 0.58, 1.07, 0.41, 1.98, 1.48, 0.91]
    )

    #print(len(K_mags), len(density_per_arcmin_per_mag))

    density_per_sqdeg_half_mag = density_per_arcmin_per_mag * 3600. * 0.5

    df = pd.DataFrame({"Kmag": K_mags, "drg": density_per_sqdeg_half_mag})

    outpath = paths.input_data_path / "plotting/kajisawa06_number_counts.csv"
    df.to_csv(outpath, index=False, float_format="%.2f")
    try:
        print_path = outpath.relative_to(paths.Path.cwd())
    except:
        print_path = outpath
    print(f"save to {print_path}")
    ###========================== GRAZIAN+2006 DRGs =============================###

    grazian06_df = pd.DataFrame([
            [20.25, 6, -1.13, -0.91, -1.38, 1.05, 20.34, 135.372],
            [20.75, 14, -0.68, -0.58, -0.82, 1.25, 20.81, 135.372],
            [21.25, 16, -0.63, -0.53, -0.75, 1.42, 21.26, 135.372],
            [21.75, 22, -0.47, -0.39, -0.57, 1.96, 21.79, 129.692],
            [22.25, 29, -0.34, -0.27, -0.43, 2.04, 22.27, 128.273],
            [22.75, 50, -0.11, -0.05, -0.17, 2.45, 22.78, 127.935],
            [23.25, 32, -0.10, -0.03, -0.19, 2.80, 23.20, 81.272],
            [23.75, 10, 0.10, 0.26, -0.06, 2.75, 23.58, 12.585],
        ], columns=[
            "Kmag", "N", "per_sqmin_mag", "pos_sigma", "neg_sigma", "z_mean", "K_mean", "area_sqmin"
        ]
    )
    grazian06_bin_width = 0.5
    grazian06_df["drgs"] = (10**grazian06_df["per_sqmin_mag"] * 60. * 60. ) * grazian06_bin_width
    
    outpath = paths.input_data_path / "plotting/grazian06_number_counts.csv"
    grazian06_df.to_csv(outpath, index=False, float_format="%.2f")

    try:
        print_path = outpath.relative_to(paths.Path.cwd())
    except:
        print_path = outpath
    print(f"save to {print_path}")
    
    


    ###====================== MCCRACKEN+2010 BzK GALS ===========================###

    K_mags = np.arange(16.25, 23.25, 0.5)

    log_Ngal = np.array(
        [1.73, 2.03, 2.41, 2.65, 2.89, 3.14, 3.33, 3.53, 3.70, 3.84, 3.95, 4.07, 4.19, 4.29]
    )
    log_NsfBzK = np.array(
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 
            1.10, 1.51, 2.01, 2.57, 3.02, 3.35, 3.61, 3.77]
    )
    log_NpeBzK = np.array(
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 
            0.84, 1.56, 2.17, 2.52, 2.73, 2.83, 2.81, 2.72]
    )

    mccracken10_dict = {
        "Kmag": K_mags, 
        "galaxies": 10**log_Ngal, 
        "sf_BzK": 10**log_NsfBzK, 
        "pe_BzK": 10**log_NpeBzK
    }
    df = pd.DataFrame(mccracken10_dict)

    outpath = paths.input_data_path / "plotting/mccracken10_number_counts.csv"
    df.to_csv(outpath, index=False, float_format="%.2f")
    try:
        print_path = outpath.relative_to(paths.Path.cwd())
    except:
        print_path = outpath
    print(f"save to {print_path}")

    ###====================== ARCILA-OSEJO+2019 BzK GALS ===========================###

    K_mags = np.arange(15.25, 23.75, 0.5)
    pe_gzK = np.array(
        [0., 0., 0., 0., 0., 0., 0., 0.2, 2.1, 19.5, 87.3, 238., 399., 509., 462., 345., 312.]
    )
    sf_gzK = np.array(
        [0.04, 0.2, 0.0, 0.2, 0.58, 0.87, 2.4, 5.07, 13.5, 43.1, 176., 447., 1140., 2288., 4034., 5934., 7688.]
    )

    #arcila_osejo19_dict = 
    df = pd.DataFrame({"Kmag": K_mags, "pe_gzK": pe_gzK, "sf_gzK": sf_gzK})

    outpath = paths.input_data_path / "plotting/arcilaosejo19_number_counts.csv"
    df.to_csv(outpath, index=False, float_format="%.2f")
    try:
        print_path = outpath.relative_to(paths.Path.cwd())
    except:
        print_path = outpath
    print(f"save to {print_path}")

    ### =============================== SEO+2021 gzK


    df = pd.DataFrame([
            [17.50, 3.68],
            [18.00, 1.85],
            [18.50, 3.68],
            [19.00, 10.92],
            [19.49, 11.46],
            [20.00, 44.60],
            [20.49, 134.42],
            [21.00, 374.04],
            [21.50, 644.21],
            [21.99, 1145.59],
            [22.50, 1681.49],
        ], columns=["Kmag", "sf_gzK"]
    )

    outpath = paths.input_data_path / "plotting/seo21_number_counts.csv"
    df.to_csv(outpath, index=False, float_format="%.2f")
    try:
        print_path = outpath.relative_to(paths.Path.cwd())
    except:
        print_path = outpath
    print(f"save to {print_path}")
    ###==================== DAVIES+2021 DEVILS galaxies
    

    K_mags = np.arange(10., 28., 0.5)
    K_counts = np.array([
        0,0,0,0,0,0,0,0,0.3,2.2,7.9,19.8,39,89.7,163,288,505,885,1544,2442,3792,5435,
        7471,10410,14741,20908,28899,37206,41654,38743,30352,21584,13944,7989,4208,2219,
    ])

    df = pd.DataFrame({"Kmag": K_mags, "galaxies": K_counts})
    outpath = paths.input_data_path / "plotting/davies21_number_counts.csv"
    df.to_csv(outpath, index=False, float_format="%.2f")
    try:
        print_path = outpath.relative_to(paths.Path.cwd())
    except:
        print_path = outpath
    print(f"save to {print_path}")

if __name__ == "__main__":
    make_all_files()

