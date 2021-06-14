import numpy as np
import pandas as pd

from dxs import paths


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

print(f"save_to_ \n{outpath}")

"""for col in df.columns:
    if col == "K":
        continue
    df[col] = df[col].map(lambda x: f"{x:.3e}")

csv_path = """

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
print(f"saved to \n{outpath}")

###====================== KAJISAWA+2006 DRG ===========================###

#kajisawa drg counts
#https://academic.oup.com/pasj/article/58/6/951/1446014

K_mags = np.arange(18.75, 23.25, 0.5) + 1.900 ## CONVERT TO AB!!!
density_per_arcmin_per_mag = np.array(
    [0.16, 0.25, 0.33, 0.58, 1.07, 0.41, 1.98, 1.48, 0.91]
)

print(len(K_mags), len(density_per_arcmin_per_mag))

density_per_sqdeg_half_mag = density_per_arcmin_per_mag * 3600. * 0.5

df = pd.DataFrame({"Kmag": K_mags, "drg": density_per_sqdeg_half_mag})

outpath = paths.input_data_path / "plotting/kajisawa06_number_counts.csv"
df.to_csv(outpath, index=False, float_format="%.2f")
print(f"saved to \n{outpath}")

###====================== MCCRACKEN+2006 BzK GALS ===========================###

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
print(f"save to \n{outpath}")

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
print(f"save to \n{outpath}")



