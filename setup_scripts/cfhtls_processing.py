import sys
import time
from itertools import product

import pandas as pd

from astropy.table import Table

from dxs.utils.table import fix_column_names

from dxs import paths

raw_cfhtls_path = paths.input_data_path / "external/cfhtls/W.I.cat.gz"

if not raw_cfhtls_path.exists():
    print(
        f"no raw cfhtls data to process at {raw_cfhtls_path} \n"
        f"download with:\n    ./cfhtls.wget"
    )
    sys.exit()
        

output_cfhtls_path = paths.input_data_path / "external/cfhtls/SA_i.fits"

keep_cols = ["RA", "DEC", "I_CLASS_STAR"]

bands = "U G R I Z".split()
quantity = "MAG MAGERR".split()
apers = "AUTO APER_1 APER_2 APER_3 APER_4 PETRO".split()
combos = [x for x in product(bands, quantity, apers)]

fixed_names = {"RA": "ra_cfhtls", "DEC": "dec_cfhtls", "I_CLASS_STAR": "I_class_star"}

for combo in combos:
    b, q, a = combo
    name = f"{b}_{q}_{a}"
    keep_cols.append(name)

    fixed_names[name] = f"{b}_{q.lower()}_{a.lower()}"

chunksize = 32_000
chunkcount = 0
keep_lines = 0

tables = []
t1 = time.time()
ts = time.time()
for chunk in pd.read_csv(
    raw_cfhtls_path,
    chunksize=chunksize, 
    delim_whitespace=True,
    #skiprows=23000000,
    compression="gzip",
    #index_col=0,
    usecols=keep_cols
):
    ra_mask = (332. < chunk["RA"]) & (chunk["RA"] < 336.0)
    dec_mask = (-1.5 < chunk["DEC"]) & (chunk["DEC"] < 2.0)

    good = chunk[ ra_mask & dec_mask ]
    if len(good) > 0:
        tables.append(good)
        keep_lines += len(good)
    chunkcount +=1

    n_lines = chunkcount*chunksize
    speed = n_lines / (time.time()-ts)
    print(f"read {n_lines:,} rows ({speed:.2f} lines/s); keep {keep_lines:,}")
    t1 = time.time()

output = pd.concat(tables)

table = Table.from_pandas(output)
table.write(output_cfhtls_path, format="fits", overwrite=True)
fix_column_names(output_cfhtls_path, column_lookup=fixed_names)






