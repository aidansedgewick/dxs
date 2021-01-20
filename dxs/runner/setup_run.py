from argparse import ArgumentParser
from itertools import product
from pathlib import Path

from dxs.runner import ScriptMaker
from dxs import paths

parser = ArgumentParser()
parser.add_argument("fields")
parser.add_argument("tiles")
parser.add_argument("bands")
parser.add_argument("python_script")
parser.add_argument("--run_name", action="store", required=True)

args = parser.parse_args()

fields = [x for x in args.fields.split(",")]
tile_ranges = [x for x in args.tiles.split(",")]
tiles = []
for t in tile_ranges:
    if "-" in t:
        ts = t.split("-")
        tiles.extend([x for x in range(int(ts[0]), int(ts[1])+1)])
    else:
        tiles.append(int(t))
bands = [x for x in args.bands.split(",")]


combinations = product(fields, tiles, bands)
combinations = [x for x in combinations] # don't want generator here
print(len(combinations))
kwargs = [{"n_cpus": 16} for _ in combinations]

base_dir = paths.runner_path / f"runs/{args.run_name}"
base_dir.mkdir(exist_ok=True, parents=True)

python_script_path = Path(args.python_script).absolute()
script_maker = ScriptMaker(python_script_path, base_dir=base_dir)

script_maker.write_all_scripts(combinations, kwargs)
    

