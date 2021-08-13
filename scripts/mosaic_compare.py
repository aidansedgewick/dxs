import sys
from argparse import ArgumentParser
from pathlib import Path

from dxs.utils.image import mosaic_compare

parser = ArgumentParser()
parser.add_argument("path1")
parser.add_argument("path2")
parser.add_argument("-d", "--diff", default=False, action="store_true")
parser.add_argument("-q", "--quot", default=False, action="store_true")

args = parser.parse_args()

if args.diff is False and args.quot is False:
    print("add flag \"-d\" for difference, and/or \"-q\" for quotient. Exiting.")
    sys.exit()
try:
    p1 = Path(args.path1).relative_to(paths.base_path)
    p2 = Path(args.path2).relative_to(paths.base_path)
    print("compare:\n    {p1}\n    {p2}")
except:
    pass

if args.diff:
    mosaic_compare(Path(args.path1), Path(args.path2), func="diff")
if args.quot:
    mosaic_compare(Path(args.path1), Path(args.path2), func="quot")
