import sys
from argparse import ArgumentParser
from pathlib import Path

from dxs.utils.image import mosaic_quotient, mosaic_difference

parser = ArgumentParser()
parser.add_argument("path1")
parser.add_argument("path2")
parser.add_argument("-d", "--diff", default=False, action="store_true")
parser.add_arguemnt("-q", "--quot", default=False, action="store_true")

args = parser.parse_args()

if args.diff is False and args.quot is False:
    print("add flag \"-d\" for difference, and/or \"-q\" for quotient. Exiting.")
    sys.exit()

if args.diff:
    mosaic_difference(Path(args.path1), Path(args.path2))
if args.quot:
    mosaic_quotient(Path(args.path1), Path(args.path2))
