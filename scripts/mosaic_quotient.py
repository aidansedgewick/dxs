from argparse import ArgumentParser
from pathlib import Path

from dxs.utils.image import mosaic_quotient

parser = ArgumentParser()
parser.add_argument("path1")
parser.add_argument("path2")

args = parser.parse_args()

mosaic_quotient(Path(args.path1), Path(args.path2))
