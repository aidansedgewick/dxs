from argparse import ArgumentParser
from pathlib import Path

from dxs.utils.image import mosaic_difference

parser = ArgumentParser()
parser.add_argument("path1")
parser.add_argument("path2")

args = parser.parse_args()

mosaic_difference(Path(args.path1), Path(args.path2))
