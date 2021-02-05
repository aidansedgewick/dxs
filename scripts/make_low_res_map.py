from itertools import product

from dxs import MosaicBuilder

combinations = product(["SA"], [x for x in range(10,13)], ["J", "K"])

combinations = [x for x in combinations]


for combo in combinations:
    builder = MosaicBuilder.coverage_from_dxs_spec(*combo, pixel_scale=2.0)
    builder.build(
        hdu_prefix=f"{builder.mosaic_path.stem}_u", n_cpus=6
    )


