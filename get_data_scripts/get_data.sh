#!/bin/bash
[[ -d ../input_data/stacks ]] || -p mkdir ../input_data/stacks
[[ -d ../input_data/stacks
cp ./dxs.wget ../input_data/stacks
( cd ../input_data/stacks ; ./dxs.wget )
