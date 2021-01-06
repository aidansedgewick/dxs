#!/bin/bash
[[ -d ../data/ ]] || mkdir ../data
cp ./dxs.wget ../data
( cd ../data ; ./dxs.wget )
