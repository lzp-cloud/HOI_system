#!/bin/bash
cd ../a-PyTorch-Tutorial-to-Object-Detection
eval "$(conda shell.bash hook)"
conda activate vsgnet
nohup python -u detect.py 

