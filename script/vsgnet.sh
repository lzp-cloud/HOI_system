#!/bin/bash
cd ../VSGNet/scripts
eval "$(conda shell.bash hook)"
conda activate vsgnet
nohup python main.py -fw new_test1 -ba 1 -r t -i t -v_i test 
#nohup python -u pred_vis.py