#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate torch_2_0_1
# python allreduce_vs_matmul.py
python allreduce_vs_matmul.py --overlap