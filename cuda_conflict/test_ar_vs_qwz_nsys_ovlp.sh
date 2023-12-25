#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate torch_2_0_1
. /usr/local/cuda-11.6/setup.sh
python allreduce_vs_qwz.py --overlap
# python allreduce_vs_matmul.py --overlap