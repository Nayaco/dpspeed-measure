#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate torch_2_0_1
python test_gpu2cpu_transfer.py