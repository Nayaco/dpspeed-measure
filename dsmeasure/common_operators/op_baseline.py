# Copyright (c) 2023, ISCS, Wenjie Zhang.

import os
import json
import torch

BASELINE_CONFIG = os.path.join(os.path.dirname(__file__), 'baseline_config.json')
b_config = None
with open(BASELINE_CONFIG, 'r') as f:
    b_config = json.load(f)
    
def get_baseline(operator_name: str) -> tuple(int, list[torch.Tensor]):
    """
    get baseline time
    """
    return b_config[operator_name]['time'], b_config[operator_name]['tensor_shape']