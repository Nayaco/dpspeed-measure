# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_device_config import AbstractDeviceConfig

@dataclass
class DeviceCUDAConfig(AbstractDeviceConfig):
    
    def __post_init__(self):
        pass