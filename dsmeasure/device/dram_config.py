# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_device import AbstractDeviceConfig

@dataclass
class DeviceDRAMConfig(AbstractDeviceConfig):
    """
    """
    memory_max_capacity: int = 0
    memory_limit_capacity: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.is_computational = False
        self.is_transferatble = False