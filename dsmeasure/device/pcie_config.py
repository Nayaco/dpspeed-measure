# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_device import AbstractDeviceConfig

@dataclass
class DevicePCIEConfig(AbstractDeviceConfig):
    """
    """
    pcie_bandwidth_source: AbstractDeviceConfig = None
    pcie_bandwidth_target: AbstractDeviceConfig = None
    
    pcie_bandwidth_s2t: float = 0
    pcie_latency_s2t: float = 0
    
    def __post_init__(self):
        pass