# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_device import AbstractDeviceConfig

@dataclass
class DeviceEthernetConfig(AbstractDeviceConfig):
    """
    """
    ethernet_nic_capacity: int = 0
    ethernet_nic_latency: float = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.is_computational = False
        self.is_transferatble = True

@dataclass
class DeviceInfinibandConfig(AbstractDeviceConfig):
    """
    """
    infiniband_nic_capacity: int = 0
    infiniband_nic_latency: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.is_computational = False
        self.is_transferatble = True

