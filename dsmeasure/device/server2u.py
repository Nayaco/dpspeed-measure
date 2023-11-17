# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_device_config import AbstractDeviceConfig

from .pcie_config import DevicePCIEConfig
from .gpu_config import DeviceCUDAConfig

class Server2U:
    def __init__(self, 
                 pcie_config_d2d: DevicePCIEConfig, 
                 pcie_config_d2h: DevicePCIEConfig, 
                 pcie_config_h2d, ) -> None:
        pass