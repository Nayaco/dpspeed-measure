# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_device import AbstractDeviceConfig

from .pcie_config import DevicePCIEConfig
from .gpu import DeviceCUDAConfig

class Server2U:
    """
    server 2u:
    CUDA-0          CUDA-1
    PCIE |---NUMA---| PCIE
    DRAM

    """
    def __init__(self, 
                 pcie_config_d2d: DevicePCIEConfig, 
                 pcie_config_d2h: DevicePCIEConfig, 
                 pcie_config_h2d, ) -> None:
        pass