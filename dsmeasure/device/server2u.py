# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_device import AbstractDeviceConfig
from dsmeasure.core.device_manager import DeviceManager

from dsmeasure.device.pcie4 import DevicePCIEConfig
from dsmeasure.device.gpu import DeviceCUDAConfig


class Server2U_1Side:
    """
    server 2u:
    CUDA-0          CUDA-1
    PCIE \---NUMA---| PCIE
    DRAM \----------| DRAM
    """
    def __init__(self, 
                 pcie_config_numa0: DevicePCIEConfig, 
                 cuda_config_cuda0: DeviceCUDAConfig) -> None:
        self.pcie_numa0,_ = DeviceManager().register(pcie_config_numa0)
        self.cuda_numa0,_ = DeviceManager().register(cuda_config_cuda0)

class Server2U_2Side:
    """
    server 2u:
    CUDA-0          CUDA-1
    PCIE \---NUMA---| PCIE
    DRAM \----------| DRAM
    """
    def __init__(self, 
                pcie_config_numa0: DevicePCIEConfig, 
                pcie_config_numa1: DevicePCIEConfig,
                ) -> None:
        pass