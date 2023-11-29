# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, Any

import torch
import torch.nn.functional as F
import math

from dsmeasure2.core.dsm_device import AbstractDeviceConfig, AbstractDevice

@dataclass
class DevicePCIEConfig(AbstractDeviceConfig):
    """
    """
    pcie_bandwidth_source: AbstractDeviceConfig = None
    pcie_bandwidth_target: AbstractDeviceConfig = None
    
    pcie_bandwidth_p2p: float = 28543 # MB/s
    pcie_latency_p2p: float = 0 # us
    
    def __post_init__(self):
        super().__post_init__()
        self.is_computational = False
        self.is_transferatble = True

class DevicePCIE4(AbstractDevice):
    """
    config: DevicePCIEConfig
    """
    def __init__(self, config: DevicePCIEConfig) -> None:
        super().__init__()
        self.config = config

        self.tranfer_job_run = False
        self.tranfer_job: tuple = None
        
    def occupy(self, run_time: int, callback: Callable[..., Any], **kwargs) -> bool:
        """
        occupy pcie4():
            run_time: time to run estimated of job (-1 means calculate automatically)
            callback: callback after job done, call automatically after job finishes
            dsize: data size to transfer(if run_time is -1)
        return: (bool,)
            is_success
        """
        if self.tranfer_job_run == True:
            return False
        self.transfer_job = (
            run_time if run_time is not None and run_time > 0 else \
                math.ceil(kwargs['dsize'] / self.config.pcie_bandwidth_p2p + self.config.pcie_latency_p2p), 
            callback)
        self.transfer_job_run = True
        return True
    
    def run(self, interval: int) -> None:
        """
        interval:
        return: ()
        """
        # computational jobs
        if self.tranfer_job_run == True:
            self.tranfer_job = \
                (self.tranfer_job[0] - interval, 
                 self.tranfer_job[1])
            if self.tranfer_job[0] <= 0:
                self.transfer_job_run = False
                if self.tranfer_job[1] is not None:
                    self.tranfer_job[1]()
    
    def try_occupy(self, run_time: int, **kwargs):
        return not self.tranfer_job_run