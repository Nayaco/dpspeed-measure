# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, Any
from functools import cache

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_device import AbstractDeviceConfig, AbstractDevice
from dsmeasure.device.gpu import DeviceCUDAConfig, DeviceCUDA
from dsmeasure.device.pcie4 import DevicePCIEConfig, DevicePCIE4

def gen_device_uid() -> int:
    device_uid: int = int(64) # from 0x40
    while True:
        yield device_uid
        device_uid += 1
IDGenerator = gen_device_uid()
@cache
class DeviceManager:
    def __init__(self) -> None:
        self.devices: dict[int, AbstractDevice] = {}
        self.cuda: dict[str, int] = {}
        self.pcie: dict[str, int] = {}

        self.cuda_count: int = 0
        self.pcie_count: int = 0

    def register(self, config: AbstractDeviceConfig) -> tuple[int, AbstractDevice]:
        """
        register device:
            config: device config
        return: (device_uid, device)
        """
        new_device_uid = next(IDGenerator)        
        if isinstance(config, DeviceCUDAConfig):
            self.devices[new_device_uid] = DeviceCUDA(config)
            self.devices[new_device_uid].config.device_uid = new_device_uid
            self.cuda[f'cuda:{self.cuda_count}'] = new_device_uid
            self.cuda_count += 1
        if isinstance(config, DevicePCIEConfig):
            self.devices[new_device_uid] = DevicePCIE4(config)
            self.devices[new_device_uid].config.device_uid = new_device_uid
            self.pcie[f'pcie:{self.pcie_count}'] = new_device_uid
            self.pcie_count += 1
        return new_device_uid, self.devices[new_device_uid]
    
    def find_by_name(self, dname: str) -> AbstractDevice:
        """
        find device:
            dname: device name
        return: (device,)
        """
        if dname.find('cuda') != -1:
            return self.devices[self.cuda[dname]]
        if dname.find('pcie') != -1:
            return self.devices[self.pcie[dname]]
        return None

    def find(self, device_uid: int) -> AbstractDevice:
        """
        find device:
            device_uid: device uid
        return: (device,)
        """
        return self.devices[device_uid]
    
    def __iter__(self):
        return iter(self.devices.values())