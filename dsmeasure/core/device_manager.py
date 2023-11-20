# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, Any

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_device import AbstractDeviceConfig, AbstractDevice
from dsmeasure.device.gpu import DeviceCUDAConfig, DeviceCUDA
from dsmeasure.device.pcie4 import DevicePCIEConfig, DevicePCIE4

def gen_device_uid() -> int:
    device_uid: int = int(0)
    while True:
        yield device_uid
        device_uid += 1

class DeviceManager:
    def __init__(self) -> None:
        self.devices: dict[int, AbstractDevice] = {}
        self.cuda: list[int] = []
        self.pcie: list[int] = []

    def register(self, config: AbstractDeviceConfig) -> tuple(int, AbstractDevice):
        """
        register device:
            config: device config
        return: (device_uid, device)
        """
        new_device_uid = gen_device_uid()        
        if isinstance(config, DeviceCUDAConfig):
            self.devices[new_device_uid] = DeviceCUDA(config)
            self.devices[new_device_uid].config.device_uid = new_device_uid
            self.cuda.append(new_device_uid)
        if isinstance(config, DevicePCIEConfig):
            self.devices[new_device_uid] = DevicePCIE4(config)
            self.devices[new_device_uid].config.device_uid = new_device_uid
            self.pcie.append(new_device_uid)
        return new_device_uid, self.devices[new_device_uid]
    
    def find(self, device_uid: int) -> AbstractDevice:
        """
        find device:
            device_uid: device uid
        return: (device,)
        """
        return self.devices[device_uid]

def GetDeviceManager():
    _device_manager = DeviceManager()
    while True:
        yield _device_manager