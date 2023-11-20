# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, Any

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_device import AbstractDeviceConfig, AbstractDevice
from dsmeasure.core.device_manager import gen_device_uid
from dsmeasure.device.gpu import DeviceCUDAConfig, DeviceCUDA

def cb():
    print('done')

cuda = DeviceCUDA(DeviceCUDAConfig(device_uid=gen_device_uid(), device_name="cuda:0", memory_max_capacity=120, memory_limit_capacity=100))
print(
    cuda.occupy(run_time=25, callback=cb, memory=100, computational=True)
)
print(
    cuda.occupy(run_time=15, callback=cb, memory=10, computational=False)
)
print(
    cuda.run(10)
)

print(
    cuda.run(10)
)


print(
    cuda.run(10)
)