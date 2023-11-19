# Copyright (c) 2023, ISCS, Wenjie Zhang.

from dataclasses import dataclass
from typing import Callable, Any

import torch
import torch.nn.functional as F

from dsmeasure.core.abstract_device import AbstractDeviceConfig, AbstractDevice

@dataclass
class DeviceCUDAConfig(AbstractDeviceConfig):
    """
    memory_max_capacity: CUDA HBM size max (MB)
    memory_limit_capacity: CUDA HBM size limited usage (MB)
    """
    memory_max_capacity: int = 0
    memory_limit_capacity: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.is_computational = True
        self.is_transferatble = False

class DeviceCUDA(AbstractDevice):
    """
    """
    def __init__(self, config: DeviceCUDAConfig) -> None:
        super().__init__()
        self.config = config

        self.memory_used: int = int(0)
        self.computational_job_run = False
        self.computational_job: tuple = None
        self.non_computational_queue: list[tuple] = []
    """
    occupy gpu(computational/non-computational jobs):
        run_time: time to run estimated of job
        callback: callback after job done, call automatically after job finishes
        memory: memory cost estimated of job
        computational: if job is computational, computational jobs will exclusive
            other computational jobs, memory will release automatically; if job is non-computational, they will done
            simultaneously, and memory won't release automatically.
    return: (bool, int, int)
        is_success
        memory_used
        memory_available(base on limited memory)
    """
    def occupy(self, run_time: int, callback: Callable[..., Any], **kwargs):
        if kwargs['computational'] == True:
            if self.computational_job_run == True:
                return False
            memory_cost = kwargs['memory']
            self.memory_used += int(memory_cost)
            self.computational_job = (run_time, memory_cost, callback)
            self.computational_job_run = True
        else:
            memory_cost = kwargs['memory']
            self.memory_used += int(memory_cost)
            self.non_computational_queue.append(run_time, memory_cost, callback)

        assert self.memory_used <= self.config.memory_max_capacity, \
            "Error: CUDA Out-Of-Memory, expect {expect} MB, capacity is {capacity} MB".format(
                expect=self.memory_used, capacity=self.config.memory_max_capacity)

        return True, self.memory_used, self.config.memory_limit_capacity - self.memory_used
    """
    interval:
    return: (int, int)
        memory_used
        memory_available(base on limited memory)
    """
    def run(self, interval: int):
        # computational jobs
        if self.computational_job_run == True:
            self.computational_job[0] -= interval
            if self.computational_job[0] <= 0:
                self.memory_used -= self.computational_job[1]
                if self.computational_job[2] is not None:
                    self.computational_job[2]()
        # non-computational jobs
        _non_computational_queue_new = []
        for i in range(len(self.non_computational_queue)):
            self.non_computational_queue[i][0] -= interval
            if self.non_computational_queue[i][0] > 0:
                _non_computational_queue_new.append(self.non_computational_queue[i])
            elif self.non_computational_queue[i] is not None:
                self.non_computational_queue[i][2]()
        self.non_computational_queue = _non_computational_queue_new

        return self.memory_used, self.config.memory_limit_capacity - self.memory_used
