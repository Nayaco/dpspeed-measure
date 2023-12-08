# Copyright (c) 2023, ISCS, Wenjie Zhang.

import numpy as np
import torch

from dsmeasure2.core.dsm_device_mng import DeviceManager
from dsmeasure2.core.dsm_operator_mng import OperatorManager

from dsmeasure2.core.dsm_tensor import AbstractTensor
from dsmeasure2.core.dsm_device import AbstractDeviceConfig, AbstractDevice
from dsmeasure2.core.dsm_operator import AbstractOperatorConfig, \
                                         AbstractOperator, \
                                         OperatorComputationalConfig, \
                                         OperatorNonComputationalConfig , \
                                         OperatorCustomConfig , \
                                         OpStaticComputational , \
                                         OpStaticNonComputational, \
                                         OpStaticDerivative

from dsmeasure2.device.device_cuda import DeviceCUDA, DeviceCUDAConfig
from dsmeasure2.device.device_pcie import DevicePCIE4, DevicePCIEConfig

from dsmeasure2.core.dsm_tensor_mng import TensorManager
from dsmeasure2.graph.tensor_define import ActivationTensor, WeightTensor, TensorState

from dsmeasure2.flatten.flatten_operator import FlattenOperator, FlattenInitiate

# class FlattenDrop()