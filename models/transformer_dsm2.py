# Copyright (c) 2023, ISCS, Wenjie Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

# miscellaneous
import builtins
import datetime
import json
import sys
import time
from typing import Tuple

import warnings

# numpy
import numpy as np
import sklearn.metrics

# pytorch
import torch
from torch import Tensor

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

from dsmeasure2.core.dsm_device_mng import DeviceManager
from dsmeasure2.device.device_cuda import DeviceCUDA
from dsmeasure2.device.device_pcie import DevicePCIE4

from dsmeasure2.core.dsm_tensor_mng import TensorManager
from dsmeasure2.graph.tensor_define import ActivationTensor, WeightTensor, TensorState
from dsmeasure2.graph.operator_graph import UnaryOperator, BinaryOperator


class GPT2Model(AbstractOperator):
    def __init__(self, 
                 config: OperatorCustomConfig,
                 compute_time_embedding: int,
                 compute_time_embedding_backward: int,
                 
                 compute_time_att_linear_qkv: int,
                 compute_time_att_matmul_kq: int, 
                 compute_time_att_sm: int,
                 compute_time_att_attention_dropout: int,
                 compute_time_att_matmul_v: int,
                 compute_time_att_linear: int,
                 compute_time_att_allreduce: int,
                 compute_time_att_dropout: int,

                 compute_time_att_linear_qkv_backward: int,
                 compute_time_att_matmul_kq_backward: int,
                 compute_time_att_sm_backward: int,
                 compute_time_att_attention_dropout_backward: int,
                 compute_time_att_matmul_v_backward: int,
                 compute_time_att_linear_backward: int,
                 compute_time_att_dropout_backward: int,
                 
                 compute_time_ffn_layer1: int,
                 compute_time_ffn_gelu : int,
                 compute_time_ffn_layer2: int,
                 compute_time_ffn_allreduce: int,
                 compute_time_ffn_dropout: int,
                 
                 compute_time_ffn_layer1_backward: int,
                 compute_time_ffn_gelu_backward : int,
                 compute_time_ffn_layer2_backward: int,
                 compute_time_ffn_dropout_backward: int,

                 compute_time_layernorm_1: int,
                 compute_time_layernorm_2: int,
                 compute_time_layernorm_1_backward: int,
                 compute_time_layernorm_2_backward: int,
                 
                 compute_time_residual_add_1: int,
                 compute_time_residual_add_2: int,
                 compute_time_residual_add_1_backward: int,
                 compute_time_residual_add_2_backward: int,
                 
                 compute_time_layernorm_final: int,
                 compute_time_layernorm_final_backward: int,

                 batch_size: int,
                 seq_len: int,
                 head_num: int,
                 head_hidden_size: int,
                 hidden_size: int,
                 vocab_size: int,
                 tensor_parallel: int|None = None,
                 precision: int = 2,
                 transformer_layers: int = 1):
        super().__init__(config)
        self.transformer_layers = transformer_layers

        tensor_parallel = 1 if tensor_parallel is None else tensor_parallel
        self.embedding = UnaryLayer(OperatorCustomConfig(
            op_name="embedding"),
            compute_time_embedding,
            compute_time_embedding_backward,
            0,
            precision * int(batch_size*seq_len*hidden_size))
        
        self.transformer_blocks = [TransformerBlockGPT(OperatorCustomConfig(
            op_name="transformer_block_%d" % i),
            compute_time_att_linear_qkv,
            compute_time_att_matmul_kq, 
            compute_time_att_sm,
            compute_time_att_attention_dropout,
            compute_time_att_matmul_v,
            compute_time_att_linear,
            compute_time_att_allreduce,
            compute_time_att_dropout,

            compute_time_att_linear_qkv_backward,
            compute_time_att_matmul_kq_backward,
            compute_time_att_sm_backward,
            compute_time_att_attention_dropout_backward,
            compute_time_att_matmul_v_backward,
            compute_time_att_linear_backward,
            compute_time_att_dropout_backward,

            compute_time_ffn_layer1,
            compute_time_ffn_gelu,
            compute_time_ffn_layer2,
            compute_time_ffn_allreduce,
            compute_time_ffn_dropout,
            
            compute_time_ffn_layer1_backward,
            compute_time_ffn_gelu_backward,
            compute_time_ffn_layer2_backward,
            compute_time_ffn_dropout_backward,

            compute_time_layernorm_1,
            compute_time_layernorm_2,
            compute_time_layernorm_1_backward,
            compute_time_layernorm_2_backward,
            
            compute_time_residual_add_1,
            compute_time_residual_add_2,
            compute_time_residual_add_1_backward,
            compute_time_residual_add_2_backward,

            batch_size,
            seq_len,
            head_num,
            head_hidden_size,
            hidden_size,

            tensor_parallel,
            precision) for i in range(transformer_layers)]
        
        self.layernorm_final = UnaryLayer(OperatorCustomConfig(
            op_name="layernorm_final"),
            compute_time_layernorm_final,
            compute_time_layernorm_final_backward,
            precision * int(batch_size*seq_len*hidden_size),
            precision * int(batch_size*seq_len*hidden_size))
        
        self.loss = UnaryLayer(OperatorCustomConfig(
            op_name="loss"),
            1,
            1,
            precision * int(batch_size*seq_len*hidden_size),
            precision * int(batch_size*seq_len*hidden_size))
        

        self._subop = self.embedding

