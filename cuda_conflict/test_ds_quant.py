import deepspeed.ops.quantizer as _quantizer
from deepspeed.ops import op_builder
import torch

# tensor_0 = torch.rand(1024, dtype=torch.float16, device='cuda:0')

# tensor_1 = _quantizer.ds_quantizer(tensor_0, bit_num=8)

# print(tensor_0)
# print(tensor_1)

tensor = torch.rand(1024*1024*1024*2, dtype=torch.float16, device='cuda:0')
quantizer_module = op_builder.QuantizerBuilder().load()
for i in range(6000000):
    intra_quant_int4, intra_q_scales = quantizer_module.swizzle_quant(tensor, 1, 4,
                                                                            quantizer_module.Symmetric, 1, 1,
                                                                            1)
# print(tensor)
# print(intra_quant_int4, intra_quant_int4.size())
# print(intra_q_scales)
