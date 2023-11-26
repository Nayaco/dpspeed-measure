import torch
import torch.nn as nn
m = nn.Dropout(p=0.2, inplace=False)
input = torch.randn(20, 16)
output = m(input)
print(input.size(), output.size())