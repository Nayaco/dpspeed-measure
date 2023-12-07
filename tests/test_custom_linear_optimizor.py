import torch
import torch.nn as nn
from torch.autograd import Function
import math
import numpy as np

saved_inputs = []
saved_gradout = []

# Inherit from Function
class LinearFunction(Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight, bias):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            # print(grad_weight.size())
            # print(grad_output.size(), input.size())
            # grad_weight = None # torch.cat((grad_output, input), 1)
            # saved_gradout.append(grad_output)
            # saved_inputs.append(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class LinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        return LinearFunction.apply(x, self.weights, self.bias)# torch.add(torch.mm(x, self.weights.t()), self.bias)  # w times x + b
    
torch.manual_seed(0)  #  for repeatable results
lin_1 = LinearLayer(65536, 1024).cuda(0)
lin_2 = LinearLayer(1024, 1).cuda(0)
print(torch.cuda.memory_allocated()/1024)
model_size = torch.cuda.memory_allocated()/1024
x = torch.ones(10, 65536, dtype=torch.float, device='cuda:0')
print(torch.cuda.memory_allocated()/1024)
y = lin_1(x)
torch.cuda.synchronize()
print(torch.cuda.memory_allocated()/1024)
y = lin_2(y)
torch.cuda.synchronize()
loss = torch.mean(y)
print('Forward computation thru model:', loss)
print(torch.cuda.memory_allocated()/1024)
loss.backward()
print(torch.cuda.memory_allocated()/1024)