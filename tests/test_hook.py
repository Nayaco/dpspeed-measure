import torch

storage = []

def pack(x):
    storage.append(x.cpu())
    print('pack:', len(storage) - 1)
    return len(storage) - 1

def unpack(x):
    print('unpack:', x)
    return storage[x].cuda()

x = torch.randn(1024, requires_grad=True, device='cuda')
y = torch.randn(1024, requires_grad=True, device='cuda')
with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    z = torch.square(x * y)
z.sum().backward()