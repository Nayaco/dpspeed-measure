import torch

X = torch.tensor([[ 2., 1., -3], 
                  [ -3, 4., 2.]], requires_grad=True)

W = torch.tensor([[ 3., 2., 1., -1], 
                  [ 2., 1., 3., 2.], 
                  [ 3., 2., 1., -2]], requires_grad=True)

# Z = torch.matmul(X, W)
# Z.retain_grad()
# Y = torch.exp(Z)
Y = torch.matmul(X, W)
dL_over_dy = torch.tensor([[ 2., 3., -3, 9.],
                           [ -8, 1., 4., 6.]])

Y.backward(dL_over_dy)

print(W.grad)
print(X.T @ dL_over_dy)
print(X.grad)
print(dL_over_dy @ W.T)