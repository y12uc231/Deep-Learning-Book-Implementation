from __future__ import print_function
import os
import numpy as np
import torch
import torch.functional as F
import torch.optim as optim


## Basic Linear Algebra

torch_tensor = torch.ones([2,4,5])
print(torch_tensor)

# Transpose
print(torch_tensor.T.shape)
print(torch_tensor.float())
# Random tensor

random_tensor = torch.rand([3, 4,5])
print(random_tensor)
print(random_tensor.view(-1, 1, 1).shape)
print(random_tensor.dtype)
print(random_tensor.T.shape)


## Torch Numpy interaction
arr_np = np.zeros([3,5])
arr_torch = torch.from_numpy(arr_np)
print(arr_torch.shape)
back_to_np = arr_torch.numpy()
print(type(back_to_np))
print(arr_torch)

## Computing norms
print("Computing Norms and determinants")
arr_np = np.array([[2, 3], [1,2]])
arr_torch = torch.from_numpy(arr_np)
arr_torch = arr_torch.type(torch.float64)
rand_tensor = torch.ones([3,3])
print(rand_tensor)
print(arr_torch.det())



## GPU details
x = torch.rand([4,4])
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))