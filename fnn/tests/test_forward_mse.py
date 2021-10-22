# Test y_hat, mse loss, and the gradient of mse loss w.r.t y_hat
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import pathlib
import sys
_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
from mlp import MLP, mse_loss, bce_loss

# manual network
num_features = [100, 200, 10]
batch_size = 10
net = MLP(
    linear_1_in_features=num_features[0],
    linear_1_out_features=num_features[1],
    f_function='relu',
    linear_2_in_features=num_features[1],
    linear_2_out_features=num_features[2],
    g_function='sigmoid'
)
x = torch.randn(batch_size, num_features[0])
y = torch.randn(batch_size, num_features[2])

net.clear_grad_and_cache()
y_hat = net.forward(x.detach())
J, dJdy_hat = mse_loss(y.detach(), y_hat)

# pytorch network
net_autograd = nn.Sequential(
    OrderedDict([
        ('linear1', nn.Linear(num_features[0], num_features[1])),
        ('relu', nn.ReLU()),
        ('linear2', nn.Linear(num_features[1], num_features[2])),
        ('sigmoid', nn.Sigmoid())
    ])
)

net_autograd.linear1.weight.data = net.parameters['W1']
net_autograd.linear1.bias.data = net.parameters['b1']
net_autograd.linear2.weight.data = net.parameters['W2']
net_autograd.linear2.bias.data = net.parameters['b2']

y_hat_torch = net_autograd(x)
y_hat_torch.retain_grad()
mse_torch = nn.MSELoss()
J_torch = mse_torch(y_hat_torch, y)
J_torch.backward()

# comparing (these all need to print True)
print((y_hat - y_hat_torch).norm() < 1e-3)
print((J - J_torch).norm() < 1e-3)
print((dJdy_hat - y_hat_torch.grad).norm() < 1e-3)