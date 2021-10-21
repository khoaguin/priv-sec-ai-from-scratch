# Test y_hat, mse loss, and the gradient of mse loss w.r.t y_hat
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP, mse_loss

net = MLP(
    linear_1_in_features=2,
    linear_1_out_features=20,
    f_function='relu',
    linear_2_in_features=20,
    linear_2_out_features=5,
    g_function='identity'
)
x = torch.randn(10, 2, requires_grad=True)
y = torch.randn(10, 5, requires_grad=True)

net.clear_grad_and_cache()
y_hat = net.forward(x.detach())
J, dJdy_hat = mse_loss(y.detach(), y_hat)

net_autograd = nn.Sequential(
    OrderedDict([
        ('linear1', nn.Linear(2, 20)),
        ('relu', nn.ReLU()),
        ('linear2', nn.Linear(20, 5)),
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

print((y_hat - y_hat_torch).norm() < 1e-3)
print((J - J_torch).norm() < 1e-3)
print((dJdy_hat - y_hat_torch.grad).norm() < 1e-3)