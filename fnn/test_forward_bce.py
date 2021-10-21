# Test y_hat, BCE loss, and the gradient of the loss w.r.t y_hat
# be careful with 0 divisions when calculating BCE
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP, bce_loss, mse_loss

# manual network
num_features = [2, 70, 5]
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
y = (torch.randn(batch_size, num_features[2]) < 0.5) * 1.0

net.clear_grad_and_cache()
y_hat = net.forward(x)
J, dJdy_hat = bce_loss(y, y_hat)

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
bce_torch = nn.BCELoss()
J_torch = bce_torch(y_hat_torch, y)
J_torch.backward()

# comparing (these all need to print True)
print((y_hat - y_hat_torch).norm() < 1e-3)
print((J - J_torch).norm() < 1e-3)
print((dJdy_hat - y_hat_torch.grad).norm() < 1e-3)