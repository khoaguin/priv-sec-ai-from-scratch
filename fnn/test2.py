from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP, mse_loss, bce_loss

num_features = [30, 150, 40]
batch_size = 10
net = MLP(
    linear_1_in_features=num_features[0],
    linear_1_out_features=num_features[1],
    f_function='sigmoid',
    linear_2_in_features=num_features[1],
    linear_2_out_features=num_features[2],
    g_function='sigmoid'
)
x = torch.randn(batch_size, num_features[0])
y = (torch.randn(batch_size, num_features[2]) < 0.5) * 1.0

net.clear_grad_and_cache()
y_hat = net.forward(x)
J, dJdy_hat = bce_loss(y, y_hat)
net.backward(dJdy_hat)

#------------------------------------------------
# compare the result with autograd
net_autograd = nn.Sequential(
    OrderedDict([
        ('linear1', nn.Linear(num_features[0], num_features[1])),
        ('sigmoid1', nn.Sigmoid()),
        ('linear2', nn.Linear(num_features[1], num_features[2])),
        ('sigmoid2', nn.Sigmoid()),
    ])
)
net_autograd.linear1.weight.data = net.parameters['W1']
net_autograd.linear1.bias.data = net.parameters['b1']
net_autograd.linear2.weight.data = net.parameters['W2']
net_autograd.linear2.bias.data = net.parameters['b2']

y_hat_autograd = net_autograd(x)

bce = torch.nn.BCELoss()
J_autograd = bce(y_hat_autograd, y)

net_autograd.zero_grad()
J_autograd.backward()

print((net_autograd.linear1.weight.grad.data - net.grads['dJdW1']).norm() < 1e-3)
print((net_autograd.linear1.bias.grad.data - net.grads['dJdb1']).norm() < 1e-3)
print((net_autograd.linear2.weight.grad.data - net.grads['dJdW2']).norm() < 1e-3)
print((net_autograd.linear2.bias.grad.data - net.grads['dJdb2']).norm()< 1e-3)
#------------------------------------------------
