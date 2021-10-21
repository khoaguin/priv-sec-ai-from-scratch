from copy import deepcopy
from typing import Tuple

import torch
from torch.functional import Tensor
from torch import Tensor

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = ActivFunc(f_function)
        self.g_function = ActivFunc(g_function)

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def linear(self, x, W, b) -> Tuple[Tensor, Tensor, Tensor]:
        """The linear layer that produces the affine transformation
            y = xW' + b
        Args:
            x (Tensor): the input tensor with shape (batch_size, x_features)
            W (Tensor): the weight tensor with shape (out_features, x_features)
            b (Tensor): the bias tensor with shape (out_features)
        Returns:
            y (Tensor): the output tensor of the linear layer with shape (batch_size, out_features)
            dydW (Tensor): the derivative of the linear function w.r.t the weights W
                            shape ()
        """
        y = torch.matmul(x, W.T) + b
        dydW = x
        dydx = W

        return y, dydW, dydx

    def forward(self, x) -> Tensor:
        """[summary]

        Args:
            x (Tensor): shape (batch_size, linear_1_in_features)
        Returns:
            y_hat (Tensor): shape (batch_size, linear_2_out_features)
        """
        # z1: (batch_size, linear_1_out_features)
        z1, dz1dW1, _ = self.linear(x, self.parameters["W1"], self.parameters["b1"]) 
        self.cache["dz1dW1"] = dz1dW1
        
        z2, dz2dz1 = self.f_function(z1)
        self.cache["dz2dz1"] = dz2dz1
        self.cache["z2"] = z2

        # z3: (batch_size, linear_2_out_features)
        z3, dz3dW2, dz3dz2 = self.linear(z2, self.parameters["W2"], self.parameters["b2"])
        self.cache["dz3dW2"] = dz3dW2
        self.cache["dz3dz2"] = dz3dz2
        
        # y_hat: (batch_size, linear_2_out_features)
        y_hat, dy_hatdz3 = self.g_function(z3)  
        self.cache["dy_hatdz3"] = dy_hatdz3

        return y_hat

    def backward(self, dJdy_hat) -> None:
        """
        Calculates the gradients of the loss w.r.t the weights and biases
        dJdW1, dJdb1, dJdW2, dJdb2 in the self.grads dict

        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        dJdz3 = dJdy_hat * self.cache["dy_hatdz3"]  # * is element-wise multiplication
        self.grads["dJdb2"] = dJdz3.sum(0)  # sum accross all batches
        self.grads["dJdW2"] = torch.matmul(dJdz3.T, self.cache["dz3dW2"])
        dJdz1 = torch.matmul(dJdz3, self.cache["dz3dz2"]) * self.cache["dz2dz1"]
        self.grads["dJdb1"] = dJdz1.sum(0)
        self.grads["dJdW1"] = torch.matmul(dJdz1.T, self.cache["dz1dW1"])

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()


class ActivFunc:
    def __init__(self, function_name: str) -> None:
        """
        Args:
            function_name (str): the name of the activation function
                                 can be relu | sigmoid | identity
        """
        if function_name not in ["relu", "sigmoid", "identity"]:
            raise("function needs to be either 'relu', 'sigmoid', or 'identity'")
        self.func = function_name

    def __call__(self, x):
        """
        Args:
            x (Tensor): the input to the activation function
                        shape (batch_size, x_features)
        Return:
            y (Tensor): the output of the activation function
                        shape (batch_size, x_features)
            dydx (Tensor): the derivative of the activation function, same shape with
                        the input: (batch_size, x_features)
        """
        if self.func == "sigmoid":
            y = 1 / (1 + torch.exp(-x))
            dydx = y*(1-y)

        elif self.func == "relu":
            y = deepcopy(x)
            y[y<0] = 0
            dydx = deepcopy(x)
            dydx[dydx<0] = 0
            dydx[dydx>0] = 1
        
        else:  # "identity"
            y = x
            dydx = torch.ones(x.shape)

        assert y.shape == x.shape, "input and output tensors have different shapes!"
        assert dydx.shape == x.shape, \
                "the derivative needs to have the same shape with input"

        return y, dydx


def mse_loss(y, y_hat):
    """
    Computes the MSE loss from 2 tensors and also the gradient of 
    the loss with respect to the predicted outputs y_hat

    Args:
        y: the label tensor, shape: (batch_size, linear_2_out_features)
        y_hat: the prediction tensor, shape: (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # calculate the MSE loss
    square_diff = torch.square(y_hat - y)
    total_features = len(y.view(-1))
    loss = 1/total_features * square_diff.sum()

    # calculate the derivative of the loss w.r.t y_hat
    dJdy_hat = 1/total_features * 2*(y_hat-y)

    assert dJdy_hat.shape == y_hat.shape, "The gradient needs to have the same shape with y_hat"

    return loss, dJdy_hat


def bce_loss(y, y_hat):
    """
    Computes the BCE loss from 2 tensors and also the gradient of 
    the loss with respect to the predicted outputs y_hat

    Args:
        y_hat: the prediction tensor, shape: (batch_size, linear_2_out_features)
        y: the label tensor, shape: (batch_size, linear_2_out_features)
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # calculate BCE loss
    epsilon = 1e-20  # to avoid inf/-inf
    y_hat = torch.clamp(y_hat, min=epsilon, max=1-epsilon)
    total_features = len(y.view(-1))
    term0 = y * torch.log(y_hat+epsilon)
    term1= (1-y) * torch.log(1-y_hat+epsilon)
    loss = -1/total_features *  (term0 + term1).sum()
    # calculate the derivative of the loss w.r.t y_hat
    dJdy_hat = 1/total_features * (y_hat-y)/(y_hat*(1-y_hat)+epsilon)

    assert dJdy_hat.shape == y_hat.shape, "The gradient needs to have the same shape with y_hat"

    return loss, dJdy_hat
    

