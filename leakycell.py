import torch
from torch import nn
from helper import random_weights
from typing import Callable


class RNNLeakyCell(nn.RNNCellBase):
    '''A leaky analog unit for recurrent neural networks

    Arguments
    ---------
    input_size: int - the dimension of the input
    hidden_size: int - the dimension of the recurrent layer
    tau: float in the interval [1, inf[ - the time constant of the leaky units, the inverse of the leak rate
    weight_seed: int - the seed used to generate the weight matrices if they are not provided
    spectral_radius: float in interval ]0, inf[ - the maximum absolute eigen value of the recurrent
        weight matrix
    bias: bool - whether to include bias
    actf: activation function - the nonlinear activation function of the leaky cell
    input_weights: 2D torch.tensor<hidden_size, input_size>, optional - a matrix containing the input weights
    hidden_weights: 2D torch.tensor<hidden_size, hidden_size>, optional - a matrix containing the hidden weights
    '''
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size: int, hidden_size: int, tau: float, weight_seed: int = None,
                 spectral_radius: float = .9, bias: bool = True,
                 actf: Callable[[torch.tensor], torch.tensor] = torch.tanh, input_weights: torch.tensor = None,
                 hidden_weights: torch.tensor = None):
        # If input_weights or hidden_weights are provided, it overrides input_size and hidden_size respectively
        if input_weights is not None:
            input_size = input_weights.shape[1]
        if hidden_weights is not None:
            hidden_size = hidden_weights.shape[0]
        super(RNNLeakyCell, self).__init__(input_size, hidden_size, bias, num_chunks=1)
        self.leak_rate = 1./tau
        input_weights, hidden_weights = self.get_weights(input_weights, hidden_weights, spectral_radius)
        self.weight_hh = nn.Parameter(hidden_weights)
        self.weight_ih = nn.Parameter(input_weights)
        self.actf = actf

    def get_weights(self, input_weights, hidden_weights, spectral_radius, weight_seed=None):
        if hidden_weights is not None:
            hidden_weights = torch.Tensor(hidden_weights).float()
            assert hidden_weights.size(0) == hidden_weights.size(1)
        else:
            hidden_weights = self.init_hidden_weights(spectral_radius, seed=weight_seed)

        if input_weights is not None:
            input_weights = torch.Tensor(input_weights).float()
            assert input_weights.size(0) == self.hidden_size and input_weights.size(1) == self.input_size
        else:
            if weight_seed is not None:
                weight_seed += 1
            input_weights = self.init_input_weights(seed=weight_seed)
        return input_weights, hidden_weights

    def init_hidden_weights(self, spectral_radius, seed=None):
        return random_weights(input_size=self.hidden_size, output_size=self.hidden_size,
                                  spectral_radius=spectral_radius, seed=seed)

    def init_input_weights(self, seed=None):
        return random_weights(input_size=self.input_size, output_size=self.hidden_size,
                                 sparsity=.1, seed=seed)

    def forward(self, input, hx=None):
        if hx is None:
            hx = torch.zeros(input.size(1), self.hidden_size,
                             dtype=input.dtype, device=input.device)
        nsteps, nbatches, _ = input.shape
        ret = torch.empty((nsteps, nbatches, self.hidden_size), device=input.device)
        input_by_w = input @ self.weight_ih.T
        for istep in range(nsteps):
            self_ = hx @ self.weight_hh.T
            htmp = input_by_w[istep, :, :] + self_
            hx = self.actf((1 - self.leak_rate) * hx + self.leak_rate * htmp)
            ret[istep, :, :] = hx

        return ret, hx
