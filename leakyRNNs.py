import torch
from torch import nn
from leakycell import RNNLeakyCell
from typing import Callable


class LeakyRNN:
    """A basic leaky RNN trainable with backpropagation

    Arguments
    ---------
    input_size: int - the number of expected features in the input
    hidden_size: int - the number of features in the hidden state
    output_size: int - the number of features in the output layer
    tau: float in the interval [1, inf[ - the time constant of the leaky units, the inverse of the leak rate
    spectral_radius: float in interval ]0, inf[ - the maximum absolute eigen value of the recurrent
        weight matrix
    input_weights: 2D torch.tensor<hidden_size, input_size>, optional - a matrix containing the input weights
    hidden_weights: 2D torch.tensor<hidden_size, hidden_size>, optional - a matrix containing the hidden weights
    actf: activation function - the nonlinear activation function of the leaky cell
    actfout: activation function - the nonlinear activation function of the output layer
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 100,
        output_size: int = 2,
        tau: float = 4,
        spectral_radius: float = 0.9,
        input_weights: torch.tensor = None,
        hidden_weights: torch.tensor = None,
        actf: Callable[[torch.tensor], torch.tensor] = torch.tanh,
        actfout: Callable[[torch.tensor], torch.tensor] = torch.sigmoid,
    ):
        self.rnn = RNNLeakyCell(
            input_size=input_size,
            hidden_size=hidden_size,
            tau=tau,
            hidden_weights=hidden_weights,
            input_weights=input_weights,
            spectral_radius=spectral_radius,
            actf=actf,
        )
        self.readout = nn.Linear(hidden_size, output_size)
        self.actfout = actfout

    def forward(self, input, h=None):
        hidden_out, h = self.rnn(input, h)
        readout = self.actfout(self.readout(hidden_out))
        return readout, hidden_out.detach(), h.detach()


class Reservoir(RNNLeakyCell):
    """An implementation of a reservoir with analog units

    Arguments
    ---------
    *args, **kwargs: see LeakyRNN arguments
    readout_size: int - the number of features of the readout layer
    actfout: activation function - the activation function of the readout layer
    """

    def __init__(
        self,
        *args,
        readout_size: int = 1,
        actfout: Callable[[torch.tensor], torch.tensor] = torch.sigmoid,
        **kwargs
    ):
        super(Reservoir, self).__init__(*args, bias=False, **kwargs)
        self.weight_ih.requires_grad_(False)
        self.weight_hh.requires_grad_(False)
        self.readout_size = readout_size
        self.readout = nn.Linear(self.hidden_size, self.readout_size, bias=True)
        self.readout.weight.requires_grad_(False)
        self.readout.bias.requires_grad_(False)
        self.actfout = actfout

    def train(self, input, target, washout=0, ridge=None):
        """Training of the reservoir with paired input and target

        Arguments
        ---------
        input: 2D torch.tensor<# time steps, # bacthes, # input features>
        target: 2D torch.tensor<# time steps, # batches, # output features>
        washout: int - if non zero, the first 'washout' time steps will be ignored during training
        ridge: float - the parameter of the ridge L2 regularization
        """
        with torch.no_grad():
            ret, hx = super(Reservoir, self).forward(input)
        ret, target = ret[washout:], target[washout:]
        ret = torch.cat(
            (ret, torch.ones((ret.size(0), ret.shape[1], 1), device=input.device)), 2
        )
        ret = ret.reshape((ret.shape[0] * ret.shape[1], ret.shape[2]))
        target = target.reshape((target.shape[0] * target.shape[1], target.shape[2]))
        try:
            if ridge is None:
                inv_xTx = torch.pinverse(ret.T @ ret)
            else:
                I = torch.eye(self.hidden_size + 1)
                I[-1, -1] = 0
                inv_xTx = torch.pinverse(ret.T @ ret + ridge * I)
        except (RuntimeError):
            print("Issue with matrix inversion")
            return None
        xTy = ret.T @ target
        new_weights = (inv_xTx @ xTy).T
        self.readout.weight = nn.Parameter(new_weights[:, :-1], requires_grad=False)
        self.readout.bias = nn.Parameter(new_weights[:, -1], requires_grad=False)

    def forward(self, input, hx=None):
        ret, hx = super(Reservoir, self).forward(input, hx=hx)
        readout = self.actfout(self.readout(ret))
        return readout, ret.detach(), hx.detach()
