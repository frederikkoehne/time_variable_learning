import torch
from torch import nn
from math import gamma as mgamma

from typing import List, Sequence


class TauBlock(nn.Module):
    r"""
    Represents one residual block as discussed in the paper (https://arxiv.org/abs/2204.08528).
    """

    def __init__(
        self,
        layer: nn.Module,
        activation: nn.Module,
        initial_tau: float = None,
        add_shortcut: bool = True,
        tau_is_trainable=True
    ):
        r"""Constructor of residual block

        Args:
            activation (nn.Module): Activation function to use. Should be an instance of a torch Activation function, e.g., the result of nn.ReLU().
            layer (nn.Module): The weight (and bias) layer to be used for the forward pass.
            initial_tau (float, optional): Initial value for tau. If not specified, tau is randomly chosen from U(0,1).
            add_shortcut (bool, optional): When set to False, no skip connection will be used. Defaults to True.
            tau_is_trainable (bool, optional): Whether tau is trainable. Defaults to True.
        """
        super(TauBlock, self).__init__()
        self.layer = layer
        self.activation = activation

        self.add_shortcut = add_shortcut

        self.forward_block = nn.Sequential(
            self.layer,
            self.activation
        )

        if tau_is_trainable:
            if initial_tau:
                self.tau = nn.Parameter(torch.tensor(initial_tau))
            else:
                self.tau = nn.Parameter(torch.rand(1))
        else:
            if initial_tau:
                self.tau = torch.tensor(initial_tau)
            else:
                self.tau = torch.rand(1)
            self.tau.requires_grad = False

    def forward(self, x):
        r"""Forward pass of the residual block
        """
        identity = x
        out = self.forward_block(x)
        return self.tau * out + identity if self.add_shortcut else self.tau * out


class FractionalDNN(nn.Module):
    r"""
        Fractional DNN as discussed in (https://arxiv.org/abs/2204.08528).

    """

    def __init__(
            self,
            layers: Sequence[nn.Module],
            activations: Sequence[nn.Module],
            initial_taus: Sequence[float],
            tau_is_trainable: bool = True,
            gamma: float = .5,
            eps: float = 1e-8,
            project_taus_before_forward_pass: bool = False,
            device=None
    ):
        r"""Constructor for Fractional DNN.

        Args:
            layers (Sequence[nn.Module]): List or Tuple of the layers to be used. Layers must not contain the activation function.
            activations (Sequence[nn.Module]): List or Tuple of the activation functions to be used. Has to be the same length as layers.
            initial_taus (Sequence[float]): Initial values for taus. Has to be the same length as layers.
            tau_is_trainable (bool, optional): Whether (or which of) the tau values are considered trainable. Not yet implemented, any input is ignored. Defaults to True.
            gamma (float, optional): gamma hyperparameter for the Fractional DNN. Defaults to .5.
            eps (float, optional): Minimal value taus are allowed to obtain. Will be enforced in the project_taus method.
            project_taus_before_forward_pass (bool, optional): Whether to automatically project taus prior to the forward pass. Defaults to False.
        """

        super().__init__()
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        assert len(layers) == len(activations) == len(
            initial_taus), f'Length of layers, activations, and initial taus must be equal, but got: {len(layers)}, {len(activations)}, {len(initial_taus)}'

        self.gamma = torch.tensor(
            gamma, dtype=torch.float32, requires_grad=False)
        self.eps = torch.tensor(eps, device=self.device)
        self.project_taus_before_forward_pass = project_taus_before_forward_pass

        self.Gamma_func_eval = mgamma(2 - gamma)

        self.activations = activations
        self.layers = layers

        # Construct network using layers and activations provided
        for k, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            self.add_module(f'layer{k+1}', layer)
            self.add_module(f'activation{k+1}', activation)

        # Check if tau_is_trainable is boolean. If so, build a list by repeating it len(layers) times. If it is not boolean, it is expected to already be a list of the correct length
        if isinstance(tau_is_trainable, bool):
            tau_is_trainable = [tau_is_trainable] * len(layers)

        assert len(tau_is_trainable) == len(
            layers), "If 'tau_is_trainable' is not specified as a boolean, it is expected to be a list of length len(layers)"

        self.tau_is_trainable = tau_is_trainable
        self.initial_taus = initial_taus
        self.number_of_layer = len(layers)

        self.taus = []
        k = 0
        for initial_tau, is_trainable in zip(self.initial_taus, self.tau_is_trainable):
            if is_trainable:
                self.taus.append(torch.nn.Parameter(torch.tensor(initial_tau)))
                self.register_parameter(f'tau{k+1}', self.taus[-1])
                k += 1
                continue

            self.taus.append(torch.tensor(initial_tau))

    def forward(self, x):
        if self.project_taus_before_forward_pass:
            self.project_taus()
        tmp_diffs = []

        for el, (layer, activation, tau) in enumerate(zip(self.layers, self.activations, self.taus), start=1):
            y = (tau ** self.gamma) * \
                self.Gamma_func_eval * activation(layer(x))
            if y.shape == x.shape:
                y += x
            for k, diff in enumerate(tmp_diffs):
                y = y - self._a(el - 1, k) * diff
            if y.shape == x.shape:
                tmp_diffs.append(y - x)
            else:
                tmp_diffs.append(y)
            x = y
        return x

    @torch.no_grad()
    def _a(self, el, j):
        sum1 = sum(self.taus[k] for k in range(j, el + 1)) ** (1 - self.gamma)
        sum2 = sum(self.taus[k]
                   for k in range(j + 1, el + 1)) ** (1 - self.gamma)

        res = sum1 - sum2
        res *= (self.taus[el] ** self.gamma) / self.taus[j]
        return res

    def to(self, device: torch.device):
        super(FractionalDNN, self).to(device)
        self.device = device
        self.gamma = self.gamma.to(device)
        self.eps = self.eps.to(device)
        return self

    @torch.no_grad()
    def project_taus(self, eps: float = None):
        r"""Projects taus to minimum allowed value.
        Args:
            eps (float, optional): Minimum allowed value for taus. If not specified, the value chosen in the constructor is used.
        """
        if not eps:
            eps = self.eps

        for tau in self.taus:
            if tau < eps:
                tau.copy_(eps)
