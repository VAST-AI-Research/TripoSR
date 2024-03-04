from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from ..utils import BaseModule


class TriplaneUpsampleNetwork(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int
        out_channels: int

    cfg: Config

    def configure(self) -> None:
        self.upsample = nn.ConvTranspose2d(
            self.cfg.in_channels, self.cfg.out_channels, kernel_size=2, stride=2
        )

    def forward(self, triplanes: torch.Tensor) -> torch.Tensor:
        triplanes_up = rearrange(
            self.upsample(
                rearrange(triplanes, "B Np Ci Hp Wp -> (B Np) Ci Hp Wp", Np=3)
            ),
            "(B Np) Co Hp Wp -> B Np Co Hp Wp",
            Np=3,
        )
        return triplanes_up


class NeRFMLP(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int
        n_neurons: int
        n_hidden_layers: int
        activation: str = "relu"
        bias: bool = True
        weight_init: Optional[str] = "kaiming_uniform"
        bias_init: Optional[str] = None

    cfg: Config

    def configure(self) -> None:
        layers = [
            self.make_linear(
                self.cfg.in_channels,
                self.cfg.n_neurons,
                bias=self.cfg.bias,
                weight_init=self.cfg.weight_init,
                bias_init=self.cfg.bias_init,
            ),
            self.make_activation(self.cfg.activation),
        ]
        for i in range(self.cfg.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.cfg.n_neurons,
                    self.cfg.n_neurons,
                    bias=self.cfg.bias,
                    weight_init=self.cfg.weight_init,
                    bias_init=self.cfg.bias_init,
                ),
                self.make_activation(self.cfg.activation),
            ]
        layers += [
            self.make_linear(
                self.cfg.n_neurons,
                4,  # density 1 + features 3
                bias=self.cfg.bias,
                weight_init=self.cfg.weight_init,
                bias_init=self.cfg.bias_init,
            )
        ]
        self.layers = nn.Sequential(*layers)

    def make_linear(
        self,
        dim_in,
        dim_out,
        bias=True,
        weight_init=None,
        bias_init=None,
    ):
        layer = nn.Linear(dim_in, dim_out, bias=bias)

        if weight_init is None:
            pass
        elif weight_init == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        else:
            raise NotImplementedError

        if bias:
            if bias_init is None:
                pass
            elif bias_init == "zero":
                torch.nn.init.zeros_(layer.bias)
            else:
                raise NotImplementedError

        return layer

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        inp_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])

        features = self.layers(x)
        features = features.reshape(*inp_shape, -1)
        out = {"density": features[..., 0:1], "features": features[..., 1:4]}

        return out
