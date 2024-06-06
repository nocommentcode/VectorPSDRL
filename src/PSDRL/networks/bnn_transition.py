import numpy as np
from ..bnn.bootstrapped_ensemble_layer import BootstreappedEnsembleLinear
from ..common.settings import REC_CELL, TM_LOSS_F, TM_OPTIM


import torch
from numpy.random import RandomState
from torch import nn


class BNNTransition(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_actions: int,
        config: dict,
        device: str,
        ensemble_size: int,
        BnnLayer: BootstreappedEnsembleLinear,
        random: RandomState,
    ):
        super().__init__()

        self.gru_dim = config["gru_dim"]
        self.latent_dim = self.gru_dim + config["hidden_dim"]
        self.ensemble_size = ensemble_size
        self.random = random

        self.layers = nn.Sequential(
            BnnLayer(
                self.gru_dim + embed_dim + n_actions,
                self.latent_dim,
                ensemble_size,
                bias=True,
                device=device,
            ),
            nn.Tanh(),
            BnnLayer(
                self.latent_dim,
                self.latent_dim,
                ensemble_size,
                bias=True,
                device=device,
            ),
            nn.Tanh(),
            BnnLayer(
                self.latent_dim,
                self.latent_dim,
                ensemble_size,
                bias=True,
                device=device,
            ),
            nn.Tanh(),
            BnnLayer(
                self.latent_dim,
                self.latent_dim,
                ensemble_size,
                bias=True,
                device=device,
            ),
            BnnLayer(
                self.latent_dim, embed_dim + 1, ensemble_size, bias=True, device=device
            ),
        )
        self._cell = REC_CELL(embed_dim + n_actions, self.gru_dim)
        self.loss_function = TM_LOSS_F
        self.optimizer = TM_OPTIM(self.parameters(), lr=config["learning_rate"])
        self.to(device)
        self.loss = 0

    def forward(self, x: torch.tensor, hidden: torch.tensor):
        h = self._cell(x, hidden)
        return self.layers(torch.cat((h, x), dim=1)), h

    def predict(
        self, x: torch.tensor, hidden: torch.tensor, ensemble_index: int = None
    ):
        with torch.no_grad():
            h = self._cell(x, hidden)

            output = torch.cat((h, x), dim=1)
            for layer in self.layers:
                # run ensemble layer only on required index
                if isinstance(layer, BootstreappedEnsembleLinear):
                    output = layer(output, ensemble_index=ensemble_index)
                else:
                    output = layer(output)

            return output, h
