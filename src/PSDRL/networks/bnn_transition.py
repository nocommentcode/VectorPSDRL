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

        self.reset_diversity_measure()

    def reset_diversity_measure(self):
        self.diversity_std = []

    def record_diversity(self, output):
        self.diversity_std.append(output.std(0).sum().item())

    def flush_diversity(self):
        std = np.array(self.diversity_std).mean()
        self.reset_diversity_measure()
        return std

    def forward(self, x: torch.tensor, hidden: torch.tensor):
        h = self._cell(x, hidden)
        return self.layers(torch.cat((h, x), dim=1)), h

    def predict(self, x: torch.tensor, hidden: torch.tensor):
        with torch.no_grad():
            x = torch.concatenate([x for _ in range(self.ensemble_size)], 0)
            h = torch.concatenate([hidden for _ in range(self.ensemble_size)], 0)

            pred, h = self.forward(x, h)

            # sample one of the  ensembles as the prediction
            sample_indx = self.random.randint(0, self.ensemble_size)

            def sample_ensemble(output, record_deversity=False):
                output = output.view((self.ensemble_size, -1, *output.shape[1:]))

                if record_deversity:
                    self.record_diversity(output)

                return output[sample_indx]

            return sample_ensemble(pred, record_deversity=True), sample_ensemble(h)
