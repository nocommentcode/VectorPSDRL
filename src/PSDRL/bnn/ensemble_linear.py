import math
from .bootstrapped_ensemble_layer import BootstreappedEnsembleLinear


import torch
import torch.nn as nn


class EnsembleLinear(BootstreappedEnsembleLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, ensemble_size, bias, device)
        self.device = device
        self.layers = [
            nn.Linear(in_features, out_features, bias=False, device=self.device).to(
                self.device
            )
            for _ in range(ensemble_size)
        ]
        self.weight = torch.stack([layer.weight for layer in self.layers]).to(device)

        self.reset_parameters()
        self.to(device)

    def ensemble_forward(self, x):
        return torch.stack([layer(mini_x) for layer, mini_x in zip(self.layers, x)]).to(
            self.device
        )

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
