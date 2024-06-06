from ..common.settings import WEIGHT_INIT
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
        self.weight = torch.nn.Parameter(
            torch.empty(ensemble_size, in_features, out_features)
        )

        self.reset_parameters()
        self.to(device)

    def ensemble_forward(self, x):
        return torch.einsum("jmi,jio->jmo", x, self.weight)

    def single_forward(self, x, ensemble_index):
        weight = self.weight[ensemble_index]
        return x.matmul(weight)

    def reset_parameters(self):
        super().reset_parameters()
        WEIGHT_INIT(self.weight)
