from ..common.settings import WEIGHT_INIT
from .bootstrapped_ensemble_layer import BootstreappedEnsembleLinear


import torch
import torch.nn as nn


class BatchEnsembleLinear(BootstreappedEnsembleLinear):
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

        self.r = nn.Parameter(torch.empty(ensemble_size, in_features))
        self.s = nn.Parameter(torch.empty(ensemble_size, out_features))

        self.linear = nn.Linear(in_features, out_features, bias=False, device=device)
        self.weight = self.linear.weight

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.uniform_(self.r, -1, 1)
        nn.init.uniform_(self.s, -1, 1)

    def ensemble_forward(self, x):
        # r -> J, 1, I
        r = self.r.unsqueeze(1)

        # s ->  J, 1, O
        s = self.s.unsqueeze(1)

        # output -> J, M, O
        output = self.linear(x * r) * s

        return output
