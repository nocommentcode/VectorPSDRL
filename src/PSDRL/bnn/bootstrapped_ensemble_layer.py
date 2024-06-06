import math
import torch
import torch.nn as nn


class BootstreappedEnsembleLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        self.J = ensemble_size
        self.I = in_features
        self.O = out_features

        self.bias = (
            nn.Parameter(torch.Tensor(ensemble_size, out_features)) if bias else 0
        ).to(device)

    def reset_parameters(self):
        if self.bias is not 0:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, ensemble_index=None):
        """
        Runs the ensemble layer

        params:
            - x: N x I tensor
            - ensemble_index: optional ensemble to isolate
        returns"
            - out: N x O tensor
        """
        if ensemble_index is None:
            # x -> J, M, I
            x = self.split_batch_to_ensemble_chunks(x)

            # output -> J, M, O
            output = self.ensemble_forward(x)

            # b -> J, 1, O
            b = self.bias.unsqueeze(1)  # J, 1, O
        else:
            # output -> N, O
            output = self.single_forward(x, ensemble_index)

            # b -> 1, O
            b = self.bias[ensemble_index].unsqueeze(0)

        # output -> J, M, O
        output = output + b

        # output -> JM, O
        output = output.view((-1, self.O))

        return output

    def split_batch_to_ensemble_chunks(self, x: torch.tensor) -> torch.tensor:
        """
        Splits a batch to ensemble chunks
            N = batch size
            J = ensembe size
            M = batch size per ensemble
            I = input dimension
        Args:
            - x: N x I input tensor
        Returns:
            - x: J X M  X I tensor"""

        # check dimensions match
        N, I = x.shape
        if N % self.J != 0:
            raise ValueError(
                f"Batch size of {N} is not compatible with {self.J} ensembles,  Batch size must be a multiple of ensemble size"
            )

        M = N // self.J

        # x -> J, M, I
        return x.view((self.J, M, I))

    def ensemble_forward(self, x):
        """
        Forward method for ensembe methods
        params:
            - x: J x M x I tensor
        returns:
            - J x M x O tensor
        """
        raise NotImplementedError(
            "ensemble_forward not implemented with AbstractEnsembleLinear"
        )

    def single_forward(self, x, ensemble_index):
        """
        Forward method for a single ensembe
        params:
            - x: N x I tensor
            - ensemble_index: ensemble to forward

        returns:
            - N x O tensor
        """
        raise NotImplementedError(
            "ensemble_forward not implemented with AbstractEnsembleLinear"
        )
