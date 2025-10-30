from typing import Optional

import mbrs.functional as F
import torch
from torch import Tensor

def svd_decomposition(matrix: Tensor, sv_threshold=None) -> Tensor:
    """Compute the singular value decomposition (SVD) of a matrix.

    Args:
        matrix (Tensor): Input matrix of shape `(H, R)`.

    Returns:
        Tensor: Decomposed matrix after filtering small singular values.
    """
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    if sv_threshold is not None:
        mask = S > sv_threshold
        U = U[:, mask]
        S = S[mask]
        Vh = Vh[mask, :]
    decomposed_matrix = (U * S) @ Vh

    print("U:", U)
    print("S:", S)
    print("Vh:", Vh)

    assert decomposed_matrix.shape == matrix.shape, (
        f"Decomposed matrix shape {decomposed_matrix.shape} does not match original shape {matrix.shape}"
    )
    return decomposed_matrix

def svd_expectation(matrix: Tensor, lprobs: Optional[Tensor] = None, sv_threshold=None) -> Tensor:
    """Compute expectation values for each row.

    Args:
        matrix (Tensor): Input matrix of shape `(H, R)`.
        lprobs (Tensor, optional): Log-probabilities for each column of shape `(R,)`.

    Returns:
        Tensor: Expected values for each row of shape `(H,)`.
    """
    matrix = svd_decomposition(matrix, sv_threshold=sv_threshold)

    if lprobs is None:
        return matrix.mean(dim=-1)
    else:
        if list(lprobs.shape) != list(matrix.shape)[1:]:
            raise ValueError(
                f"`weights` must have {list(matrix.shape)[1:]} elements, but got {list(lprobs.shape)}"
            )

        return (
            (
                matrix.float()
                * lprobs.to(matrix).softmax(dim=-1, dtype=torch.float32)[None, :]
            )
            .sum(dim=-1)
            .to(matrix)
        )

F.expectation = svd_expectation