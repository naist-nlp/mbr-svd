import torch
from torch import Tensor

import numpy as np

def svd_decomposition(matrix: Tensor, top_k=None, is_reduced=False) -> tuple[Tensor, Tensor]:
    """Compute the singular value decomposition (SVD) of a matrix.

    Args:
        matrix (Tensor): Input matrix of shape `(H, R)`.
        top_k (int, optional): Number of top singular values to keep. If None, keep all.
        is_reduced (bool, optional): Whether to use reduced SVD.

    Returns:
        Tensor: Decomposed matrix after filtering small singular values.
    """
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=not is_reduced)
    if top_k is not None:
        U = U[:, :top_k]
        S = S[:top_k]
        Vh = Vh[:top_k, :]
    decomposed_matrix = (U * S) @ Vh

    assert decomposed_matrix.shape == matrix.shape, (
        f"Decomposed matrix shape {decomposed_matrix.shape} does not match original shape {matrix.shape}"
    )
    return decomposed_matrix, S