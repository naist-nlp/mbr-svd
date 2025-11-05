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
        Tensor: Top-k singular values.
    """
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=not is_reduced)
    if top_k is not None:
        top_k = min(top_k, S.shape[0])
        U = U[:, :top_k]
        S = S[:top_k]
        Vh = Vh[:top_k, :]
    if is_reduced:
        decomposed_matrix = (U * S) @ Vh
    else:
        S_full = torch.zeros(U.shape[1], Vh.shape[0], dtype=matrix.dtype, device=matrix.device)
        S_full[:S.shape[0], :S.shape[0]] = torch.diag(S)
        decomposed_matrix = U @ S_full @ Vh

    assert decomposed_matrix.shape == matrix.shape, (
        f"Decomposed matrix shape {decomposed_matrix.shape} does not match original shape {matrix.shape}"
    )
    return decomposed_matrix, S