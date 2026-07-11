from typing import Optional
import torch
from torch import Tensor

import numpy as np

def svd_decomposition(
    matrix: Tensor, 
    top_k: Optional[int]=None,
    bottom_k: Optional[int]=None, 
    is_reduced=False,
    device: Optional[torch.device]=None
    ) -> tuple[Tensor, Tensor]:
    """Compute the singular value decomposition (SVD) of a matrix.

    Args:
        matrix (Tensor): Input matrix of shape `(H, R)`.
        top_k (int, optional): Number of top singular values to keep. If None, keep all.
        bottom_k (int, optional): Number of bottom singular values to keep. If None, keep all. If both top_k and bottom_k are provided, ignore bottom_k.
        is_reduced (bool, optional): Whether to use reduced SVD.
        device (torch.device, optional): Device to perform computation on. If None, use GPU if available.
    Returns:
        Tensor: Decomposed matrix after filtering small singular values.
        Tensor: Top-k singular values.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    matrix = matrix.to(device)
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=not is_reduced)
    rank = S.shape[0]
    if bottom_k and top_k:
        bottom_k = None  # Ignore bottom_k if top_k is provided
    if top_k is not None:
        top_k = min(top_k, rank)
        U = U[:, :top_k]
        S = S[:top_k]
        Vh = Vh[:top_k, :]
    elif bottom_k is not None:
        bottom_k = min(bottom_k, rank)
        U = U[:, (rank-bottom_k):rank]
        S = S[(rank-bottom_k):rank]
        Vh = Vh[(rank-bottom_k):rank, :]
    if is_reduced:
        decomposed_matrix = (U * S) @ Vh
    else:
        S_full = torch.zeros(U.shape[1], Vh.shape[0], dtype=matrix.dtype, device=matrix.device)
        S_full[:S.shape[0], :S.shape[0]] = torch.diag(S)
        decomposed_matrix = U @ S_full @ Vh

    assert decomposed_matrix.shape == matrix.shape, (
        f"Decomposed matrix shape {decomposed_matrix.shape} does not match original shape {matrix.shape}"
    )
    
    if device == torch.device("cuda"):
        torch.cuda.synchronize()
        decomposed_matrix = decomposed_matrix.detach().cpu()
        S = S.detach().cpu()
    else:
        decomposed_matrix = decomposed_matrix.detach()
        S = S.detach()

    return decomposed_matrix, S, U, Vh