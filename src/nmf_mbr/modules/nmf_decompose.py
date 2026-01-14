from typing import Optional
import torch
from torch import Tensor

from torchnmf.nmf import NMF

def nmf_decomposition(
        matrix: Tensor, 
        rank: Optional[int]=None,
        beta: float=1.0,
        l1_ratio: float=0.0,
        epsilon: float=1e-8,
        device: Optional[torch.device]=None
    ) -> tuple[Tensor, Tensor, Tensor]:
    """
    Perform Non-negative Matrix Factorization (NMF) on the input matrix.
    Args:
        matrix (Tensor): The input non-negative matrix to decompose.
        rank (Optional[int]): The number of components to use for the decomposition.
        beta (float): The beta divergence to use. Default is 1.0 (Kullback-Leibler divergence).
        l1_ratio (float): The L1 regularization ratio. Default is 0.
    Returns:
        tuple[Tensor, Tensor, Tensor]: The reconstructed matrix after NMF decomposition, and the factor matrices (W, H).
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if not torch.all(matrix >= 0):
        print("Input matrix contains negative values. Shifting to non-negative by adding the absolute minimum value.")
        min_val = torch.min(matrix)
        matrix = matrix - min_val + epsilon  # Shift to non-negative
    elif not torch.all(matrix > 0) and beta <=0:
        print("Input matrix contains zero values. Adding epsilon to avoid issues with beta <= 0.")
        matrix = matrix + epsilon
    matrix = matrix.to(device)
    if rank is None or rank == 0 or rank > min(matrix.shape):
        rank = min(matrix.shape)
    nmf_model = NMF(matrix.shape, rank=rank).to(device)
    nmf_model.fit(matrix, beta=beta, l1_ratio=l1_ratio)
    decomposed_matrix = nmf_model()

    if device == torch.device("cuda"):
        torch.cuda.synchronize()
        W = nmf_model.W.detach().cpu()
        H = nmf_model.H.detach().cpu()
        decomposed_matrix = decomposed_matrix.detach().cpu()
        matrix = matrix.detach().cpu()
    else:
        W = nmf_model.W.detach()
        H = nmf_model.H.detach()
        decomposed_matrix = decomposed_matrix.detach()
        matrix = matrix.detach()

    assert decomposed_matrix.shape == matrix.shape, (
        f"Decomposed matrix shape {decomposed_matrix.shape} does not match original shape {matrix.shape}"
    )
    return decomposed_matrix, W, H