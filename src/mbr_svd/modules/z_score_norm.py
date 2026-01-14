import torch
from torch import Tensor
from typing import Optional

def z_score_norm(
    matrix: Tensor,
    dim: Optional[int] = None,
    epsilon: float = 1e-8,
    device: Optional[torch.device]=None
) -> Tensor:
    """
    Apply z-score normalization to the input matrix.
    Args:
        matrix (Tensor): Input matrix of shape `(H, R)`.
        dim (Optional[int]): Dimension along which to normalize. If None, normalize over the entire matrix.
        epsilon (float): Small value to avoid division by zero.
        device (torch.device, optional): Device to perform computation on. If None, use GPU if available.
    Returns:
        Tensor: Z-score normalized matrix.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    matrix = matrix.to(device)
    if matrix.numel() <= 1:
        print("Warning: Matrix has one or no elements, skipping normalization.")
        return matrix

    if dim is None:
        mean = matrix.mean()
        std = matrix.std()
    else:
        mean = matrix.mean(dim=dim, keepdim=True)
        std = matrix.std(dim=dim, keepdim=True)

    normed_matrix = (matrix - mean) / (std + epsilon)
    
    if device == torch.device("cuda"):
        torch.cuda.synchronize()
        normed_matrix = normed_matrix.detach().cpu()
    else:
        normed_matrix = normed_matrix.detach()

    return normed_matrix