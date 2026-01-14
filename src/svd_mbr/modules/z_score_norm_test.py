import pytest
import torch
from torch import Tensor
from .z_score_norm import z_score_norm
from typing import Optional

import numpy as np

np.random.seed(0)
torch.manual_seed(0)

H = 8
R = 5

rng = np.random.default_rng()
matrix1 = rng.random((H, R)).astype(np.float32)
matrix2 = rng.random((R, H)).astype(np.float32)

dims = [None, 0, 1]

@pytest.mark.parametrize(
    "matrix, dim", 
    [
        (matrix, dim)
        for matrix in [matrix1, matrix2] 
        for dim in dims
    ]
)
def test_z_score_norm(matrix, dim: Optional[int]):
    matrix = Tensor(matrix)
    normed_matrix = z_score_norm(matrix, dim=dim)

    if dim is None:
        mean = matrix.mean()
        std = matrix.std()
        expected = (matrix - mean) / (std + 1e-8)
        torch.testing.assert_close(normed_matrix, expected)
    else:
        mean = matrix.mean(dim=dim, keepdim=True)
        std = matrix.std(dim=dim, keepdim=True)
        expected = (matrix - mean) / (std + 1e-8)
        torch.testing.assert_close(normed_matrix, expected)

    assert matrix.shape == normed_matrix.shape
    
    # Check that the mean is approximately 0 and std is approximately 1
    if dim is None:
        torch.testing.assert_close(normed_matrix.mean(), torch.tensor(0.0))
        torch.testing.assert_close(normed_matrix.std(), torch.tensor(1.0))
    else:
        torch.testing.assert_close(normed_matrix.mean(dim=dim, keepdim=True), torch.zeros_like(mean))
        torch.testing.assert_close(normed_matrix.std(dim=dim, keepdim=True), torch.ones_like(std))