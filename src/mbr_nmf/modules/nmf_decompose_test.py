import pytest
import torch
from torch import Tensor
from .nmf_decompose import nmf_decomposition

import numpy as np

np.random.seed(0)
torch.manual_seed(0)

H = 8
R = 5

rng = np.random.default_rng()
matrix1 = rng.random((H, R)).astype(np.float32)
matrix2 = rng.random((R, H)).astype(np.float32)
matrix_minus = rng.random((H, R)).astype(np.float32) - 0.5  # Contains negative values

rank = [None, 0, 2]
beta = [0.0, 1.0, 2.0]
l1_ratio = [0.0, 0.1, 0.5]

@pytest.mark.parametrize('matrix, rank, beta, l1_ratio', [(matrix, r, b, l) for matrix in [matrix1, matrix2, matrix_minus] for r in rank for b in beta for l in l1_ratio])
def test_nmf_decomposition(matrix, rank, beta, l1_ratio):
    matrix = torch.tensor(matrix)
    decomposed_matrix, W, H = nmf_decomposition(matrix, rank=rank, beta=beta, l1_ratio=l1_ratio)
    assert W.shape[0] == matrix.shape[1], "W matrix has incorrect number of rows. W shape: {}, matrix shape: {}".format(W.shape, matrix.shape)
    assert H.shape[0] == matrix.shape[0], "H matrix has incorrect number of columns. H shape: {}, matrix shape: {}".format(H.shape, matrix.shape)
    assert W.shape[1] == H.shape[1], "W and H matrices have different number of columns. W shape: {}, H shape: {}".format(W.shape, H.shape)
    assert W.shape[1] == rank if rank not in [None, 0] else min(matrix.shape), "W matrix has incorrect number of columns. W shape: {}, expected rank: {}".format(W.shape, rank if rank is not None else min(matrix.shape))
    assert H.shape[1] == rank if rank not in [None, 0] else min(matrix.shape), "H matrix has incorrect number of columns. H shape: {}, expected rank: {}".format(H.shape, rank if rank is not None else min(matrix.shape))
    assert decomposed_matrix.shape == matrix.shape, "Decomposed matrix has incorrect shape. Decomposed shape: {}, original shape: {}".format(decomposed_matrix.shape, matrix.shape)
    assert torch.all(decomposed_matrix >= 0), "Decomposed matrix contains negative values"
    assert torch.all(W >= 0), "W matrix contains negative values"
    assert torch.all(H >= 0), "H matrix contains negative values"