import pytest
import torch
from torch import Tensor
# from .svd_decompose import svd_decomposition

import numpy as np

np.random.seed(0)
torch.manual_seed(0)

H = 2
R = 3

rng = np.random.default_rng()
matrix1 = rng.random((H, R)).astype(np.float32)
matrix2 = rng.random((R, H)).astype(np.float32)

top_k_el = [None, 2, 3]
is_reduced_el = [False, True]

def manual_svd(matrix, top_k=None, is_reduced=False):
    """
    Manually compute SVD and filter singular values.
    """
    if matrix.shape[0] <= matrix.shape[1]:
        mt_m = matrix.T @ matrix
        eigvals, V = np.linalg.eigh(mt_m)
        eigvals = eigvals[::-1]
        S = np.sqrt(eigvals)
        S = S[~np.isnan(S)]
        V = V[:, ::-1]
        Vh = V.T
        S_inv = np.diag(1.0 / S)
        U = matrix @ Vh[:, :len(S)] @ S_inv
        U = U[:len(S), :]

        print(U)
        print(S)
        print(V)
        print(Vh)
        print(U @ np.diag(S) @ Vh)
    else:
        m_mt = matrix @ matrix.T
        eigvals, U = np.linalg.eigh(m_mt)
        sort_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_indices]
        U = U[:, sort_indices]
        S = np.sqrt(eigvals)
        S_inv = np.diag(1.0 / S)
        Vh = matrix.T @ U @ S_inv

    assert U.shape == (matrix.shape[0], matrix.shape[0]), (f"U shape {U.shape} incorrect")
    assert Vh.shape == (matrix.shape[1], matrix.shape[1]), (f"Vh shape {Vh.shape} incorrect")

    if is_reduced:
        k = min(matrix.shape)
        eigvals = eigvals[-k:]
        U = U[:, -k:]
        Vh = Vh[-k:, :]
    
    if top_k is not None:
        U = U[:, :top_k]
        S = S[:top_k]
        Vh = Vh[:top_k, :]
    decomposed_matrix = (U * S) @ Vh
    return decomposed_matrix, S

@pytest.mark.parametrize(
    "matrix, top_k, is_reduced", 
    [
        (matrix, top_k, is_reduced)
        for matrix in [matrix1, matrix2] 
        for top_k in top_k_el 
        for is_reduced in is_reduced_el
    ]
)
def test_svd_decomposition(matrix, top_k, is_reduced):
    correct_mat, S = manual_svd(matrix, top_k=top_k, is_reduced=is_reduced)
    
    decomposed_matrix, S_decomposed = svd_decomposition(
        Tensor(matrix), top_k=top_k, is_reduced=is_reduced
    )
    decomposed_matrix = decomposed_matrix.numpy()
    S_decomposed = S_decomposed.numpy()

    assert np.allclose(decomposed_matrix, correct_mat)
    assert np.allclose(S_decomposed, S)

    assert decomposed_matrix.shape == matrix.shape
    assert S.shape == (min(H, R),) if top_k is None else (top_k,)   

def main():
    matrix = np.array([[0,1,2], [1,2,1]])
    print(matrix, matrix.shape)
    manual_svd(matrix, top_k=None, is_reduced=False)

if __name__ == "__main__":
    main()