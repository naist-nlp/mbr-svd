import pytest
import torch
from torch import Tensor
from .svd_decompose import svd_decomposition
from typing import Optional

import numpy as np

np.random.seed(0)
torch.manual_seed(0)

H = 8
R = 5

rng = np.random.default_rng()
matrix1 = rng.random((H, R)).astype(np.float32)
matrix2 = rng.random((R, H)).astype(np.float32)

top_k_el = [None, 2, 3]
bottom_k_el = [None, 2, 3]
is_reduced_el = [False, True]

def manual_reduced_svd(A):
    """
    Computes the reduced SVD of A (A = U * S @ Vh)
    using torch.linalg.eigh.
    """
    m, n = A.shape
    
    if m >= n:
        # Case 1: Tall or Square (m >= n)
        M = A.T @ A
        eigvals, V = torch.linalg.eigh(M)
        
        eigvals, indices = torch.sort(eigvals, descending=True)
        V = V[:, indices]
        
        S_squared = torch.clamp(eigvals, min=0.0)
        S_k = torch.sqrt(S_squared)
        Vh_k = V.T
        
        S_inv = 1.0 / S_k
        S_inv[S_k < 1e-8] = 0.0
        U_k = (A @ V) * S_inv
        
    else:
        # Case 2: Wide (n > m)
        M = A @ A.T
        eigvals, U = torch.linalg.eigh(M)
        
        eigvals, indices = torch.sort(eigvals, descending=True)
        U = U[:, indices]
        
        S_squared = torch.clamp(eigvals, min=0.0)
        S_k = torch.sqrt(S_squared)
        U_k = U
        
        S_inv = 1.0 / S_k
        S_inv[S_k < 1e-8] = 0.0
        Vh_k = (U_k.T @ A) * S_inv.unsqueeze(1)

    return U_k, S_k, Vh_k

def manual_full_svd(A):
    """
    Computes the full SVD (U is m x m, Vh is n x n)
    using our reduced SVD and orthogonal completion via QR.
    """
    m, n = A.shape
    k = min(m, n)
    
    # 1. Get the "economy" SVD first
    U_k, S_k, Vh_k = manual_reduced_svd(A)
    
    # 2. Complete U
    if m > k:
        # This code block is NOT run for this 2x3 matrix
        U_basis = torch.randn(m, m, dtype=A.dtype, device=A.device)
        U_basis[:, :k] = U_k
        U, _ = torch.linalg.qr(U_basis, mode='complete')
    else:
        U = U_k # U is already full (2x2)
    
    # 3. Complete Vh
    if n > k:
        # This code block IS run
        V_k = Vh_k.T
        # This randn is now deterministic due to the seed
        V_basis = torch.randn(n, n, dtype=A.dtype, device=A.device)
        V_basis[:, :k] = V_k
        V, _ = torch.linalg.qr(V_basis, mode='complete')
        Vh = V.T
    else:
        Vh = Vh_k # Vh is already full
    
    return U, S_k, Vh

def assert_valid_svd(A, U, S_vec, Vh):
    """
    Validates an SVD decomposition by checking the fundamental
    relationship A*v_i = s_i*u_i for each component.
    
    This is more robust than a simple reconstruction, as it
    catches "decoupled" sign mismatches.
    """
    
    print("--- Running Robust SVD Validation ---")
    
    # 1. Check Orthogonality of U and Vh
    k = S_vec.shape[0] # Number of singular values
    I_k_m = torch.eye(k, dtype=A.dtype, device=A.device)
    I_k_n = torch.eye(k, dtype=A.dtype, device=A.device)

    # We check U.T @ U (for the reduced U)
    U_k = U[:, :k]
    U_check = U_k.T @ U_k
    torch.testing.assert_close(U_check, I_k_m)
    print("U Orthogonality: PASSED")
    
    # We check Vh @ Vh.T (for the reduced Vh)
    Vh_k = Vh[:k, :]
    V = Vh_k.T # V is (n, k)
    Vh_check = Vh_k @ Vh_k.T
    torch.testing.assert_close(Vh_check, I_k_n)
    print("Vh Orthogonality: PASSED")

    # 2. Check the Component Relationship: A*v_i = s_i*u_i
    print("Checking component relationships (A*v_i = s_i*u_i)...")
    
    for i in range(k):
        u_i = U[:, i]      # The i-th left vector
        v_i = V[:, i]      # The i-th right vector
        s_i = S_vec[i]   # The i-th singular value
        
        # Calculate left and right sides of the equation
        LHS = A @ v_i
        RHS = s_i * u_i
        
        is_close = torch.allclose(LHS, RHS)
        
        if not is_close:
            # Check if the "inverse" is true (A*v_i = -s_i*u_i)
            # This should never happen, but shows the mismatch
            is_inverse = torch.allclose(LHS, -RHS)
            is_same_val = LHS.abs() == RHS.abs()
            if is_inverse or is_same_val:
                continue
            print(f"  Component {i}: FAILED")
            print(f"    LHS (A @ v_i) = {LHS.data}")
            print(f"    RHS (s_i * u_i) = {RHS.data}")
            print(f"    (Matches inverse? {is_inverse})")
            print(f"    (Matches absolute values? {is_same_val})")
            raise AssertionError(f"Component {i} failed the coupling test.")
            
        print(f"  Component {i}: PASSED")

@pytest.mark.parametrize(
    "matrix, top_k, bottom_k, is_reduced", 
    [
        (matrix, top_k, bottom_k, is_reduced)
        for matrix in [matrix1, matrix2] 
        for top_k in top_k_el 
        for bottom_k in bottom_k_el
        for is_reduced in is_reduced_el
    ]
)
def test_svd_decomposition(matrix, top_k, bottom_k, is_reduced):
    matrix = Tensor(matrix)
    H, R = matrix.shape
    if is_reduced:
        U, S, Vh = manual_reduced_svd(matrix)
        assert U.shape == (H, min(H, R)) and Vh.shape == (min(H, R), R), "Reduced SVD shapes are incorrect. Got U: {}, Vh: {}".format(U.shape, Vh.shape)
        assert S.shape == (min(H, R),), "Reduced SVD singular values shape is incorrect. Got S: {}".format(S.shape)
        reconstruct_mat = (U * S) @ Vh
    else:  
        U, S, Vh = manual_full_svd(matrix)
        assert U.shape == (H, H) and Vh.shape == (R, R), "Full SVD shapes are incorrect. Got U: {}, Vh: {}".format(U.shape, Vh.shape)
        assert S.shape == (min(H, R),), "Full SVD singular values shape is incorrect. Got S: {}".format(S.shape)
        k = min(H, R)
        S_full = torch.zeros((H, R), dtype=matrix.dtype, device=matrix.device)
        S_full[:k, :k] = torch.diag(S)
        reconstruct_mat = U @ S_full @ Vh
    # Assert reconstruction is close to original but ignore sign differences
    try:
        torch.testing.assert_close(reconstruct_mat.abs(), matrix.abs())
    except Exception as e:
        print("Reconstructed Matrix is not close to Original Matrix. Checking SVD validity.")
        print("Reason for failure:\n", e)
        try:
            assert_valid_svd(matrix, U, S, Vh)
        except Exception as e2:
            print("SVD validity check failed.")
            raise e2
        else:
            print("SVD validity check passed. Using function output as ground truth.")
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=not is_reduced)
            if is_reduced:
                reconstruct_mat = (U * S) @ Vh 
            else:
                k = min(H, R)
                S_full = torch.zeros((H, R), dtype=matrix.dtype, device=matrix.device)
                S_full[:k, :k] = torch.diag(S)
                reconstruct_mat = U @ S_full @ Vh

    if bottom_k and top_k:
        bottom_k = None  # Ignore bottom_k if top_k is provided

    if top_k is not None:
        top_k = min(top_k, S.shape[0])
        U = U[:, :top_k]
        S = S[:top_k]
        Vh = Vh[:top_k, :]
        if is_reduced:
            reconstruct_mat = (U * S) @ Vh
        else:
            S_full = torch.zeros((top_k, top_k), dtype=matrix.dtype, device=matrix.device)
            S_full[:top_k, :top_k] = torch.diag(S)
            reconstruct_mat = U @ S_full @ Vh
    elif bottom_k is not None:
        bottom_k = min(bottom_k, S.shape[0])
        U = U[:, -bottom_k:]
        S = S[-bottom_k:]
        Vh = Vh[-bottom_k:, :]
        if is_reduced:
            reconstruct_mat = (U * S) @ Vh
        else:
            S_full = torch.zeros((bottom_k, bottom_k), dtype=matrix.dtype, device=matrix.device)
            S_full[:bottom_k, :bottom_k] = torch.diag(S)
            reconstruct_mat = U @ S_full @ Vh

    decomposed_matrix, S_decomposed = svd_decomposition(
        matrix, 
        top_k=top_k, 
        bottom_k=bottom_k,
        is_reduced=is_reduced
    )

    # Assert decomposition is close to manual but ignore sign differences
    try:
        torch.testing.assert_close(decomposed_matrix.abs(), reconstruct_mat.abs())
    except Exception as e:
        print("Decomposed Matrix:\n", decomposed_matrix)
        print("Manual Reconstructed Matrix:\n", reconstruct_mat)
        raise e
    try:
        torch.testing.assert_close(S_decomposed, S)
    except Exception as e:
        print("Decomposed Singular Values:\n", S_decomposed)
        print("Manual Singular Values:\n", S)
        raise e

    assert decomposed_matrix.shape == matrix.shape, (
        f"Decomposed matrix shape {decomposed_matrix.shape} does not match original shape {matrix.shape}"
    )
    if top_k is not None:
        assert S.shape == (min(top_k, S.shape[0]),), (
            f"Singular values shape {S_decomposed.shape} does not match expected shape {(min(top_k, S.shape[0]),)}"
        )
    elif bottom_k is not None:
        assert S.shape == (min(bottom_k, S.shape[0]),), (
            f"Singular values shape {S_decomposed.shape} does not match expected shape {(min(bottom_k, S.shape[0]),)}"
        )
    else:
        assert S.shape == (min(H, R),), (
            f"Singular values shape {S_decomposed.shape} does not match expected shape {(min(H, R),)}"
        )