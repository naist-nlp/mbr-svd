from typing import Optional

from matplotlib.pylab import eigvals
import pytest
import torch
from torch import Tensor

from mbrs.functional import expectation
from svd_functional import svd_expectation

H = 8
R = 5

A = torch.rand(H, R)

lprobs_el = [None, torch.FloatTensor([-3.0, -2.4, -1.9, -5.5, -4.3])]
sv_threshold_el = [None, 0.5]

def manual_svd(matrix, sv_threshold=None) -> Tensor:
    "Manual SVD implementation for testing purposes."
    if matrix.shape[0] >= matrix.shape[1]:
        mt_m = matrix.T @ matrix
        eigenvals, Vh = torch.linalg.eigh(mt_m)
        eigenvals, indices = torch.sort(eigenvals, descending=True)
        Vh = Vh[:, indices]
        S = torch.sqrt(torch.clamp(eigenvals, min=0.0))
        S_inv = torch.zeros_like(S)
        nonzero_mask = S > 1e-8
        S_inv[nonzero_mask] = 1.0 / S[nonzero_mask]
        U = matrix @ Vh.T @ torch.diag(S_inv)
    else:
        m_mt = matrix @ matrix.T
        eigenvals, U = torch.linalg.eigh(m_mt)
        eigenvals, indices = torch.sort(eigenvals, descending=True)
        U = U[:, indices]
        S = torch.sqrt(torch.clamp(eigenvals, min=0.0))
        S_inv = torch.zeros_like(S)
        nonzero_mask = S > 1e-8
        S_inv[nonzero_mask] = 1.0 / S[nonzero_mask]
        Vh = matrix.T @ U @ torch.diag(S_inv)

    if sv_threshold is not None:
        mask = S > sv_threshold
        U = U[:, mask]
        S = S[mask]
        Vh = Vh[mask, :]
    decomposed_matrix = (U * S) @ Vh

    print("U (manual):", U)
    print("S (manual):", S)
    print("Vh (manual):", Vh)

    assert decomposed_matrix.shape == matrix.shape, (
        f"Decomposed matrix shape {decomposed_matrix.shape} does not match original shape {matrix.shape}"
    )
    return decomposed_matrix


@pytest.mark.parametrize(
    "lprobs,sv_threshold", [
        (lprob, sv)
        for lprob in lprobs_el
        for sv in sv_threshold_el
    ]
)
def test_svd_expectation(lprobs: Optional[Tensor], sv_threshold: Optional[Tensor]):
    e = svd_expectation(A, lprobs=lprobs, sv_threshold=sv_threshold)
    assert list(e.shape) == [H]

    if sv_threshold is not None:
        correct_mat = manual_svd(A, sv_threshold=sv_threshold).mean(dim=-1)
    else:
        correct_mat = A.mean(dim=-1)

    if lprobs is not None:
        with pytest.raises(ValueError):
            svd_expectation(torch.rand(H, R - 1), lprobs=lprobs, sv_threshold=sv_threshold)
        weights = lprobs.exp() / lprobs.exp().sum(dim=-1, keepdim=True)
        correct_mat = (A @ weights[None, :].T).squeeze(-1)
    else:
        correct_mat = A.mean(dim=-1)
    print(e, correct_mat)
    assert torch.allclose(e, correct_mat)

# @pytest.mark.parametrize(
#     "lprobs,sv_threshold", [
#         (lprob, sv)
#         for lprob in lprobs_el
#         for sv in sv_threshold_el
#     ]
# )
# def test_expectation(lprobs: Optional[Tensor], sv_threshold: Optional[Tensor]):
#     e = expectation(A, lprobs=lprobs, sv_threshold=sv_threshold)
#     assert list(e.shape) == [H]
#     if lprobs is None:
#         assert torch.allclose(e, A.mean(dim=-1))
#     else:
#         with pytest.raises(ValueError):
#             expectation(torch.rand(H, R - 1), lprobs=lprobs)

#         weights = lprobs.exp() / lprobs.exp().sum(dim=-1, keepdim=True)
#         assert torch.allclose(e, (A @ weights[None, :].T).squeeze(-1))