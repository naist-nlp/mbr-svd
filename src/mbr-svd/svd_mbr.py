from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from mbrs import functional

from mbrs.decoders import register
from mbrs.decoders.mbr import DecoderMBR

def svd_decomposition(matrix: Tensor, sv_threshold=None) -> Tensor:
    """Compute the singular value decomposition (SVD) of a matrix.

    Args:
        matrix (Tensor): Input matrix of shape `(H, R)`.

    Returns:
        Tensor: Decomposed matrix after filtering small singular values.
    """
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    if sv_threshold is not None:
        mask = S > sv_threshold
        U = U[:, mask]
        S = S[mask]
        Vh = Vh[mask, :]
    decomposed_matrix = (U * S) @ Vh

    print("U:", U)
    print("S:", S)
    print("Vh:", Vh)

    assert decomposed_matrix.shape == matrix.shape, (
        f"Decomposed matrix shape {decomposed_matrix.shape} does not match original shape {matrix.shape}"
    )
    return decomposed_matrix


@register("svd_mbr")
class DecoderSvdMBR(DecoderMBR):
    """SVD-based MBR decoder.
    """

    cfg: Config

    @dataclass
    class Config(DecoderMBR.Config):
        """Configuration for the decoder.

        - svd_threshold (float, optional): Threshold for singular values. Singular values below this threshold will be discarded.
        - seed (int): Random seed.
        """

        svd_threshold: float = Optional[float]  # type: ignore
        seed: int = 0

    def pairwise_scoring(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
    ) -> Tensor:
        """Compute pairwise scores using the Naive MBR algorithm.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.

        Returns:
            Tensor: Pairwise scores of shape `(H, R)`.
        """
        H = len(hypotheses)
        R = len(references)

        pairwise_scores = self.metric.pairwise_scores(
            hypotheses, references, source
        )

        return pairwise_scores

    def decode(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        nbest: int = 1,
        reference_lprobs: Optional[Tensor] = None,
    ) -> DecoderMBR.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            DecoderMBR.Output: The n-best hypotheses.
        """

        if self.cfg.svd_threshold is None:
            expected_scores = self.metric.expected_scores(
                hypotheses, references, source, reference_lprobs=reference_lprobs
            )
        else:  # SVD MBR decoding
            pairwise_scores = self.pairwise_scoring(
                hypotheses, references, source
            )
            pairwise_scores = svd_decomposition(pairwise_scores, sv_threshold=self.cfg.svd_threshold)
            expected_scores = functional.expectation(
                pairwise_scores, lprobs=reference_lprobs
            )

        selector_outputs = self.select(
            hypotheses, expected_scores, nbest=nbest, source=source
        )
        return (
            self.Output(
                idx=selector_outputs.idx,
                sentence=selector_outputs.sentence,
                score=selector_outputs.score,
            )
            | selector_outputs
        )
