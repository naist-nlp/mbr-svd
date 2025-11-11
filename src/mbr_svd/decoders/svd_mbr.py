from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from mbrs import functional, timer

from mbrs.decoders import register
from mbrs.decoders.mbr import DecoderMBR
from ..modules.svd_decompose import svd_decomposition

@register("svd_mbr")
class DecoderSvdMBR(DecoderMBR):
    """SVD-based MBR decoder.
    """

    cfg: Config

    @dataclass
    class Config(DecoderMBR.Config):
        """Configuration for the decoder.

        - top_k_sv (int, optional): Only get the top-k singular values. If None, no SVD is applied. if 0, all singular values are kept.
        - bottom_k_sv (int, optional): Only get the bottom-k singular values. If None, no SVD is applied. if 0, all singular values are kept.
        - is_reduced (bool): Whether to use reduced SVD.
        - seed (int): Random seed.
        """

        top_k_sv: Optional[int] = None
        bottom_k_sv: Optional[int] = None
        is_reduced: bool = False
        seed: int = 0
    
    @dataclass
    class Output(DecoderMBR.Output):
        """Output of the SVD MBR decoder.
        """
        original_matrix: Optional[Tensor] = None
        eigenvals: Optional[Tensor] = None
        decomposed_matrix: Optional[Tensor] = None

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
    ) -> DecoderSvdMBR.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            DecoderSvdMBR.Output: The n-best hypotheses.
        """

        if self.cfg.top_k_sv is None and self.cfg.bottom_k_sv is None: # Naive MBR decoding
            expected_scores = self.metric.expected_scores(
                hypotheses, references, source, reference_lprobs=reference_lprobs
            )
            eigenvals = None
            original_matrix = None
            decomposed_matrix = None
        else:  # SVD MBR decoding
            pairwise_scores = self.pairwise_scoring(
                hypotheses, references, source
            )
            original_matrix = {
                "data": pairwise_scores.tolist(),
                "shape": pairwise_scores.shape,
                "dtype": str(pairwise_scores.dtype)
            }
            with timer.measure("svd_decomposition"):
                pairwise_scores, eigenvals = svd_decomposition(
                    pairwise_scores, 
                    top_k=None if self.cfg.top_k_sv == 0 else self.cfg.top_k_sv,
                    bottom_k=None if self.cfg.bottom_k_sv == 0 else self.cfg.bottom_k_sv,
                    is_reduced=self.cfg.is_reduced
                )
                eigenvals = {
                    "data": eigenvals.tolist(),
                    "shape": eigenvals.shape,
                    "dtype": str(eigenvals.dtype)
                }
                decomposed_matrix = {
                    "data": pairwise_scores.tolist(),
                    "shape": pairwise_scores.shape,
                    "dtype": str(pairwise_scores.dtype)
                }
            with timer.measure("expectation"):
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
                eigenvals=eigenvals,
                original_matrix=original_matrix,
                decomposed_matrix=decomposed_matrix,
            )
        )
