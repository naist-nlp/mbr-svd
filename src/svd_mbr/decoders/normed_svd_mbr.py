from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from mbrs import functional, timer

from mbrs.decoders import register
from ..decoders.svd_mbr import DecoderSvdMBR
from ..modules.z_score_norm import z_score_norm
from ..modules.svd_decompose import svd_decomposition

@register("normed_svd_mbr")
class DecoderNormedSvdMBR(DecoderSvdMBR):
    """SVD-based MBR decoder.
    However, before decomposing the pairwise score matrix, each row is normalized using z-score normalization.
    """

    cfg: Config

    @dataclass
    class Config(DecoderSvdMBR.Config):
        """Configuration for the decoder.
        Initially inherits from DecoderSvdMBR.Config.
        - norm_dim (int, optional): Dimension along which to normalize. If None, normalize over the entire matrix.
        """
        norm_dim: Optional[int] = None
        norm_eps: float = 1e-8

    def decode(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        nbest: int = 1,
        reference_lprobs: Optional[Tensor] = None,
    ) -> DecoderNormedSvdMBR.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            DecoderNormedSvdMBR.Output: The n-best hypotheses.
        """
        pairwise_scores = self.pairwise_scoring(hypotheses, references, source)
            
        if self.cfg.top_k_sv is None and self.cfg.bottom_k_sv is None: # Naive MBR decoding
            print("Warning: Using Naive MBR decoding instead of SVD MBR decoding.")
            with timer.measure("expectation"):
                expected_scores = functional.expectation(
                    matrix=pairwise_scores,
                    lprobs=reference_lprobs,
                )
            singularvals = None
            dec_pairwise_scores = None
        else:  # Normed SVD MBR decoding
            with timer.measure("z_score_normalization"):
                normed_pairwise_scores = z_score_norm(pairwise_scores, dim=self.cfg.norm_dim, epsilon=self.cfg.norm_eps)

            if not torch.isfinite(normed_pairwise_scores).all():
                print("Pairwise scores after normalization:", normed_pairwise_scores)
                raise ValueError("Non-finite values found in the normalized pairwise score matrix.")
    
            with timer.measure("svd_decomposition"):
                dec_pairwise_scores, singularvals = svd_decomposition(
                    normed_pairwise_scores, 
                    top_k=None if self.cfg.top_k_sv == 0 else self.cfg.top_k_sv,
                    bottom_k=None if self.cfg.bottom_k_sv == 0 else self.cfg.bottom_k_sv,
                    is_reduced=self.cfg.is_reduced
                )
            with timer.measure("expectation"):
                expected_scores = functional.expectation(
                    dec_pairwise_scores, 
                    lprobs=reference_lprobs
                )
        
        selector_outputs = self.select(
            hypotheses, expected_scores, nbest=nbest, source=source
        )
        if self.cfg.save_components:
            return (
                self.Output(
                    idx=selector_outputs.idx,
                    sentence=selector_outputs.sentence,
                    score=selector_outputs.score,
                    singularvals=singularvals,
                    original_matrix=pairwise_scores,
                    decomposed_matrix=dec_pairwise_scores,
                )
            )
        else:
            return (
                self.Output(
                    idx=selector_outputs.idx,
                    sentence=selector_outputs.sentence,
                    score=selector_outputs.score,
                )
            )
