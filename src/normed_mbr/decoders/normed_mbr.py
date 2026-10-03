from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from mbrs import functional, timer

from mbrs.decoders import register
from mbrs.decoders.mbr import DecoderMBR
from ..modules.z_score_norm import z_score_norm

@register("normed_mbr")
class DecoderNormedMBR(DecoderMBR):
    """Z-score normalized MBR decoder.
    """

    cfg: Config

    @dataclass
    class Config(DecoderMBR.Config):
        """Configuration for the decoder.
        Initially inherits from DecoderMBR.Config.
        - norm_dim (int, optional): Dimension along which to normalize. If None, normalize over the entire matrix.
        """
        norm_dim: Optional[int] = None
        norm_eps: float = 1e-8
        save_components: bool = True

    @dataclass
    class Output(DecoderMBR.Output):
        """Output of the Normed MBR decoder.
        """
        original_matrix: Optional[Tensor] = None

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
        with timer.measure("pairwise_scoring"):
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
    ) -> DecoderNormedMBR.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            DecoderNormedMBR.Output: The n-best hypotheses.
        """
        pairwise_scores = self.pairwise_scoring(hypotheses, references, source)
            
        with timer.measure("z_score_normalization"):
            normed_pairwise_scores = z_score_norm(pairwise_scores, dim=self.cfg.norm_dim, epsilon=self.cfg.norm_eps)

        if not torch.isfinite(normed_pairwise_scores).all():
            print("Pairwise scores after normalization:", normed_pairwise_scores)
            raise ValueError("Non-finite values found in the normalized pairwise score matrix.")

        with timer.measure("expectation"):
            expected_scores = functional.expectation(
                normed_pairwise_scores, 
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
                    original_matrix=normed_pairwise_scores,
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
