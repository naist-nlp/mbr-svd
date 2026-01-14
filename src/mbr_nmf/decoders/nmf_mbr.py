from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from mbrs import functional, timer

from mbrs.decoders import register
from mbrs.decoders.mbr import DecoderMBR
from ..modules.nmf_decompose import nmf_decomposition

@register("nmf_mbr")
class DecoderNmfMBR(DecoderMBR):
    """NMF-based MBR decoder.
    """

    cfg: Config

    @dataclass
    class Config(DecoderMBR.Config):
        """Configuration for the decoder.

        - rank (Optional[int]): Rank for NMF decomposition. If None, no decomposition is applied. If 0, use min(H, R)//2.
        - beta (float): Beta divergence for NMF decomposition. Default is 1.0 (Kullback-Leibler divergence).
        - l1_ratio (float): L1 regularization ratio for NMF decomposition. Default is 0.0.
        - device (Optional[torch.device]): Device to perform NMF decomposition. If None, automatically select CPU or GPU.
        """
        
        rank: Optional[int] = None
        beta: float = 1.0
        l1_ratio: float = 0.0
        device: Optional[torch.device] = None
    
    @dataclass
    class Output(DecoderMBR.Output):
        """Output of the NMF MBR decoder.
        """
        original_matrix: Optional[Tensor] = None
        decomposed_matrix: Optional[Tensor] = None
        W: Optional[Tensor] = None
        H: Optional[Tensor] = None

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
    ) -> DecoderNmfMBR.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            DecoderNmfMBR.Output: The n-best hypotheses.
        """

        if self.cfg.rank is None: # Naive MBR decoding
            expected_scores = self.metric.expected_scores(
                hypotheses, references, source, reference_lprobs=reference_lprobs
            )
            original_matrix = None
            decomposed_matrix = None
            W = None
            H = None
        else:  # NMF MBR decoding
            pairwise_scores = self.pairwise_scoring(
                hypotheses, references, source
            )
            original_matrix = {
                "data": pairwise_scores.tolist(),
                "shape": pairwise_scores.shape,
                "dtype": str(pairwise_scores.dtype)
            }
            with timer.measure("nmf_decomposition"):
                pairwise_scores, W, H = nmf_decomposition(
                    pairwise_scores, 
                    rank=self.cfg.rank, 
                    beta=self.cfg.beta, 
                    l1_ratio=self.cfg.l1_ratio,
                    device=self.cfg.device
                )
                decomposed_matrix = {
                    "data": pairwise_scores.tolist(),
                    "shape": pairwise_scores.shape,
                    "dtype": str(pairwise_scores.dtype)
                }
                W = {
                    "data": W.tolist(),
                    "shape": W.shape,
                    "dtype": str(W.dtype)
                }
                H = {
                    "data": H.tolist(),
                    "shape": H.shape,
                    "dtype": str(H.dtype)
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
                original_matrix=original_matrix,
                decomposed_matrix=decomposed_matrix,
                W=W,
                H=H,
            )
        )
