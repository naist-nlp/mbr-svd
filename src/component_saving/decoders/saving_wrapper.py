from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field
from typing import Any, Optional, Dict

from mbrs import timer, functional
from mbrs.decoders import register, get_decoder, DecoderMBR
from mbrs.metrics import Metric
from mbrs.selectors import SELECTOR_NBEST, Selector

@register("component_saving_wrapper")
class ComponentSavingWrapper(DecoderMBR):
    """
    A smart wrapper decoder that intercepts components generated during 
    the decoding process and packages them into the output object for 
    batch saving downstream.
    """
    
    cfg: Config

    @dataclass
    class Config(DecoderMBR.Config):
        """Configuration for the Component Saving Wrapper."""
        base_decoder: str
        base_decoder_config: Dict[str, Any] = field(default_factory=dict)
        

    @dataclass
    class Output(DecoderMBR.Output):
        """Extended output dataclass to hold the intercepted matrices."""
        reconstructed_pairwise_scores: Optional[torch.Tensor] = None
        sampled_pairwise_indices: Optional[torch.Tensor] = None
        original_matrix: Optional[torch.Tensor] = None

    def __init__(
        self, 
        cfg: Config,        
        metric: Metric,
        selector: Selector = SELECTOR_NBEST
    ):
        # 1. Pass both config and metric to the parent DecoderMBR
        super().__init__(cfg, metric, selector)
        self.cfg = cfg
        
        # 2. Dynamically load the base decoder plugin class
        decoder_cls = get_decoder(self.cfg.base_decoder)
        
        # 3. Manually cast the dictionary into the specific decoder's Config dataclass
        filtered_config = {
            k:v for k,v in self.cfg.base_decoder_config.items()
            if v is not None
        }
        base_cfg = decoder_cls.Config(**filtered_config)
        
        # 4. Instantiate the underlying decoder WITH the properly typed config AND the metric
        self.underlying_decoder = decoder_cls(base_cfg, metric, selector)

    def extract_pmbr_components(
        self, 
        hypotheses: list[str], 
        references: list[str], 
        source: Optional[str] = None,
        nbest: int = 1,
        reference_lprobs: Optional[torch.Tensor] = None,
    ):
        """
        Specifically for probabilistic MBR decoders, compute pairwise scores, 
        regenerate indices, and attach them to the output.
        """
        # Intercept the reconstructed scores
        reconstructed_scores = self.underlying_decoder.pairwise_scores_probabilistic(
            hypotheses, references, source
        )
            
        # Deterministically recreate the sampled indices
        cfg = self.underlying_decoder.cfg
        rng = torch.Generator().manual_seed(cfg.seed)
        H = len(hypotheses)
        R = len(references)
        num_ucalcs = math.ceil(H * R / cfg.reduction_factor)

        pairwise_sample_indices = torch.randperm(H * R, generator=rng)[:num_ucalcs]
        hypothesis_sample_indices = (pairwise_sample_indices // R).tolist()
        reference_sample_indices = (pairwise_sample_indices % R).tolist()
        sampled_indices = torch.tensor(list(zip(hypothesis_sample_indices, reference_sample_indices)))

        # Continue with standard MBR selection using the intercepted scores
        expected_scores = functional.expectation(reconstructed_scores, lprobs=reference_lprobs)
        selector_outputs = self.underlying_decoder.select(hypotheses, expected_scores, nbest=nbest, source=source)
        
        # Package everything into our extended Output dataclass
        wrapper_output = self.Output(
            idx=selector_outputs.idx,
            sentence=selector_outputs.sentence,
            score=selector_outputs.score,
            reconstructed_pairwise_scores=reconstructed_scores,
            sampled_pairwise_indices=sampled_indices,
        )
        
        return wrapper_output | selector_outputs

    def decode(
        self, 
        hypotheses: list[str], 
        references: list[str], 
        source: Optional[str] = None,
        nbest: int = 1,
        reference_lprobs: Optional[torch.Tensor] = None,
    ):
        """
        Overrides the main decode pipeline to intercept pairwise scores
        and pass them downstream in memory.
        """
        with timer.measure("Total Decoding Time"):
            if "probabilistic_mbr" in self.cfg.base_decoder and self.underlying_decoder.cfg.reduction_factor > 1.0:
                return self.extract_pmbr_components(hypotheses, references, source, nbest, reference_lprobs)
            
            else:
                # Intercept standard MBR dense matrix
                pairwise_matrix = self.underlying_decoder.metric.pairwise_scores(hypotheses, references, source=source)
                
                # Let the underlying decoder finish its process
                base_output = self.underlying_decoder.decode(
                    hypotheses, references, source=source, nbest=nbest, reference_lprobs=reference_lprobs
                )
                
                # Wrap its result in our extended Output dataclass
                wrapper_output = self.Output(
                    idx=base_output.idx,
                    sentence=base_output.sentence,
                    score=base_output.score,
                    original_matrix=pairwise_matrix
                )
                
                return wrapper_output | base_output