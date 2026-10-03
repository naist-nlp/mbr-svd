import pytest
import torch

from mbrs.metrics import MetricChrF, MetricBLEU
from mbrs.selectors import Selector
from mbrs.conftest import *

# Make sure to adjust this import to match your actual file path
from .saving_wrapper import ComponentSavingWrapper

SOURCE = [
    "これはテストです",
    "これはテストです",
]
HYPOTHESES = [
    ["another test", "this is a fest", "this is a test"],
    ["this is a test"],
]
REFERENCES = [
    ["this is a test", "ref", "these are tests", "this is the test"],
    ["this is a test"],
]

BEST_INDICES = [2, 0]
BEST_SENTENCES = [
    "this is a test",
    "this is a test",
]

class TestComponentSavingWrapper:

    def test_saving_standard_mbr(self):
        """Test wrapping a standard MBR decoder and passing the native dense matrix in memory."""
        # Config for standard MBR
        base_decoder_config = {}
        
        cfg = ComponentSavingWrapper.Config(
            base_decoder="mbr",
            base_decoder_config=base_decoder_config
        )
        
        metric = MetricChrF(MetricChrF.Config())
        wrapper = ComponentSavingWrapper(cfg=cfg, metric=metric)

        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = wrapper.decode(hyps, refs, source=SOURCE[i], nbest=1)
            
            # Assert decoding outputs match expected standard MBR results
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]
            
            # Assert the component was actually attached to the output object
            assert output.original_matrix is not None
            
            # Verify the tensor shape matches (H x R)
            assert output.original_matrix.shape == (len(hyps), len(refs))

    @pytest.mark.parametrize("reduction_factor", [2.0, 4.0])
    def test_saving_probabilistic_mbr(self, reduction_factor: float):
        """Test wrapping PMBR to ensure the sparse matrix and indices are extracted to memory."""
        base_decoder_config = {"reduction_factor": reduction_factor, "seed": 42}
        
        cfg = ComponentSavingWrapper.Config(
            base_decoder="probabilistic_mbr",
            base_decoder_config=base_decoder_config
        )
        
        metric = MetricChrF(MetricChrF.Config())
        wrapper = ComponentSavingWrapper(cfg=cfg, metric=metric)

        hyps, refs = HYPOTHESES[0], REFERENCES[0]
        output = wrapper.decode(hyps, refs, source=SOURCE[0], nbest=1)
        
        assert output is not None

        # Check if the exact properties we defined in the wrapper's custom Output were populated
        assert output.reconstructed_pairwise_scores is not None
        assert output.sampled_pairwise_indices is not None
        
        # Verify the dimensions of the saved index tensor
        # Check that it generated 2 columns (hypothesis index, reference index)
        assert output.sampled_pairwise_indices.shape[1] == 2