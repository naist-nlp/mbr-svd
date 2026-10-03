import pytest
import torch
import numpy as np

from mbrs.metrics import MetricChrF, MetricCOMET
from mbrs.selectors import Selector
from mbrs.selectors.nbest import SelectorNbest

from .normed_mbr import DecoderNormedMBR
from mbrs.decoders import DecoderMBR

SOURCE = [
    "これはテストです",
    "これはテストです",
    "これはテストです",
    "これはテストです",
]
HYPOTHESES = [
    ["another test", "this is a test", "this is a fest", "x", "this is test"],
    ["another test", "this is a fest", "this is a test"],
    ["this is a test"],
    ["Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;"],
]
REFERENCES = [
    ["another test", "this is a test", "this is a fest", "x", "this is test"],
    ["this is a test", "ref", "these are tests", "this is the test"],
    ["this is a test"],
    ["producţia de zahăr brut se exprimă în zahăr alb;"],
]

BEST_INDICES = [1, 2, 0, 0]
BEST_SENTENCES = [
    "this is a test",
    "this is a test",
    "this is a test",
    "Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;",
]

TOP_K = 0
BOTTOM_K = None
NORM_DIM = None

top_k_el = [None, 0, 1, 2, 5, 10]
bottom_k_el = [None, 0, 1]
norm_dim_el = [None, 0, 1]


class TestDecoderNormedSvdMBR:
    def test_decode_chrf(self):
        metric = MetricChrF(MetricChrF.Config())
        decoder = DecoderNormedMBR(
            DecoderNormedMBR.Config(
                norm_dim=NORM_DIM,
            ),
            metric,
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, SOURCE[i], nbest=1)
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]

    def test_decode_comet(self):
        metric_comet = MetricCOMET(MetricCOMET.Config())
        decoder = DecoderNormedMBR(
            DecoderNormedMBR.Config(
                norm_dim=NORM_DIM,
            ),
            metric_comet,
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, SOURCE[i], nbest=1)
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]

            output = decoder.decode(
                hyps,
                refs,
                SOURCE[i],
                nbest=1,
                reference_lprobs=torch.Tensor([-2.000]).repeat(len(refs)),
            )
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]

    @pytest.mark.parametrize("nbest", [1, 2])
    def test_decode_selector(self, nbest: int):
        selector = SelectorNbest(SelectorNbest.Config())
        metric = MetricChrF(MetricChrF.Config())
        decoder = DecoderNormedMBR(
            DecoderNormedMBR.Config(
                norm_dim=NORM_DIM,
            ),
            metric,
            selector=selector,
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, SOURCE[i], nbest=nbest)
            assert len(output.sentence) == min(nbest, len(hyps))
            assert len(output.score) == min(nbest, len(hyps))

            output = decoder.decode(
                hyps,
                refs,
                SOURCE[i],
                nbest=nbest,
                reference_lprobs=torch.Tensor([-2.000]).repeat(len(refs)),
            )
            assert len(output.sentence) == min(nbest, len(hyps))
            assert len(output.score) == min(nbest, len(hyps))