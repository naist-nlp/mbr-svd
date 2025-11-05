import pytest
import torch

from mbrs.metrics import MetricChrF, MetricCOMET
from mbrs.selectors import Selector
from mbrs.selectors.nbest import SelectorNbest

from .svd_mbr import DecoderSvdMBR
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


class TestDecoderSvdMBR:
    def test_decode_chrf(self):
        metric = MetricChrF(MetricChrF.Config())
        decoder = DecoderSvdMBR(
            DecoderSvdMBR.Config(
                top_k_sv=TOP_K,
            ),
            metric,
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, SOURCE[i], nbest=1)
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]

    def test_decode_comet(self):
        metric_comet = MetricCOMET(MetricCOMET.Config())
        decoder = DecoderSvdMBR(
            DecoderSvdMBR.Config(
                top_k_sv=TOP_K,
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
        decoder = DecoderSvdMBR(
            DecoderSvdMBR.Config(
                top_k_sv=TOP_K,
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
    
    @pytest.mark.parametrize("top_k_sv", [None, 0, 1, 2, 5, 10])
    def test_decode_svd(self, top_k_sv: int | None):
        metric = MetricChrF(MetricChrF.Config())
        decoder = DecoderSvdMBR(
            DecoderSvdMBR.Config(
                top_k_sv=top_k_sv,
            ),
            metric,
        )
        naive_decoder = DecoderMBR(
            DecoderMBR.Config(),
            metric,
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            naive_output = naive_decoder.decode(hyps, refs, SOURCE[i], nbest=len(hyps))
            output = decoder.decode(hyps, refs, SOURCE[i], nbest=len(hyps))
            print(naive_output.score, output.score)
            assert output.idx == naive_output.idx
            assert output.sentence == naive_output.sentence
            if top_k_sv is None:
                torch.testing.assert_close(
                    torch.tensor(output.score), torch.tensor(naive_output.score)
                )
            else:
                if top_k_sv == 0 or top_k_sv >= len(hyps):
                    torch.testing.assert_close(
                        torch.tensor(output.score), torch.tensor(naive_output.score)
                    )
                else:
                    assert not torch.allclose(torch.tensor(output.score), torch.tensor(naive_output.score))
                    assert len(output.eigenvals) == min(top_k_sv, len(hyps))
