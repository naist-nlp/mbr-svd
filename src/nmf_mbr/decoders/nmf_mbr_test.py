import pytest
import torch

from mbrs.metrics import MetricChrF, MetricCOMET
from mbrs.selectors import Selector
from mbrs.selectors.nbest import SelectorNbest

from .nmf_mbr import DecoderNmfMBR
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

RANK = 0

rank_el = [None, 0, 2]
beta_el = [0.0, 0.5, 1.0, 2.0]
l1_ratio_el = [0.0, 0.1, 0.5]


class TestDecoderNmfMBR:
    def test_decode_chrf(self):
        metric = MetricChrF(MetricChrF.Config())
        decoder = DecoderNmfMBR(
            DecoderNmfMBR.Config(
                rank=RANK,
            ),
            metric,
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, SOURCE[i], nbest=1)
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]

    def test_decode_comet(self):
        metric_comet = MetricCOMET(MetricCOMET.Config())
        decoder = DecoderNmfMBR(
            DecoderNmfMBR.Config(
                rank=RANK,
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
        decoder = DecoderNmfMBR(
            DecoderNmfMBR.Config(
                rank=RANK,
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

    @pytest.mark.parametrize("rank, beta, l1_ratio", [(r, b, l) for r in rank_el for b in beta_el for l in l1_ratio_el])
    def test_decode_nmf(self, rank: int | None, beta: float, l1_ratio: float):
        metric = MetricChrF(MetricChrF.Config())
        decoder = DecoderNmfMBR(
            DecoderNmfMBR.Config(
                rank=rank,
                beta=beta,
                l1_ratio=l1_ratio,
                save_components=True,
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
        
            if rank != None:
                assert output.original_matrix is not None
                assert output.W is not None
                assert output.H is not None
                assert output.decomposed_matrix is not None
                assert output.original_matrix.shape == output.decomposed_matrix.shape
                assert torch.all(output.decomposed_matrix >= 0)
                assert torch.all(output.W >= 0)
                assert torch.all(output.H >= 0)
            else:
                torch.testing.assert_close(
                    torch.tensor(naive_output.score),
                    torch.tensor(output.score),
                )