import pytest
import torch
import numpy as np

from mbrs.metrics import MetricChrF, MetricCOMET
from mbrs.selectors import Selector
from mbrs.selectors.nbest import SelectorNbest

from .normed_svd_mbr import DecoderNormedSvdMBR
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
        decoder = DecoderNormedSvdMBR(
            DecoderNormedSvdMBR.Config(
                top_k_sv=TOP_K,
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
        decoder = DecoderNormedSvdMBR(
            DecoderNormedSvdMBR.Config(
                top_k_sv=TOP_K,
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
        decoder = DecoderNormedSvdMBR(
            DecoderNormedSvdMBR.Config(
                top_k_sv=TOP_K,
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

    @pytest.mark.parametrize("top_k_sv, bottom_k_sv, norm_dim", [(t_k, b_k, n_d) for t_k in top_k_el for b_k in bottom_k_el for n_d in norm_dim_el])
    def test_decode_svd(self, top_k_sv: int | None, bottom_k_sv: int | None, norm_dim: int | None):
        metric = MetricChrF(MetricChrF.Config())
        decoder = DecoderNormedSvdMBR(
            DecoderNormedSvdMBR.Config(
                top_k_sv=top_k_sv,
                bottom_k_sv=bottom_k_sv,
                norm_dim=norm_dim,
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
            print("scores")
            print(naive_output.score, output.score)
            assert len(output.sentence) == len(naive_output.sentence)
            assert len(output.score) == len(naive_output.score)
            if (top_k_sv is None and bottom_k_sv is None) or (len(hyps) <= 1):
                assert output.idx == naive_output.idx
                assert output.sentence == naive_output.sentence
                torch.testing.assert_close(
                    torch.tensor(output.score), torch.tensor(naive_output.score)
                )
            else:
                assert output.original_matrix["shape"] == output.decomposed_matrix["shape"]
                if top_k_sv is not None:
                    if top_k_sv == 0 and bottom_k_sv != 0:
                        assert output.singularvals["shape"][0] == np.nanmin(np.array([len(hyps), len(refs), bottom_k_sv], dtype=np.float32))
                    elif top_k_sv == 0 and bottom_k_sv == 0:
                        assert output.singularvals["shape"][0] == np.nanmin(np.array([len(hyps), len(refs)], dtype=np.float32))
                    else:
                        assert output.singularvals["shape"][0] == np.nanmin(np.array([top_k_sv, len(hyps), len(refs)], dtype=np.float32))
                else:
                    if bottom_k_sv == 0:
                        assert output.singularvals["shape"][0] == np.nanmin(np.array([len(hyps), len(refs)], dtype=np.float32))
                    else:
                        assert output.singularvals["shape"][0] == np.nanmin(np.array([len(hyps), len(refs), bottom_k_sv], dtype=np.float32))

