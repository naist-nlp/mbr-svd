# Overfitting Mitigation via Singular Value Decomposition in Minimum Bayes Risk Decoding

[![arXiv](https://img.shields.io/badge/arXiv-2609.01135-b31b1b.svg)](https://arxiv.org/abs/2609.01135)

Implementation of **SVD-MBR**, from the EMNLP 2026 (Main Conference) paper:

> **Overfitting Mitigation via Singular Value Decomposition in Minimum Bayes Risk Decoding**
> Riza Setiawan Soetedjo, Yusuke Sakai, Hidetaka Kamigaito, Katsuhiko Hayashi, Taro Watanabe

Minimum Bayes Risk (MBR) decoding selects the hypothesis with the highest expected utility against a set of pseudo-references. Because it optimizes a single utility metric, MBR tends to *overfit* to that metric. SVD-MBR treats the pairwise utility matrix as a noisy signal and keeps only its top-*k* singular components (a low-rank approximation). This separates the consensus among the hypotheses from metric-specific noise before the expected utility is computed. In our experiments, this regularization helps across multiple evaluation metrics, and neural utility metrics benefit more from the denoising than surface-level metrics.

## How SVD-MBR works

For each source sentence, given hypotheses $\mathcal{H}$ and pseudo-references $\mathcal{Y}$:

1. Compute the pairwise utility matrix $U \in \mathbb{R}^{|\mathcal{H}| \times |\mathcal{Y}|}$, $U_{ij} = u(h_i, y_j)$.
2. Z-score normalize $U$ over the whole matrix.
3. Decompose it with SVD and reconstruct it from the top-*k* singular components: $\tilde{U} = U_k \Sigma_k V_k^\top$.
4. Select $\hat{h} = \arg\max_i \frac{1}{|\mathcal{Y}|}\sum_j \tilde{U}_{ij}$.

With *k* equal to the full rank, this reduces to standard MBR (up to the normalization).

## Repository structure

```
mbr-svd/
├── src/                     # Decoders, as an mbrs plugin
│   ├── svd_mbr/             #   normed_svd_mbr (SVD-MBR in the paper) and svd_mbr (without normalization)
│   ├── normed_mbr/          #   normed_mbr: MBR on the z-score normalized matrix (ablation)
│   └── component_saving/    #   component_saving_wrapper: saves standard MBR, PMBR, and Model-based MBR pairwise matrices for analysis
├── configs/                 # mbrs config templates (common settings, decoders, metrics)
├── scripts/                 # Experiment pipeline: download, generate, decode, score
├── analysis/                # Analysis code for the paper's figures and tables
│   ├── src/                 #   One module per analysis
│   ├── notebooks/           #   Exploration notebooks
│   └── run_analysis.sh      #   Runs every analysis
└── metadata/                # Index of the decoding outputs used by the analysis
```

## Installation

Requires Python 3.10 or later. The decoders are built on [mbrs](https://github.com/naist-nlp/mbrs).

With [uv](https://docs.astral.sh/uv/), using the locked versions:

```bash
git clone <this repository> mbr-svd && cd mbr-svd
uv sync
source .venv/bin/activate
```

Or with pip:

```bash
pip install "mbrs>=0.1.7" "torchnmf>=0.3.5" "rouge-score>=0.1.2" msgspec scikit-learn seaborn pytest
```

## Quick start

### Command line

The decoders register themselves with mbrs, so `mbrs-decode` can use them through `--plugin_dir src`:

```bash
# hypotheses.txt: N candidates per source sentence, one per line
mbrs-decode --plugin_dir src hypotheses.txt -n 256 \
    --metric comet \
    --decoder normed_svd_mbr --decoder.top_k_sv 1 --decoder.is_reduced true
```

You can also use a config file. The experiment scripts build one from the templates in `configs/`, for example:

```yaml
common:
  hypotheses: data/wmt22-ende.tgt.eps.256
  references: data/wmt22-ende.tgt.eps.256
  source: data/wmt22-ende.src
  num_candidates: 256
  num_references: 256
  decoder: normed_svd_mbr
  metric: comet
decoder:
  top_k_sv: 1
  is_reduced: True
  norm_dim: None
metric:
  model: Unbabel/wmt22-comet-da
```

```bash
mbrs-decode --plugin_dir src --config_path config.yaml
```

### Python

```python
from mbrs.metrics import MetricChrF
from src.svd_mbr import DecoderNormedSvdMBR

metric = MetricChrF(MetricChrF.Config())
decoder = DecoderNormedSvdMBR(DecoderNormedSvdMBR.Config(top_k_sv=1, is_reduced=True), metric)

hypotheses = ["this is a test", "this is a fest", "another test", "x"]
output = decoder.decode(hypotheses, hypotheses, nbest=1)
print(output.sentence, output.idx)  # ['this is a test'] [0]
```

## Decoders and options

| Decoder | Description |
|---|---|
| `normed_svd_mbr` | **SVD-MBR** (the proposed method): z-score normalization, then a truncated SVD reconstruction |
| `svd_mbr` | Truncated SVD reconstruction without normalization |
| `normed_mbr` | MBR on the z-score normalized matrix, without SVD |
| `component_saving_wrapper` | Wraps a decoder and saves its intermediate matrices for analysis |

`svd_mbr` and `normed_svd_mbr`:

| Option | Default | Description |
|---|---|---|
| `top_k_sv` | `None` | Number of top singular components kept (the *k* of SVD-MBR). `0` keeps all of them; `None` (with `bottom_k_sv=None`) falls back to standard MBR. |
| `bottom_k_sv` | `None` | Keep the bottom-*k* components instead. Ignored when `top_k_sv` is set. |
| `is_reduced` | `False` | Use the reduced SVD. Set to `True` in our experiments. |
| `variants` | `None` | `"skip_top1"` drops the first component; `"only_k"` keeps only the *k*-th component. |
| `norm_dim` | `None` | (`normed_svd_mbr` only) Normalization axis. `None` normalizes over the whole matrix, as in the paper. |
| `norm_eps` | `1e-8` | (`normed_svd_mbr` only) Epsilon added to the standard deviation. |

## Reproducing the experiments

The scripts in `scripts/` are numbered in the order they are run. Data and outputs go under `MAIN_DIR`, so set it to your own storage before running.

| Step | Script | Description |
|---|---|---|
| 1 | `00.download.{translation,summarization}.sh` | Download WMT22/WMT23 (via sacrebleu), XSum and CNN/DailyMail |
| 2 | `10.generate.{translation,summarization}.sh` | Sample hypotheses / pseudo-references (`scripts/generate.py`), e.g. epsilon sampling with M2M-100 |
| 3 | `20.decode.sh` | Standard MBR decoding |
| 3 | `27.decode.parameters.sh` | SVD-MBR decoding over hypothesis and reference pool sizes |
| 3 | `26.decode.save_components.sh` | Decoding that also saves the pairwise matrices used in the analysis |
| 4 | `30.score.sh` | Score the selected outputs with every evaluation metric |
| 4 | `32.score.oracle.sh` | Score every hypothesis, for the oracle and the sentence-level analyses |

Evaluation metrics: BLEU, chrF, BLEURT, COMET, BERTScore and COMETKiwi for translation; ROUGE-1/2/L and BERTScore for summarization.

## Analysis

`analysis/src/` contains one module per analysis in the paper: performance and win/tie/loss tables, bootstrap significance tests, variance and reconstruction-error correlations, singular value spectra, and qualitative examples. Each module can be run on its own from `analysis/`:

```bash
cd analysis
```

`analysis/run_analysis.sh` runs every analysis in order. The analyses read the result CSVs (`results/`), the metadata index (`metadata/`), and the saved matrices and sentence-level scores. The locations of the matrices and scores are set in `analysis/src/config.py`.

## Citation

If you use this code, please cite:

```bibtex
@misc{soetedjo2026svdmbr,
  title         = {Overfitting Mitigation via Singular Value Decomposition in Minimum Bayes Risk Decoding},
  author        = {Soetedjo, Riza Setiawan and Sakai, Yusuke and Kamigaito, Hidetaka and Hayashi, Katsuhiko and Watanabe, Taro},
  year          = {2026},
  eprint        = {2609.01135},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  note          = {Accepted to EMNLP 2026 (Main Conference)}
}
```

## Acknowledgements

This implementation is built on [mbrs](https://github.com/naist-nlp/mbrs), a library for MBR decoding.
