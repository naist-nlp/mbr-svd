#!/usr/bin/env python3

import dataclasses
import enum
import json
import logging
import os
import sys
from argparse import FileType, Namespace
from dataclasses import asdict, dataclass, fields, make_dataclass
from typing import Sequence

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

import simple_parsing
import torch
from simple_parsing import choice, field, flag
from simple_parsing.wrappers import dataclass_wrapper
from tabulate import tabulate, tabulate_formats
from tqdm import tqdm

from mbrs import registry, timer
from mbrs.args import ArgumentParser, DataclassWrapper
from mbrs.decoders import (
    DecoderBase,
    DecoderReferenceBased,
    DecoderReferenceless,
    get_decoder,
)
from mbrs.metrics import Metric, MetricEnum, MetricReferenceless, get_metric
from mbrs.selectors import Selector, get_selector

from sklearn.model_selection import ParameterGrid # 追加したもの

class Format(enum.Enum):
    plain = "plain"
    json = "json"


@dataclass
class CommonArguments:
    """Common arguments."""

    # Hypotheses file.
    hypotheses: str = field(positional=True)
    # Number of candidates.
    num_candidates: int = field(alias=["-n"])
    # Source file.
    source: str | None = field(default=None, alias=["-s"])
    # References file.
    references: str | None = field(default=None, alias=["-r"])
    # References log-probabilities file.
    reference_lprobs: str | None = field(default=None)
    # Output file.
    # output: FileType("w", encoding="utf-8") = field(default="-", alias=["-o"])
    output: str | None = field(default="-", alias=["-o"])
    # Output format.
    format: Format = choice(Format, default=Format.plain)
    # Number of references for each sentence.
    num_references: int | None = field(default=None)
    # Type of the decoder.
    decoder: str = field(
        default="mbr",
        metadata={
            "choices": registry.get_registry(
                DecoderReferenceBased | DecoderReferenceless
            )
        },
    )
    # Type of the metric.
    metric: str = field(
        default="bleu",
        metadata={"choices": registry.get_registry(Metric | MetricReferenceless)},
    )
    # Type of the selector.
    selector: str = field(
        default="nbest", metadata={"choices": registry.get_registry(Selector)}
    )
    # Return the n-best hypotheses.
    nbest: int = field(default=1)
    # No verbose information and report.
    quiet: bool = flag(default=False)
    # Report file.
    report: FileType("w") = field(default="-")
    # Report runtime statistics with the given format.
    report_format: str = choice(*tabulate_formats, default="rounded_outline")
    # Number of digits for values of float point.
    width: int = field(default=1, alias=["-w"])

    def get_decoder_type(self) -> type[DecoderReferenceBased | DecoderReferenceless]:
        return get_decoder(self.decoder)

    def get_metric_type(self) -> type[Metric | MetricReferenceless]:
        return get_metric(self.metric)

    def get_selector_type(self) -> type[Selector]:
        return get_selector(self.selector)

    def output_results(self, res: DecoderBase.Output):
        return self.format.output_results(res, self.output)


def get_argparser(args: Sequence[str] | None = None) -> ArgumentParser:
    meta_parser = ArgumentParser(add_help=False, add_config_path_arg=True)
    meta_parser.add_arguments(
        CommonArguments, "common", dataclass_wrapper_class=DataclassWrapper
    )
    for _field in meta_parser._wrappers[0].fields:
        _field.required = False
    known_args, _ = meta_parser.parse_known_args(args=args)
    metric_type = get_metric(known_args.common.metric)
    decoder_type = get_decoder(known_args.common.decoder)
    selector_type = get_selector(known_args.common.selector)

    parser = ArgumentParser(add_help=False, add_config_path_arg=True)
    parser.add_arguments(
        CommonArguments, "common", dataclass_wrapper_class=DataclassWrapper
    )
    parser.add_arguments(metric_type.Config, "metric", prefix="metric.")
    parser.add_arguments(decoder_type.Config, "decoder", prefix="decoder.")
    parser.add_arguments(selector_type.Config, "selector", prefix="selector.")
    for _field in parser._wrappers[0].fields:
        _field.required = False

    known_args, _ = parser.parse_known_args(args=args)
    for cfg, m in [
        (known_args.metric, metric_type),
        (known_args.decoder, decoder_type),
        (known_args.selector, selector_type),
    ]:
        for _field in fields(cfg):
            field_name = _field.name
            field_attr = getattr(cfg, field_name)
            if isinstance(field_attr, MetricEnum):
                config_type = get_metric(field_attr).Config
                m.Config = make_dataclass(
                    m.Config.__name__,
                    fields=[
                        (
                            field_name,
                            type(field_attr),
                            dataclasses.field(default=field_attr),
                        ),
                        (
                            field_name + "_config",
                            config_type,
                            dataclasses.field(default_factory=config_type),
                        ),
                    ],
                    bases=(m.Config,),
                )

    parser = ArgumentParser(add_help=True, add_config_path_arg=True)
    parser.add_arguments(
        CommonArguments, "common", dataclass_wrapper_class=DataclassWrapper
    )
    parser.add_arguments(metric_type.Config, "metric", prefix="metric.")
    parser.add_arguments(decoder_type.Config, "decoder", prefix="decoder.")
    parser.add_arguments(selector_type.Config, "selector", prefix="selector.")
    return parser


def format_argparser() -> ArgumentParser:
    parser = get_argparser()
    parser.preprocess_parser()
    return parser


def main(args: Namespace) -> None:
    if not args.common.quiet:
        logger.info(args)

    sources = None
    if args.common.source is not None:
        with open(args.common.source, mode="r") as f:
            sources = f.readlines()

    with open(args.common.hypotheses, mode="r") as f:
        hypotheses = f.readlines()

    references = None
    if args.common.references is not None:
        with open(args.common.references, mode="r") as f:
            references = f.readlines()
    else:
        references = hypotheses

    reference_lprobs = None
    if args.common.reference_lprobs is not None:
        with open(args.common.reference_lprobs, mode="r") as f:
            reference_lprobs = f.readlines()
        assert len(references) == len(reference_lprobs)

    metric: Metric = get_metric(args.common.metric)(args.metric)
    selector: Selector = get_selector(args.common.selector)(args.selector)
    decoder: DecoderReferenceBased | DecoderReferenceless = get_decoder(
        args.common.decoder
    )(args.decoder, metric, selector)

    num_cands = args.common.num_candidates
    num_refs = args.common.num_references or num_cands
    num_sents = len(hypotheses) // num_cands
    assert num_sents * num_cands == len(hypotheses)

    # ここもいいかんじに変更した。args.common.outputをoutput_pathにして、forでガッツリ回せるようにする
    def output_results(res_list: list[DecoderBase.Output], output_file):
        
        if args.common.format == Format.plain:
            mbr_data_file = open(output_path+".mbr_data", "w", encoding="utf-8")

            torch_results = {}
            for i, res in enumerate(res_list):
                sent, idx, score = res.sentence, res.idx, res.score
                print(sent, file=output_file)
                print(json.dumps({
                    "selected_idx": idx[0] if isinstance(idx, list) else idx,
                    "rank": i,
                    "expected_score": score
                }), file=mbr_data_file)

                for k, v in asdict(res).items():
                    if k not in {"sentence", "idx", "score"}:
                        if v is not None and type(v) == torch.Tensor:
                            v_load = v.cpu().unsqueeze(0)
                            if k not in torch_results:
                                torch_results[k] = v_load
                            else:
                                torch_results[k] = torch.cat((torch_results[k], v_load), dim=0)

            mbr_data_file.close()
            print("Available keys", list(torch_results.keys()))
            for k, v in torch_results.items():
                if "original_matrix" in k:
                    if "-normed_svd_mbr" in output_path:
                        new_output_path = output_path.replace("-normed_svd_mbr", "-mbr")
                    elif "-svd_mbr" in output_path:
                        new_output_path = output_path.replace("-svd_mbr", "-mbr")
                    elif "-nmf_mbr" in output_path:
                        new_output_path = output_path.replace("-nmf_mbr", "-mbr")
                    else:
                        new_output_path = output_path
                        
                    new_fname = new_output_path.split(".")
                    new_fname = ".".join(new_fname[:6]) + ".original_matrix.pt"
                    if os.path.exists(new_fname):
                        print(f"{new_fname} already exists, asserting similar values")
                        existing_tensor = torch.load(new_fname)
                        current_tensor = v.cpu()
                        if torch.allclose(existing_tensor, current_tensor):
                            print(f"{new_fname} is close to the current tensor, skipping saving.")
                            continue
                    else:
                        print(f"Saving original matrix to {new_fname}")
                        torch.save(v.cpu(), new_fname)
                else:
                    torch.save(v, f"{output_path}.{k}.pt")
            
        elif args.common.format == Format.json:
            for i, res in enumerate(res_list):
                sent, idx, score = res.sentence, res.idx, res.score
                print(
                    json.dumps(
                        {
                            "rank": i,
                            "sentence": sent,
                            "selected_idx": idx[0] if isinstance(idx, list) else idx,
                            "expected_score": score,
                            **{
                                k: v
                                for k, v in asdict(res).items()
                                if k not in {"sentence", "idx", "score"}
                            },
                        },
                        ensure_ascii=False,
                    ),
                    file=output_file,
                )

    # referenceなしにして、defにリフォームした。
    # 思い切って、model_baseはなしにした。
    #src_list = list()
    #hyps_list = list()
    #refs_list = list()
    #
    #for i in tqdm(range(num_sents), desc= 'preprocessing'):
    #    src_list = sources[i].strip() if sources is not None else None
    #    hyps_list = [h.strip() for h in hypotheses[i * num_cands : (i + 1) * num_cands]]
    #    refs_list = [r.strip() for r in references[i * num_refs : (i + 1) * num_refs]]
    #ref_lprobs = None
    #print(num_sents, len(src_list), len(hyps_list), len(refs_list))
    # forを2つにわけることで、いいかんじに変更。
    # やめた
    
    def run(hyperparams):
        output_file = open(output_path, "w", encoding="utf-8")
        new_config = args.decoder.__class__(**{**asdict(args.decoder), **hyperparams})
        args.decoder = new_config
        print(args.decoder)
        decoder = get_decoder(
            args.common.decoder
        )(args.decoder, metric, selector)

        results = []
        
        for i in tqdm(range(num_sents), leave=False):
            src = sources[i].strip() if sources is not None else None
            hyps = [h.strip() for h in hypotheses[i * num_cands : (i + 1) * num_cands]]
            refs = [r.strip() for r in references[i * num_refs : (i + 1) * num_refs]]
            ref_lprobs = None
            #src = src_list[i]
            #hyps = hyps_list[i]
            #refs = refs_list[i]
            with timer.measure("total"):
                output = decoder.decode(
                    hyps,
                    refs,
                    src,
                    nbest=args.common.nbest,
                    reference_lprobs=ref_lprobs
                )
                results.append(output)
        output_results(results, output_file)
        output_file.close()
    
    if args.common.decoder == "component_saving_wrapper":
        output_path = args.common.output
        # Check if the components file already exists
        if args.decoder.base_decoder == "mbr":
            components = ["original_matrix.pt"]
        elif args.decoder.base_decoder == "probabilistic_mbr":
            components = ["reconstructed_pairwise_scores.pt", "sampled_pairwise_indices.pt"]
        else:
            raise ValueError("Unknown base decoder in component_saving_wrapper.")
        if all(os.path.exists(f"{output_path}.{comp}") for comp in components):
            print(f"Component files for {output_path} already exist. Skipping decoding.")
            return
        run({})
        if not args.common.quiet:
            try:
                statistics = timer.aggregate().result(num_sents)
            except Exception as e:
                print(f"Failed to get statistics: {e}", file=args.common.report)
            else:
                table = tabulate(
                    statistics,
                    headers="keys",
                    tablefmt=args.common.report_format,
                    floatfmt=f".{args.common.width}f",
                )
                print(table, file=args.common.report)
        return

    # ここでgrid searchする
    if args.common.decoder == "svd_mbr":
        param_grid = {
            "top_k_sv": [el for el in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 20, 32, 64, 100, 128, 256, 500, 512, 800, 1000, 1024] if el <= min(num_cands, num_refs)],  # 0 means all singular values
            "bottom_k_sv": [None],  # 0 means all singular values
            "is_reduced": [True],  # Whether to use reduced SVD
        }
        if min(num_cands, num_refs) not in param_grid["top_k_sv"]:
            param_grid["top_k_sv"].append(min(num_cands, num_refs))
    elif args.common.decoder == "normed_svd_mbr":
        param_grid = {
            "top_k_sv": [el for el in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 20, 32, 64, 100, 128, 256, 500, 512, 800, 1000, 1024] if el <= min(num_cands, num_refs)],  # 0 means all singular values
            "bottom_k_sv": [None],
            "is_reduced": [True],  # Whether to use reduced SVD
            "norm_dim": [None, 0],  # Dimension to normalize over
        }
        if min(num_cands, num_refs) not in param_grid["top_k_sv"]:
            param_grid["top_k_sv"].append(min(num_cands, num_refs))
    elif args.common.decoder == "nmf_mbr":
        param_grid = {
            "rank": [el for el in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 20, 32, 64, 100, 128, 256, 500, 512, 800, 1000, 1024] if el <= min(num_cands, num_refs)],
            "beta": [-1.0, 0, 1.0],
            "l1_ratio": [0.0, 0.5, 1.0],
        }
        if min(num_cands, num_refs) not in param_grid["rank"]:
            param_grid["rank"].append(min(num_cands, num_refs))
    elif args.common.decoder == "bootstrap_probabilistic_mbr":
        max_sample_size = num_cands * num_refs
        param_grid = {
            "top_k": [el for el in [2, 3, 4, 5, 6, 7, 8, 9, 10, 16] if el <= max_sample_size],
            "normalize_before_factorization": [True, False],
            "sampling_type": ["greedy_top_k", "cumulative_mass", "edge_k"],
        }
        if min(num_cands, num_refs) not in param_grid["top_k"]:
            param_grid["top_k"].append(min(num_cands, num_refs))
        for k_percent in [0.01, 0.015, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            k_value = int(max_sample_size * k_percent)
            if k_value > 0 and k_value <= max_sample_size and k_value not in param_grid["top_k"]:
                param_grid["top_k"].append(k_value)
    else:
        raise ValueError("Unknown MBR method in hypotheses filename.")


    for hyperparams in tqdm(ParameterGrid(param_grid), desc=f"Hyperparameter Search for hyp.{num_cands} - ref.{num_refs}"):
        print(hyperparams)
        output_path = args.common.output + f'.params-{"-".join([f"{k}-{v}" for k,v in hyperparams.items()])}'
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                data = f.readlines()
            if len(data) == num_sents:
                print(f"Output file {output_path} already exists with correct number of sentences. Skipping...")
                continue
        run(hyperparams)

    #####################
    
    if not args.common.quiet:
        try:
            statistics = timer.aggregate().result(num_sents)
        except Exception as e:
            print(f"Failed to get statistics: {e}", file=args.common.report)
        else:
            table = tabulate(
                statistics,
                headers="keys",
                tablefmt=args.common.report_format,
                floatfmt=f".{args.common.width}f",
            )
            print(table, file=args.common.report)


def cli_main():
    args = get_argparser().parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
