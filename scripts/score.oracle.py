#!/usr/bin/env python3

import enum
import json
import logging
import os
import sys
from argparse import Namespace
from dataclasses import asdict, dataclass
from typing import Sequence
import torch

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

import simple_parsing
from simple_parsing import choice, field, flag
from simple_parsing.wrappers import dataclass_wrapper

from mbrs import registry
from mbrs.args import ArgumentParser
from mbrs.metrics import Metric, MetricReferenceless, get_metric

import json
from tqdm import tqdm

class Format(enum.Enum):
    plain = "plain"
    json = "json"


@dataclass
class CommonArguments:
    """Common arguments."""

    # Hypotheses file.
    hypotheses: str = field(positional=True)
    # Sources file.
    sources: str | None = field(default=None, alias=["-s"])
    # References file.
    references: str | None = field(default=None, alias=["-r"], nargs="+")
    # Output format.
    format: Format = choice(Format, default=Format.json)
    # Output file.
    output: str | None = field(default="-", alias=["-o"])
    # Type of the metric.
    metric: str = field(
        default="bleu",
        metadata={"choices": registry.get_registry(Metric | MetricReferenceless)},
    )
    # No verbose information and report.
    quiet: bool = flag(default=False)
    # Number of digits for values of float point.
    width: int = field(default=1, alias=["-w"])


def get_argparser(args: Sequence[str] | None = None) -> ArgumentParser:
    meta_parser = ArgumentParser(add_help=False, add_config_path_arg=True)
    meta_parser.add_arguments(CommonArguments, "common")
    for _field in meta_parser._wrappers[0].fields:
        _field.required = False
    known_args, _ = meta_parser.parse_known_args(args=args)

    parser = ArgumentParser(add_help=True, add_config_path_arg=True)
    parser.add_arguments(CommonArguments, "common")
    parser.add_arguments(
        get_metric(known_args.common.metric).Config, "metric", prefix="metric."
    )
    return parser

def format_argparser() -> ArgumentParser:
    parser = get_argparser()
    parser.preprocess_parser()
    return parser

def main(args: Namespace) -> None:
    if os.path.exists(args.common.hypotheses) is False:
        print(f"Hypotheses file {args.common.hypotheses} does not exist. Skipping this hypotheses.")
        return

    with open(args.common.hypotheses, mode="r") as f:
        hypotheses = []
        cand_count = int(args.common.hypotheses.split(".")[-1])
        for line in f:
            try:
                hypothesis = line.strip()
                hypotheses.append(hypothesis)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")

    num_sents = len(hypotheses)

    sources = None
    if args.common.sources is not None:
        sources = []
        with open(args.common.sources, mode="r") as f:
            for line in f.readlines():
                for _ in range(cand_count):
                    sources.append(line)
        assert num_sents == len(sources), f"{num_sents} != {len(sources)}"

    references_lists: list[list[str]] | None = None
    if args.common.references is not None:
        references_lists = []
        for references_path in args.common.references:
            with open(references_path, mode="r") as f:
                references = []
                for line in f.readlines():
                    for _ in range(cand_count):
                        references.append(line)
            assert num_sents == len(references)
            references_lists.append(references)

    metric = get_metric(args.common.metric)(args.metric)

    if isinstance(metric, MetricReferenceless):
        assert sources is not None
        all_scores = metric.scores(hypotheses, sources=sources)
        all_scores = all_scores.view(-1, 1)
        assert all_scores.shape[0] == num_sents
    else:
        all_scores = []
        assert references_lists is not None
        for references_list in references_lists:
            if cand_count > 3:
                batch_size = 16
                scores_batches = []
                for i in tqdm(range(0, len(hypotheses), batch_size), desc="Scoring batches"):
                    batch_hypotheses = hypotheses[i:i+batch_size]
                    batch_references = references_list[i:i+batch_size]
                    batch_sources = sources[i:i+batch_size] if sources is not None else None
                    batch_scores = metric.scores(batch_hypotheses, batch_references, batch_sources)
                    scores_batches.append(batch_scores)
                scores = torch.cat(scores_batches, dim=0)
            else:
                scores = metric.scores(hypotheses, references_list, sources)
            all_scores.append(scores)
        all_scores = torch.stack(all_scores, dim=1)

    assert all_scores.shape[0] == num_sents, f"{all_scores.shape} != {num_sents}"

    with open(args.filename, mode="w") as f:
        for score in all_scores.to("cpu").tolist():
            if args.common.format == Format.plain:
                print(score, sep="\n", file=f)
            elif args.common.format == Format.json:
                json.dump({
                    "name": args.common.metric,
                    "score": score,
                    "metric_cfg": asdict(args.metric),
                }, f)
                f.write("\n")

def cli_main():
    args = get_argparser().parse_args()

    args.template_hypotheses = args.common.hypotheses
    args.filename = args.common.output

    main(args)


if __name__ == "__main__":
    cli_main()