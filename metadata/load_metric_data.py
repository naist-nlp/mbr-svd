import json
from mbrs import metrics
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from mbrs.metrics import Metric, get_metric
from ast import literal_eval


data_path = "/var/autofs/cl/home2/share/mbrs/data"
hyps_path = "/var/autofs/cl/home2/share/mbrs/generated_text"
result_path = "/var/autofs/cl/home2/share/mbrs/results"
score_path = "/var/autofs/cl/home2/share/mbrs/scores"
validated_path = "/var/autofs/cl/home2/share/mbrs/validated"
# validated_path = "/var/autofs/cl/home2/share/mbrs/ensemble"
# analysis_path = "/var/autofs/cl/home2/share/mbrs/analysis"
translation_metrics = ["bleu", "chrf", "comet", "bleurt", "cometkiwi", "bleu_corpus", "bertscore"]
summarization_metrics = ["rouge_1", "rouge_2", "rouge_l", "rouge_lsum", "bertscore"]
metrics_by_task = {
    "translation": translation_metrics,
    "summarization": summarization_metrics
}
mode_types = ["mbr","svd_mbr", "normed_svd_mbr", "nmf_mbr"]
decoding_types = ["eps", "topp", "ancestral"]
cand_counts = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

def calculate_bleu_corpus(selected_text, references, tgt_lang):
    metric_conf = get_metric("bleu").Config(effective_order=True, trg_lang=tgt_lang)
    corpus_score = get_metric("bleu")(metric_conf).corpus_score(selected_text, [references])
    return corpus_score

def compile_scores(hyp_metadata, score_metadata, task, dataset, hyp_sampling, hyp_count, selected_indexes, selected_text_file=None, calc_bleu_corpus=True):
    combined_df = hyp_metadata.merge(score_metadata, on=["task", "dataset", "hyp_sampling", "hyp_count"], how="inner")
    combined_df = combined_df.loc[
        (combined_df["task"] == task) &
        (combined_df["dataset"] == dataset) &
        (combined_df["hyp_sampling"] == hyp_sampling) &
        (combined_df["hyp_count"] == hyp_count)
    ]
    scores = {}
    for row in combined_df.itertuples():
        metric = row.metric_name
        filename = f"{score_path}/{row.task}/{row.score_file}"
        with open(filename, "r") as f:
            row_score = [json.loads(line)["score"][0] for line in f.readlines()]
            row_score = np.array(row_score).astype(float).reshape(-1, int(hyp_count))
        f.close()
        try:        
            selected_scores = row_score[np.arange(row_score.shape[0]), np.array(selected_indexes).astype(int)]
        except IndexError as e:
            print(f"Index error found for row: {row}. Error: {e}")
            selected_scores = np.array([0.0])  # or some default value
        scores[metric] = np.mean(selected_scores)
    
    # Handle bleu corpus calculation
    if task in ["translation"] and calc_bleu_corpus:
        refs_filename = f"{data_path}/{task}/{dataset}.tgt"
        try:
            if selected_text_file is not None and selected_text_file != "line count mismatch":
                with open(f"{validated_path}/{task}/{selected_text_file}", "r") as f:
                    hyps = [line for line in f.readlines()]
            else:
                hyps_filename = f"{hyps_path}/{task}/{dataset}.tgt.{hyp_sampling}.{hyp_count}"
                with open(hyps_filename, "r") as f:
                    hyps = [line for line in f.readlines()]
                hyps = np.array(hyps).reshape(-1, int(hyp_count))
                hyps = hyps[np.arange(hyps.shape[0]), np.array(selected_indexes).astype(int)].tolist()
        except IndexError:
            # print(f"Index error found for row: {row}")
            scores["bleu_corpus"] = 0.0  # or some default value
        except Exception as e:
            print(f"error: {e}")
            scores["bleu_corpus"] = 0.0  # or some default value
        else:
            with open(refs_filename, "r") as f:
                references = [line for line in f.readlines()]
            if len(references) != len(hyps):
                print(hyps_filename, refs_filename)
            scores["bleu_corpus"] = calculate_bleu_corpus(hyps, references, dataset[-2:])
    return scores

def get_oracles(row):
    filename = f"{score_path}/{row.task}/{row.score_file}"
    with open(filename, "r") as f:
        row_score = [json.loads(line)["score"][0] for line in f.readlines()]
        row_score = np.array(row_score).astype(float).reshape(-1, int(row.hyp_count))
    f.close()
    oracle_indices = np.argmax(row_score, axis=1).tolist()
    return oracle_indices

def get_without_mbr_indices(row):
    if row.supporting_files.get("lprobs_norm", None) is None:
        return []
    lprobs_filename = row.supporting_files["lprobs_norm"]
    if lprobs_filename == "line count mismatch":
        print(f"Line count mismatch for row: {row}")
        return []
    with open(f"{hyps_path}/{row.task}/{lprobs_filename}", "r") as f:
        lprobs = [line.strip() for line in f.readlines()]
    lprobs_idx = np.argmax(np.array(lprobs).astype(float).reshape(-1, int(row.hyp_count)), axis=1).tolist()
    return lprobs_idx

def get_validated_mbr_indices(row):
    if row.supporting_files.get("mbr_data", None) is None:
        return []
    mbr_data_filename = row.supporting_files["mbr_data"]
    with open(f"{validated_path}/{row.task}/{mbr_data_filename}", "r") as f:
        mbr_data = [json.loads(line) for line in f.readlines()]
    selected_indices = [content["selected_idx"] if type(content["selected_idx"]) is int else content["selected_idx"][0] for content in mbr_data]
    return selected_indices

def get_vanilla_mbr_indices(row):
    if row.supporting_files.get("original_matrix.pt", None) is None:
        return []
    original_matrix_filename = row.supporting_files["original_matrix.pt"]
    original_matrix = torch.load(f"{validated_path}/{row.task}/{original_matrix_filename}")
    selected_indices = torch.argmax(torch.mean(original_matrix, dim=2), dim=1).tolist()
    return selected_indices

def get_other_mbr_indices(row):
    selected_text_file = row.selected_text_file
    if selected_text_file == "line count mismatch":
        print(f"Line count mismatch for row: {row}")
        return []
    with open(f"{validated_path}/{row.task}/{selected_text_file}", "r") as f:
        selected_text = f.readlines()
    f.close()
    with open(f"{hyps_path}/{row.task}/{row.dataset}.tgt.{row.hyp_sampling}.{row.hyp_count}", "r") as f:
        hyps = f.readlines()
    f.close()
    # Turn hyps to 2D list of shape (num_samples, hyp_count)
    hyps = np.array(hyps).reshape(-1, int(row.hyp_count))
    selected_indices = []
    for sample_idx in range(hyps.shape[0]):
        sample_hyps = hyps[sample_idx]
        sample_selected_text = selected_text[sample_idx]
        if sample_selected_text in sample_hyps:
            selected_idx = np.where(sample_hyps == sample_selected_text)[0][0]
        else:
            print(f"Selected text not found in hypotheses for sample index {sample_idx} in row: {row}")
            selected_idx = 0  # Default to the first hypothesis if not found
        selected_indices.append(selected_idx)
    return selected_indices

def compile_no_mbr_scores(hyp_metadata, score_metadata, existing_df=None):
    no_mbr_data = []
    for row in tqdm(hyp_metadata.itertuples(), desc="Compiling no MBR scores", total=len(hyp_metadata)):
        row_data = {
            "task": row.task,
            "dataset": row.dataset,
            "hyp_sampling": row.hyp_sampling,
            "hyp_count": row.hyp_count,
            "method": "no_mbr",
            "ref_sampling": "None",
            "ref_count": None,
            "util_function": None,
            "params": None,
        }
        calc_bleu_corpus = True
        if existing_df is not None:
            existing_row = existing_df.loc[
                (existing_df["task"] == row.task) &
                (existing_df["dataset"] == row.dataset) &
                (existing_df["hyp_sampling"] == row.hyp_sampling) &
                (existing_df["hyp_count"] == row.hyp_count) &
                (existing_df["method"] == "no_mbr")
            ]
            only_missing_scores = score_metadata.copy()
            if not existing_row.empty and existing_row[metrics_by_task[row.task]].notnull().all(axis=None):
                row_data.update(existing_row.iloc[0].to_dict())
                no_mbr_data.append(row_data)
                continue
            elif not existing_row.empty and existing_row[metrics_by_task[row.task]].isnull().any(axis=None):
                row_data.update(existing_row.iloc[0].to_dict())
                missing_metrics = list(set(existing_row.columns[existing_row.isnull().any()].tolist()).intersection(metrics_by_task[row.task]))
                only_missing_scores = score_metadata.loc[score_metadata["metric_name"].isin(missing_metrics)].copy()
                calc_bleu_corpus = True if existing_row["bleu_corpus"].isnull().any() else False
        selected_indices = get_without_mbr_indices(row)
        row_data["selected_indexes"] = selected_indices
        scores = compile_scores(hyp_metadata, only_missing_scores, row.task, row.dataset, row.hyp_sampling, row.hyp_count, selected_indices, calc_bleu_corpus=calc_bleu_corpus)
        row_data.update(scores)
        no_mbr_data.append(row_data)
    return pd.DataFrame(no_mbr_data)

def compile_mbr_scores(hyp_metadata, score_metadata, validated_metadata, mbr_type, existing_df=None):
    mbr_data = []
    validated_subset = validated_metadata.loc[validated_metadata["decompose_mode"] == mbr_type].copy()
    for row in tqdm(validated_subset.itertuples(), desc=f"Compiling {mbr_type} MBR scores", total=len(validated_subset)):
        row_data = {
            "task": row.task,
            "dataset": row.dataset,
            "hyp_sampling": row.hyp_sampling,
            "hyp_count": row.hyp_count,
            "method": row.decompose_mode,
            "ref_sampling": row.ref_sampling,
            "ref_count": row.ref_count,
            "util_function": row.util_function,
            "params": row.params
        }
        selected_indices = []
        calc_bleu_corpus = True
        only_missing_scores = score_metadata.copy()
        if existing_df is not None:
            existing_row = existing_df.loc[
                (existing_df["task"] == row.task) &
                (existing_df["dataset"] == row.dataset) &
                (existing_df["hyp_sampling"] == row.hyp_sampling) &
                (existing_df["hyp_count"] == row.hyp_count) &
                (existing_df["method"] == row.decompose_mode) &
                (existing_df["ref_sampling"] == row.ref_sampling) &
                (existing_df["ref_count"] == row.ref_count) &
                (existing_df["util_function"] == row.util_function)
            ]
            if not pd.isnull(row.params):
                existing_row = existing_row[existing_row["params"] == row.params]
            if not existing_row.empty and existing_row[metrics_by_task[row.task]].notnull().all(axis=None):
                row_data.update(existing_row.iloc[0].to_dict())
                mbr_data.append(row_data)
                continue
            elif not existing_row.empty and existing_row[metrics_by_task[row.task]].isnull().any(axis=None):
                row_data.update(existing_row.iloc[0].to_dict())
                missing_metrics = list(set(existing_row.columns[existing_row.isnull().any()].tolist()).intersection(metrics_by_task[row.task]))
                only_missing_scores = score_metadata.loc[score_metadata["metric_name"].isin(missing_metrics)].copy()
                selected_indices = literal_eval(existing_row["selected_indexes"].iloc[0])
                calc_bleu_corpus = True if existing_row["bleu_corpus"].isnull().any() else False
        if selected_indices == []:
            if mbr_type in ["mbr", "normed_mbr"]:
                selected_indices = get_vanilla_mbr_indices(row)
            elif mbr_type in ["model_mbr", "probabilistic_mbr", "model_norm_mbr"]:
                selected_indices = get_other_mbr_indices(row)
            else:
                selected_indices = get_validated_mbr_indices(row)
        row_data["selected_indexes"] = selected_indices
        scores = compile_scores(hyp_metadata, only_missing_scores, row.task, row.dataset, row.hyp_sampling, row.hyp_count, selected_indices, selected_text_file=row.selected_text_file, calc_bleu_corpus=calc_bleu_corpus)
        row_data.update(scores)
        mbr_data.append(row_data)
    return pd.DataFrame(mbr_data)

def compile_oracle_scores(hyp_metadata, score_metadata, existing_df=None):
    oracle_data = []
    for row in tqdm(score_metadata.itertuples(), desc="Compiling oracle scores", total=len(score_metadata)):
        row_data = {
            "task": row.task,
            "dataset": row.dataset,
            "hyp_sampling": row.hyp_sampling,
            "hyp_count": row.hyp_count,
            "method": f"oracle_{row.metric_name}",
            "ref_sampling": "None",
            "ref_count": None,
            "util_function": None,
            "params": None,
        }
        calc_bleu_corpus = True
        if existing_df is not None:
            existing_row = existing_df.loc[
                (existing_df["task"] == row.task) &
                (existing_df["dataset"] == row.dataset) &
                (existing_df["hyp_sampling"] == row.hyp_sampling) &
                (existing_df["hyp_count"] == row.hyp_count) &
                (existing_df["method"] == f"oracle_{row.metric_name}")
            ]
            only_missing_scores = score_metadata.copy()
            if not existing_row.empty and existing_row[metrics_by_task[row.task]].notnull().all(axis=None):
                row_data.update(existing_row.iloc[0].to_dict())
                oracle_data.append(row_data)
                continue
            elif not existing_row.empty and existing_row[metrics_by_task[row.task]].isnull().any(axis=None):
                row_data.update(existing_row.iloc[0].to_dict())
                missing_metrics = list(set(existing_row.columns[existing_row.isnull().any()].tolist()).intersection(metrics_by_task[row.task]))
                only_missing_scores = score_metadata.loc[score_metadata["metric_name"].isin(missing_metrics)].copy()
                calc_bleu_corpus = True if existing_row["bleu_corpus"].isnull().any() else False

        selected_indices = get_oracles(row)
        row_data["selected_indexes"] = selected_indices
        scores = compile_scores(hyp_metadata, only_missing_scores, row.task, row.dataset, row.hyp_sampling, row.hyp_count, selected_indices, calc_bleu_corpus=calc_bleu_corpus)
        row_data.update(scores)
        oracle_data.append(row_data)
    return pd.DataFrame(oracle_data)