import os
import json
import torch
from tqdm import tqdm
from ast import literal_eval
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from vis_plots import process_matrices
from src.svd_mbr.modules.z_score_norm import z_score_norm

hyps_path = "/var/autofs/cl/home2/share/mbrs/generated_text"
result_path = "/var/autofs/cl/home2/share/mbrs/results"
score_path = "/var/autofs/cl/home2/share/mbrs/scores"
validated_path = "/var/autofs/cl/home2/share/mbrs/validated"
metrics = ["bleu", "chrf", "comet", "bleurt", "cometkiwi"]
mode_types = ["mbr","svd_mbr", "normed_svd_mbr", "nmf_mbr"]
decoding_types = ["eps", "topp", "ancestral"]
cand_counts = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

def compile_scores(hyp_metadata, score_metadata, task, dataset, hyp_sampling, hyp_count, selected_indexes):
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
        except IndexError:
            print(f"Index error found for row: {row}")
            selected_scores = np.array([0.0])  # or some default value
        scores[metric] = np.mean(selected_scores)
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

def compile_no_mbr_scores(hyp_metadata, score_metadata):
    no_mbr_data = []
    for row in tqdm(hyp_metadata.itertuples(), desc="Compiling no MBR scores"):
        row_data = {
            "task": row.task,
            "dataset": row.dataset,
            "hyp_sampling": row.hyp_sampling,
            "hyp_count": row.hyp_count,
            "method": "no_mbr",
            "ref_sampling": "None",
            "ref_count": None,
            "util_function": None,
            "params": None
        }
        selected_indices = get_without_mbr_indices(row)
        row_data["selected_indexes"] = selected_indices
        scores = compile_scores(hyp_metadata, score_metadata, row.task, row.dataset, row.hyp_sampling, row.hyp_count, selected_indices)
        row_data.update(scores)
        no_mbr_data.append(row_data)
    return pd.DataFrame(no_mbr_data)

def compile_mbr_scores(hyp_metadata, score_metadata, validated_metadata, mbr_type):
    mbr_data = []
    validated_subset = validated_metadata.loc[validated_metadata["decompose_mode"] == mbr_type].copy()
    for row in tqdm(validated_subset.itertuples(), desc=f"Compiling {mbr_type} MBR scores"):
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
        if mbr_type == "mbr":
            selected_indices = get_vanilla_mbr_indices(row)
        else:
            selected_indices = get_validated_mbr_indices(row)
        row_data["selected_indexes"] = selected_indices
        scores = compile_scores(hyp_metadata, score_metadata, row.task, row.dataset, row.hyp_sampling, row.hyp_count, selected_indices)
        row_data.update(scores)
        mbr_data.append(row_data)
    return pd.DataFrame(mbr_data)

def compile_oracle_scores(hyp_metadata, score_metadata):
    oracle_data = []
    for row in tqdm(score_metadata.itertuples(), desc="Compiling oracle scores"):
        row_data = {
            "task": row.task,
            "dataset": row.dataset,
            "hyp_sampling": row.hyp_sampling,
            "hyp_count": row.hyp_count,
            "method": f"oracle_{row.metric_name}",
            "ref_sampling": "None",
            "ref_count": None,
            "util_function": None,
            "params": None
        }
        selected_indices = get_oracles(row)
        row_data["selected_indexes"] = selected_indices
        scores = compile_scores(hyp_metadata, score_metadata, row.task, row.dataset, row.hyp_sampling, row.hyp_count, selected_indices)
        row_data.update(scores)
        oracle_data.append(row_data)
    return pd.DataFrame(oracle_data)

# def get_data_by_row(row):
#     res = {}
#     file_name = f'{row["dataset"]}.tgt.{row["utility_function"]}-{row["decompose_mode"]}.{row["hyp_type"]}-{row["ref_type"]}.params-{"-".join([f"{k}-{v}" for k, v in row["param"].items()])}'
#     with open(os.path.join(result_path, file_name), "r") as f:
#         res["decode_content"] = [json.loads(line) for line in f.readlines()]
#     for metric in metrics:
#         score_file = f'{file_name}.{metric}'
#         res[f"score_{metric}"] = json.load(open(os.path.join(score_path, score_file), "r"))["score"]
#     return res

# def compile_scores():
#     all_scores_file = os.listdir(score_path)
#     scores_list = []
#     for score_file in tqdm(all_scores_file, desc="Processing Score Files"):
#         if score_file.endswith(".yaml") or "comet-" in score_file:
#             continue
#         hyp_type, hyp_count, metric = score_file.split(".")[-3:]
#         scores = []
#         hyp_filename = score_file.replace(f".{metric}", "")
#         lprobs_filename = hyp_filename + ".lprobs"
#         with open(os.path.join(hyps_path, hyp_filename), "r") as f:
#             hyps = [line.strip() for line in f.readlines()]
#         with open(os.path.join(hyps_path, lprobs_filename), "r") as f:
#             lprobs = [line.strip() for line in f.readlines()]
#         with open(os.path.join(score_path, score_file), "r") as f:
#             for line in f.readlines():
#                 try:
#                     score = json.loads(line.strip())["score"]
#                 except:
#                     score = literal_eval(line)
#                 if isinstance(score, list) and len(score) == 1:
#                     scores.append(score[0])
#                 else:
#                     scores.append(score)
#         lprobs_idx = np.argmax(np.array(lprobs).astype(float).reshape(-1, int(hyp_count)), axis=1)
#         assert lprobs_idx.shape[0] == len(hyps) // int(hyp_count), "Length mismatch: lprobs_idx {}, hyps {}".format(lprobs_idx.shape[0], len(hyps))
#         assert len(hyps) == len(scores)
#         scores_list.append({
#             "hyp_type": f"{hyp_type}.{hyp_count}",
#             "metric": metric,
#             "scores": scores,
#             "hyps": hyps,
#             "lprobs_idx": lprobs_idx.tolist()
#         })
#     scores_df = pd.DataFrame(scores_list)
#     # assert scores_df.shape[0] == len(cand_counts) * len(metrics), "Scores dataframe shape mismatch. Got {}, expected {}".format(scores_df.shape[0], len(cand_counts) * len(metrics))

#     scores_pivot = scores_df.pivot(index='hyp_type', columns='metric', values='scores')
#     scores_pivot.reset_index(inplace=True)
#     scores_df = scores_df.drop_duplicates(["hyp_type"])[["hyp_type", "hyps", "lprobs_idx"]].merge(scores_pivot, on="hyp_type")

#     for metric in metrics:
#         scores_df[f"{metric}_oracle_idx"] = scores_df[["hyp_type", metric]].apply(lambda x: np.argmax(np.array(x[metric]).astype(float).reshape(-1, int(x["hyp_type"].split(".")[-1])), axis=1).tolist(), axis=1)
#     return scores_df

# def compile_decodes():
#     decode_combinations = []
#     for decompose_mode in mode_types:
#         for decode_mode in decoding_types:
#             for i, hyp_count in enumerate(cand_counts):
#                 for j, ref_count in enumerate(cand_counts):
#                     batch_number = i * len(cand_counts) + j + 1
                    
#                     if decompose_mode == "svd_mbr":
#                         param_grid = {
#                             "top_k_sv": [el for el in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 500, 800, 1000, min(hyp_count, ref_count)] if el <= min(hyp_count, ref_count)],  # 0 means all singular values
#                             "bottom_k_sv": [None],  # 0 means all singular values
#                             "is_reduced": [True],  # Whether to use reduced SVD
#                         }
#                     elif decompose_mode == "normed_svd_mbr":
#                         param_grid = {
#                             "top_k_sv": [el for el in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 500, 800, 1000, min(hyp_count, ref_count)] if el <= min(hyp_count, ref_count)],  # 0 means all singular values
#                             "bottom_k_sv": [None],
#                             "is_reduced": [True],  # Whether to use reduced SVD
#                             "norm_dim": [None, 0],  # Dimension to normalize over
#                         }
#                     elif decompose_mode == "nmf_mbr":
#                         param_grid = {
#                             "rank": [el for el in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 500, 800, 1000, min(hyp_count, ref_count)] if el <= min(hyp_count, ref_count)],
#                             "beta": [-1.0, 0, 1.0],
#                             "l1_ratio": [0.0, 0.5, 1.0],
#                         }
#                     else:
#                         param_grid = {}
                    
#                     all_param_combinations = list(ParameterGrid(param_grid))
#                     for param in all_param_combinations:
#                         decode_combinations.append({
#                             "task": "translation",
#                             "dataset": "wmt22-ende",
#                             "utility_function": "bleurt",
#                             "decompose_mode": decompose_mode,
#                             "hyp_type": f"eps.{hyp_count}",
#                             "ref_type": f"{decode_mode}.{ref_count}",
#                             "param": param,
#                             "batch_number": batch_number,
#                             "decode_exist": False,
#                         })
#     decode_df = pd.DataFrame(decode_combinations)
#     return decode_df

# def update_decode_status(row):
#     file_name = f'{row["dataset"]}.tgt.{row["utility_function"]}-{row["decompose_mode"]}.{row["hyp_type"]}-{row["ref_type"]}'
#     if row["param"]:
#         file_name += f'.params-{"-".join([f"{k}-{v}" for k, v in row["param"].items()])}'
#     if os.path.exists(os.path.join(validated_path, file_name)):
#         return True
#     else:
#         return False

# def get_combination_result(row, all_hyps):
#     default_result = {
#         "avg_l2_norm": 0.0,
#         "selected_indexes": [],
#     }
#     if row["decode_exist"] == False:
#         return default_result
#     baseline_file_name = f'{row["dataset"]}.tgt.{row["utility_function"]}-mbr.{row["hyp_type"]}-{row["ref_type"]}'
    
#     # Handle comet-mbr baseline
#     if not row["param"]:
#         file_name = baseline_file_name
#         original_matrix = torch.load(os.path.join(validated_path, baseline_file_name + ".original_matrix.pt"))

#         with open(os.path.join(validated_path, baseline_file_name), "r") as f:
#             chosen_sentences = [line.strip() for line in f.readlines()]
#         all_hyps = all_hyps.reshape(-1, int(row["hyp_type"].split(".")[-1]))
#         selected_indices = []
#         for i, chosen in enumerate(chosen_sentences):
#             hyp_candidates = all_hyps[i]
#             selected_idx = np.where(hyp_candidates == chosen)[0][0]
#             selected_indices.append(int(selected_idx))
#         default_result["selected_indexes"] = selected_indices
#         default_result["avg_l2_norm"] = 0.0
#         return default_result
    
#     file_name = f'{row["dataset"]}.tgt.{row["utility_function"]}-{row["decompose_mode"]}.{row["hyp_type"]}-{row["ref_type"]}.params-{"-".join([f"{k}-{v}" for k, v in row["param"].items()])}'
#     original_matrix = torch.load(os.path.join(validated_path, baseline_file_name + ".original_matrix.pt"))
#     if original_matrix.shape[0] != all_hyps.shape[0]//int(row["hyp_type"].split(".")[-1]):
#         print(baseline_file_name, "Original matrix shape does not match hyps shape")
#     if row["decompose_mode"] == "normed_svd_mbr":
#         normed_matrix = []
#         for mat_cell in original_matrix:
#             temp_normed = z_score_norm(mat_cell, dim=row["param"]["norm_dim"])
#             normed_matrix.append(temp_normed)
#         original_matrix = torch.stack(normed_matrix)
#     try:
#         with open(os.path.join(validated_path, file_name + ".mbr_data"), "r") as f:
#             selected_indices = [json.loads(line)["selected_idx"] for line in f.readlines()]
        
#         decomposed_matrix = torch.load(os.path.join(validated_path, file_name + ".decomposed_matrix.pt"))
#     except:
#         with open(os.path.join(result_path, file_name), "r") as f:
#             decode_content = [json.loads(line) for line in f.readlines()]
        
#         selected_indices = [content["selected_idx"] for content in decode_content]

#         decomposed_matrix = torch.stack([torch.tensor(content["decomposed_matrix"]["data"]) for content in decode_content])    
    
#     matrix_process = process_matrices(original_matrix, decomposed_matrix)   
#     default_result["selected_indexes"] = selected_indices
#     default_result["avg_l2_norm"] = matrix_process["avg_error_value"].item()
    
#     return default_result

# def get_scores(indices, scores_series):
#     default_score_dict = {f"score_{metric}": 0.0 for metric in metrics}
#     hyp_count = int(scores_series["hyp_type"].iloc[0].split(".")[-1])
#     if len(indices) == 0:
#         return default_score_dict
#     for metric in metrics:
#         all_metric_scores = np.array(scores_series[metric].iloc[0]).astype(float).reshape(-1, int(hyp_count))
#         selected_scores = all_metric_scores[np.arange(all_metric_scores.shape[0]), np.array(indices).astype(int)]
#         default_score_dict[f"score_{metric}"] = np.mean(selected_scores)
        
#     return default_score_dict