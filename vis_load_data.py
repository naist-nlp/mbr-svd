import os
import json
import torch
from tqdm import tqdm
from ast import literal_eval
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from vis_matrices import process_matrices
from src.svd_mbr.modules.z_score_norm import z_score_norm

hyps_path = "/var/autofs/cl/home2/share/mbrs/generated_text/translation"
result_path = "/var/autofs/cl/home2/share/mbrs/results/translation"
score_path = "/var/autofs/cl/home2/share/mbrs/scores/translation"
validated_path = "/var/autofs/cl/home2/share/mbrs/validated/translation"
metrics = ["bleu", "chrf", "comet", "bleurt", "cometkiwi"]
mode_types = ["mbr","svd_mbr", "normed_svd_mbr", "nmf_mbr"]
decoding_types = ["eps", "topp", "ancestral"]
cand_counts = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

def get_data_by_row(row):
    res = {}
    file_name = f'{row["dataset"]}.tgt.{row["utility_function"]}-{row["decompose_mode"]}.{row["hyp_type"]}-{row["ref_type"]}.params-{"-".join([f"{k}-{v}" for k, v in row["param"].items()])}'
    with open(os.path.join(result_path, file_name), "r") as f:
        res["decode_content"] = [json.loads(line) for line in f.readlines()]
    for metric in metrics:
        score_file = f'{file_name}.{metric}'
        res[f"score_{metric}"] = json.load(open(os.path.join(score_path, score_file), "r"))["score"]
    return res

def compile_scores():
    all_scores_file = os.listdir(score_path)
    scores_list = []
    for score_file in tqdm(all_scores_file, desc="Processing Score Files"):
        if score_file.endswith(".yaml") or "comet-" in score_file:
            continue
        hyp_type, hyp_count, metric = score_file.split(".")[-3:]
        scores = []
        hyp_filename = score_file.replace(f".{metric}", "")
        lprobs_filename = hyp_filename + ".lprobs"
        with open(os.path.join(hyps_path, hyp_filename), "r") as f:
            hyps = [line.strip() for line in f.readlines()]
        with open(os.path.join(hyps_path, lprobs_filename), "r") as f:
            lprobs = [line.strip() for line in f.readlines()]
        with open(os.path.join(score_path, score_file), "r") as f:
            for line in f.readlines():
                try:
                    score = json.loads(line.strip())["score"]
                except:
                    score = literal_eval(line)
                if isinstance(score, list) and len(score) == 1:
                    scores.append(score[0])
                else:
                    scores.append(score)
        lprobs_idx = np.argmax(np.array(lprobs).astype(float).reshape(-1, int(hyp_count)), axis=1)
        assert lprobs_idx.shape[0] == len(hyps) // int(hyp_count), "Length mismatch: lprobs_idx {}, hyps {}".format(lprobs_idx.shape[0], len(hyps))
        assert len(hyps) == len(scores)
        scores_list.append({
            "hyp_type": f"{hyp_type}.{hyp_count}",
            "metric": metric,
            "scores": scores,
            "hyps": hyps,
            "lprobs_idx": lprobs_idx.tolist()
        })
    scores_df = pd.DataFrame(scores_list)
    # assert scores_df.shape[0] == len(cand_counts) * len(metrics), "Scores dataframe shape mismatch. Got {}, expected {}".format(scores_df.shape[0], len(cand_counts) * len(metrics))

    scores_pivot = scores_df.pivot(index='hyp_type', columns='metric', values='scores')
    scores_pivot.reset_index(inplace=True)
    scores_df = scores_df.drop_duplicates(["hyp_type"])[["hyp_type", "hyps", "lprobs_idx"]].merge(scores_pivot, on="hyp_type")

    for metric in metrics:
        scores_df[f"{metric}_oracle_idx"] = scores_df[["hyp_type", metric]].apply(lambda x: np.argmax(np.array(x[metric]).astype(float).reshape(-1, int(x["hyp_type"].split(".")[-1])), axis=1).tolist(), axis=1)
    return scores_df

def compile_decodes():
    decode_combinations = []
    for decompose_mode in mode_types:
        for decode_mode in decoding_types:
            for i, hyp_count in enumerate(cand_counts):
                for j, ref_count in enumerate(cand_counts):
                    batch_number = i * len(cand_counts) + j + 1
                    
                    if decompose_mode == "svd_mbr":
                        param_grid = {
                            "top_k_sv": [el for el in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 500, 800, 1000, min(hyp_count, ref_count)] if el <= min(hyp_count, ref_count)],  # 0 means all singular values
                            "bottom_k_sv": [None],  # 0 means all singular values
                            "is_reduced": [True],  # Whether to use reduced SVD
                        }
                    elif decompose_mode == "normed_svd_mbr":
                        param_grid = {
                            "top_k_sv": [el for el in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 500, 800, 1000, min(hyp_count, ref_count)] if el <= min(hyp_count, ref_count)],  # 0 means all singular values
                            "bottom_k_sv": [None],
                            "is_reduced": [True],  # Whether to use reduced SVD
                            "norm_dim": [None, 0],  # Dimension to normalize over
                        }
                    elif decompose_mode == "nmf_mbr":
                        param_grid = {
                            "rank": [el for el in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 500, 800, 1000, min(hyp_count, ref_count)] if el <= min(hyp_count, ref_count)],
                            "beta": [-1.0, 0, 1.0],
                            "l1_ratio": [0.0, 0.5, 1.0],
                        }
                    else:
                        param_grid = {}
                    
                    all_param_combinations = list(ParameterGrid(param_grid))
                    for param in all_param_combinations:
                        decode_combinations.append({
                            "task": "translation",
                            "dataset": "wmt22-ende",
                            "utility_function": "bleurt",
                            "decompose_mode": decompose_mode,
                            "hyp_type": f"eps.{hyp_count}",
                            "ref_type": f"{decode_mode}.{ref_count}",
                            "param": param,
                            "batch_number": batch_number,
                            "decode_exist": False,
                        })
    decode_df = pd.DataFrame(decode_combinations)
    return decode_df

def update_decode_status(row):
    file_name = f'{row["dataset"]}.tgt.{row["utility_function"]}-{row["decompose_mode"]}.{row["hyp_type"]}-{row["ref_type"]}'
    if row["param"]:
        file_name += f'.params-{"-".join([f"{k}-{v}" for k, v in row["param"].items()])}'
    if os.path.exists(os.path.join(validated_path, file_name)):
        return True
    else:
        return False

def get_combination_result(row, all_hyps):
    default_result = {
        "avg_l2_norm": 0.0,
        "selected_indexes": [],
    }
    if row["decode_exist"] == False:
        return default_result
    baseline_file_name = f'{row["dataset"]}.tgt.{row["utility_function"]}-mbr.{row["hyp_type"]}-{row["ref_type"]}'
    
    # Handle comet-mbr baseline
    if not row["param"]:
        file_name = baseline_file_name
        original_matrix = torch.load(os.path.join(validated_path, baseline_file_name + ".original_matrix.pt"))

        with open(os.path.join(validated_path, baseline_file_name), "r") as f:
            chosen_sentences = [line.strip() for line in f.readlines()]
        all_hyps = all_hyps.reshape(-1, int(row["hyp_type"].split(".")[-1]))
        selected_indices = []
        for i, chosen in enumerate(chosen_sentences):
            hyp_candidates = all_hyps[i]
            selected_idx = np.where(hyp_candidates == chosen)[0][0]
            selected_indices.append(int(selected_idx))
        default_result["selected_indexes"] = selected_indices
        default_result["avg_l2_norm"] = 0.0
        return default_result
    
    file_name = f'{row["dataset"]}.tgt.{row["utility_function"]}-{row["decompose_mode"]}.{row["hyp_type"]}-{row["ref_type"]}.params-{"-".join([f"{k}-{v}" for k, v in row["param"].items()])}'
    original_matrix = torch.load(os.path.join(validated_path, baseline_file_name + ".original_matrix.pt"))
    if original_matrix.shape[0] != all_hyps.shape[0]//int(row["hyp_type"].split(".")[-1]):
        print(baseline_file_name, "Original matrix shape does not match hyps shape")
    if row["decompose_mode"] == "normed_svd_mbr":
        normed_matrix = []
        for mat_cell in original_matrix:
            temp_normed = z_score_norm(mat_cell, dim=row["param"]["norm_dim"])
            normed_matrix.append(temp_normed)
        original_matrix = torch.stack(normed_matrix)
    try:
        with open(os.path.join(validated_path, file_name + ".mbr_data"), "r") as f:
            selected_indices = [json.loads(line)["selected_idx"] for line in f.readlines()]
        
        decomposed_matrix = torch.load(os.path.join(validated_path, file_name + ".decomposed_matrix.pt"))
    except:
        with open(os.path.join(result_path, file_name), "r") as f:
            decode_content = [json.loads(line) for line in f.readlines()]
        
        selected_indices = [content["selected_idx"] for content in decode_content]

        decomposed_matrix = torch.stack([torch.tensor(content["decomposed_matrix"]["data"]) for content in decode_content])    
    
    matrix_process = process_matrices(original_matrix, decomposed_matrix)   
    default_result["selected_indexes"] = selected_indices
    default_result["avg_l2_norm"] = matrix_process["avg_error_value"].item()
    
    return default_result

def get_scores(indices, scores_series):
    default_score_dict = {f"score_{metric}": 0.0 for metric in metrics}
    hyp_count = int(scores_series["hyp_type"].iloc[0].split(".")[-1])
    if len(indices) == 0:
        return default_score_dict
    for metric in metrics:
        all_metric_scores = np.array(scores_series[metric].iloc[0]).astype(float).reshape(-1, int(hyp_count))
        selected_scores = all_metric_scores[np.arange(all_metric_scores.shape[0]), np.array(indices).astype(int)]
        default_score_dict[f"score_{metric}"] = np.mean(selected_scores)
        
    return default_score_dict