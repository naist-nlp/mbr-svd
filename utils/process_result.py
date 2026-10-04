from ast import literal_eval
import os
import json
import numpy as np
import pandas as pd
from matplotlib.pyplot import grid
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

hyps_path = "/var/autofs/cl/home2/share/mbrs/generated_text/translation"
result_path = "/var/autofs/cl/home2/share/mbrs/results/translation"
score_path = "/var/autofs/cl/home2/share/mbrs/scores/translation"

mode_types = ["mbr", "svd_mbr", "normed_svd_mbr", "nmf_mbr"]
metrics = ["bleu", "chrf", "comet", "bleurt", "cometkiwi"]
cand_counts = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
dim_max_param = [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100, 500, 800, 1000]

remaining_sbatch_decode = {}
remaining_sbatch_score = {}

columns = ["task", "dataset", "utility_function", "decompose_mode", "hyp_type", "ref_type", "param", "decode_exist", "score_bleu", "score_chrf", "score_comet", "score_bleurt", "score_cometkiwi", "batch_number"]
data_dict = []

all_results_file = os.listdir(result_path)
all_scores_file = os.listdir(score_path)

scores_list = []
for score_file in tqdm(all_scores_file, desc="Processing Score Files"):
    if score_file.endswith(".yaml") or "comet-" in score_file:
        continue
    hyp_type, hyp_count, metric = score_file.split(".")[-3:]
    scores = []
    hyp_filename = score_file.replace(f".{metric}", "")
    with open(os.path.join(hyps_path, hyp_filename), "r") as f:
        hyps = [line.strip() for line in f.readlines()]
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
    assert len(hyps) == len(scores)
    scores_list.append({
        "hyp_type": f"{hyp_type}.{hyp_count}",
        "metric": metric,
        "scores": scores,
        "hyps": hyps
    })
scores_df = pd.DataFrame(scores_list)
assert scores_df.shape[0] == len(cand_counts) * len(metrics), "Scores dataframe shape mismatch. Got {}, expected {}".format(scores_df.shape[0], len(cand_counts) * len(metrics))

scores_pivot = scores_df.pivot(index='hyp_type', columns='metric', values='scores')
scores_pivot.reset_index(inplace=True)
scores_df = scores_df.drop_duplicates(["hyp_type"])[["hyp_type", "hyps"]].merge(scores_pivot, on="hyp_type")

for mode_type in tqdm(mode_types, desc="Mode Types"):
    for hyp_idx,hyp_count in tqdm(enumerate(cand_counts), desc="Hyp Counts", total=len(cand_counts)):
        curr_scores = scores_df[scores_df["hyp_type"] == f"eps.{hyp_count}"].to_dict(orient="records")[0]
        for ref_idx,ref_count in enumerate(cand_counts):
            if "svd_mbr" == mode_type:
                param_grid = {
                    "top_k_sv": [dim for dim in dim_max_param if dim is None or dim <= min(hyp_count, ref_count)],  # 0 means all singular values
                    "bottom_k_sv": [None],  # 0 means all singular values
                    "is_reduced": [True, False],  # Whether to use reduced SVD
                }
            elif "normed_svd_mbr" == mode_type:
                param_grid = {
                    "top_k_sv": [dim for dim in dim_max_param if dim is None or dim <= min(hyp_count, ref_count)],  # 0 means all singular values
                    "bottom_k_sv": [None],
                    "is_reduced": [True, False],  # Whether to use reduced SVD
                    "norm_dim": [None, 0, 1],  # Dimension to normalize over
                }
            elif "nmf_mbr" == mode_type:
                param_grid = {
                    "rank": [dim for dim in dim_max_param if dim is None or dim <= min(hyp_count, ref_count)],
                    "beta": [-1.0, 0, 0.5, 1.0, 1.5, 2.0],
                    "l1_ratio": [0.0, 0.1, 0.5, 0.9, 1.0],
                }
            else:
                param_grid = {}
            filename = f"wmt22-ende.tgt.comet-{mode_type}.eps.{hyp_count}-eps.{ref_count}"
            
            yaml_file = f"{filename}.yaml"
            log_file = f"{filename}.logs"

            decode_yaml_exist = yaml_file in all_results_file
            decode_log_exist = log_file in all_results_file

            if not decode_yaml_exist:
                print(f"Missing YAML file: {yaml_file}")
            if not decode_log_exist:
                print(f"Missing log file: {log_file}")

            grid = ParameterGrid(param_grid)

            for param in grid:
                row = {
                    "task": "translation",
                    "dataset": "wmt22-ende",
                    "utility_function": "comet",
                    "decompose_mode": mode_type,
                    "hyp_type": f"eps.{hyp_count}",
                    "ref_type": f"eps.{ref_count}",
                    "param": param,
                    "batch_number": hyp_idx * len(cand_counts) + ref_idx +1,
                }
                output_filename = filename
                if param:
                    output_filename += f".params-{'-'.join([f'{k}-{v}' for k, v in param.items()])}"
                row["decode_exist"] = output_filename in all_results_file

                if mode_type not in remaining_sbatch_score.keys():
                    remaining_sbatch_score[mode_type] = {}

                if row["decode_exist"]:
                    try:
                        with open(os.path.join(result_path, output_filename), "r") as f:
                            result_lines = np.array([json.loads(line.strip())["sentence"] for line in f.readlines()])
                    except Exception as e:
                        with open(os.path.join(result_path, output_filename), "r") as f:
                            result_lines = np.array(f.readlines())
                    hyps_subset = np.array(curr_scores["hyps"])
                    chosen_idx = np.where(np.isin(np.array(hyps_subset), np.char.strip(result_lines)))[0]
                    for metric in metrics:
                        scores_subset = np.array(curr_scores[metric])
                        scores = [np.array(scores_subset)[chosen_idx].tolist()]
                        mean_score = np.mean(scores)
                        
                        row[f"score_{metric}"] = mean_score
                
                data_dict.append(row)
    decode_df = pd.DataFrame(data_dict)
    os.makedirs("results/20260109", exist_ok=True)
    decode_df.to_csv("results/20260109/translation_mbr_results.csv", index=False)