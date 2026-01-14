import os
import json
from tqdm import tqdm
from ast import literal_eval
import numpy as np
import pandas as pd

hyps_path = "/var/autofs/cl/home2/share/mbrs/generated_text/translation"
result_path = "/var/autofs/cl/home2/share/mbrs/results/translation"
score_path = "/var/autofs/cl/home2/share/mbrs/scores/translation"
metrics = ["bleu", "chrf", "comet", "bleurt", "cometkiwi"]

def get_data_by_row(row):
    res = {}
    file_name = f'{row["dataset"]}.tgt.{row["utility_function"]}-{row["decompose_mode"]}.{row["hyp_type"]}-{row["ref_type"]}.params-{"-".join([f"{k}-{v}" for k, v in row["param"].items()])}'
    with open(os.path.join(result_path, file_name), "r") as f:
        res["decode_content"] = [json.loads(line) for line in f.readlines()]
    for metric in metrics:
        score_file = f'{file_name}.{metric}'
        res[f"score_{metric}"] = json.load(open(os.path.join(score_path, score_file), "r"))["score"]
    return res

def compile_scores(cand_counts: list):
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