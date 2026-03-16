import torch 
import os
from ast import literal_eval
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Split results and move to new folder")
    parser.add_argument(
        "--hyp_count",
        type=int,
        help="hyp count",
    )
    parser.add_argument(
        "--ref_count",
        type=int,
        help="ref count",
    )
    return parser.parse_args()

def run(hyp_count: int, ref_count: int):

    result_path = "/var/autofs/cl/home2/share/mbrs/results/translation"
    new_result_path = "/var/autofs/cl/home2/share/mbrs/validated/translation"
    os.makedirs(new_result_path, exist_ok=True)

    scores_df = pd.read_csv("results/20260108/compiled_scores.csv")
    scores_df = scores_df.map(lambda x: literal_eval(x) if isinstance(x, str) and x.startswith("[") else x)

    all_results_file = os.listdir(result_path)
    for result_file in tqdm(all_results_file, desc="Processing Result Files"):
        if os.path.exists(os.path.join(new_result_path, result_file)):
            continue
        if f"eps.{hyp_count}-eps.{ref_count}.params" not in result_file:
            continue
        print(result_file)
        all_hyps = scores_df[scores_df["hyp_type"] == f"eps.{hyp_count}"]["hyps"].values[0]
        all_hyps = np.array(all_hyps).reshape(-1, int(hyp_count))
        file_data = []
        with open(os.path.join(result_path, result_file), "r") as f:
            for line in f.readlines():
                row = json.loads(line.strip())
                file_data.append(row)
        row_data = pd.DataFrame(file_data)
        selected_indices = row_data["selected_idx"].tolist()
        selected_sentences = row_data["sentence"].tolist()

        assert len(selected_sentences) == all_hyps.shape[0], "Length mismatch: selected_sentences {}, all_hyps {}".format(len(selected_sentences), all_hyps.shape[0])
        # found_selected_indices = np.where(all_hyps == np.array(selected_sentences).reshape(-1, 1))
        # grouped_indices = {}
        # for row, col in zip(*found_selected_indices):
        #     if row not in grouped_indices:
        #         grouped_indices[row] = []
        #     grouped_indices[row].append(col)
        # print(grouped_indices.keys())
        # for row, col_idx in enumerate(selected_indices):
        #     if col_idx not in grouped_indices[row]:
        #         print(f"Warning: selected index {col_idx} not found in grouped indices {grouped_indices[row]} for row {row}")
        #         continue
        # print(selected_indices, type(selected_indices))
        assert_error = False
        for row, col_idx in enumerate(selected_indices):
            expected_sentence = all_hyps[row, col_idx]
            actual_sentence = selected_sentences[row]
            try:
                assert expected_sentence == actual_sentence, f"Validation failed at row {row}: expected '{expected_sentence}', got '{actual_sentence}'"
            except AssertionError as e:
                print(e)
                assert_error = True
                continue
        if assert_error:
            print(f"Skipping file {result_file} due to validation errors.")
            continue
        # -- Validation over --

        with open(os.path.join(new_result_path, result_file), "w") as f:
            for sentence in selected_sentences:
                print(sentence, file=f)
        with open(os.path.join(new_result_path, result_file+".mbr_data"), "w") as f:
            mbr_data_df = row_data[["selected_idx", "rank", "expected_score"]].copy()
            for row in mbr_data_df.to_dict(orient="records"):
                print(json.dumps(row), file=f)

        for col in row_data.columns:
            if col in ["sentence", "selected_idx", "rank", "expected_score"]:
                continue
            data_list = row_data[col].tolist()
            if data_list[0] is None or len(data_list) == 0:
                continue
            elif type(data_list[0]["data"]) == str and data_list[0]["data"].startswith("["):
                data_list = [literal_eval(x)["data"] for x in data_list]

            data_list = torch.tensor([row["data"] for row in data_list])
            torch.save(data_list, os.path.join(new_result_path, result_file+f".{col}.pt"))
        
        os.remove(os.path.join(result_path, result_file))

def main():
    args = parse_args()
    hyp_count = args.hyp_count
    ref_count = args.ref_count
    print(hyp_count, ref_count)
    run(hyp_count, ref_count)

if __name__ == "__main__":
    main()