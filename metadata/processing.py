from math import e
import os
import json
import pandas as pd
from tqdm import tqdm
from ast import literal_eval

hyps_path = "/var/autofs/cl/home2/share/mbrs/generated_text"
cache_path = "/var/autofs/cl/home2/share/mbrs/cache"
score_path = "/var/autofs/cl/home2/share/mbrs/scores"
validated_path = "/var/autofs/cl/home2/share/mbrs/validated"
# validated_path = "/var/autofs/cl/home2/share/mbrs/ensemble"
dataset_path = "/var/autofs/cl/home2/share/mbrs/data"

dataset_counts = {}

def get_dataset_count():
    global dataset_counts
    for dir in os.listdir(dataset_path):
        for filename in os.listdir(os.path.join(dataset_path, dir)):
            dataset_name, file_ext = filename.split(".")
            if file_ext == "src":
                with open(os.path.join(dataset_path, dir, filename), "r") as f:
                    line_count = len(f.readlines())
                f.close()
                dataset_counts[f"{dir}/{dataset_name}"] = line_count

def create_hyp_metadata(existing_df=None):
    dict_info = {
        "task": [],
        "dataset": [],
        "hyp_sampling": [],
        "hyp_count": [],
        "supporting_files": [],
    }
    supporting_file_ext = ["logs", "lprobs", "lprobs_norm"]
    for dir in tqdm(os.listdir(hyps_path), desc="Processing Hypothesis Files"):
        for filename in tqdm(os.listdir(os.path.join(hyps_path, dir)), desc=f"Processing {dir}"):
            if filename.split(".")[-1] not in supporting_file_ext:
                info = filename.split(".")
                dict_info["task"].append(dir)
                dict_info["dataset"].append(info[0])
                dict_info["hyp_sampling"].append(info[2])
                dict_info["hyp_count"].append(info[3])
                temp_support_files = {}

                if existing_df is not None:
                    existing_row = existing_df.loc[
                        (existing_df["task"] == dir) &
                        (existing_df["dataset"] == info[0]) &
                        (existing_df["hyp_sampling"] == info[2]) &
                        (existing_df["hyp_count"] == int(info[3]))
                    ]
                    if not existing_row.empty and not existing_row.iloc[0]["supporting_files"] == "line count mismatch":
                        existing_support_files = existing_row.iloc[0]["supporting_files"]
                        dict_info["supporting_files"].append(existing_support_files)
                        continue

                with open (os.path.join(hyps_path, dir, filename), "r") as f:
                    line_count = len(f.readlines())
                f.close()
                for ext in supporting_file_ext:
                    supporting_file = f"{filename}.{ext}"
                    if ext == "logs":
                        temp_support_files[ext] = supporting_file
                        continue
                    if os.path.exists(os.path.join(hyps_path, dir, supporting_file)):
                        with open(os.path.join(hyps_path, dir, supporting_file), "r") as f:
                            supporting_file_line_count = len(f.readlines())
                        if supporting_file_line_count == line_count:
                            temp_support_files[ext] = supporting_file
                        else:
                            temp_support_files[ext] = "line count mismatch"
                        f.close()
                    else:
                        temp_support_files[ext] = None
                dict_info["supporting_files"].append(temp_support_files)
    return pd.DataFrame(dict_info)

def create_cache_metadata(existing_df=None):
    dict_info = {
        "task": [],
        "dataset": [],
        "util_function": [],
        "hyp_sampling": [],
        "hyp_count": [],
        "ref_sampling":[],
        "ref_count": [],
        "cache_file": [],
    }
    for dir in tqdm(os.listdir(cache_path), desc="Processing Cache Directories"):
        for filename in tqdm(os.listdir(os.path.join(cache_path, dir)), desc=f"Processing {dir}"):
            if filename.split(".")[-1] == "cache":
                info = filename.split(".")
                dict_info["task"].append(dir)
                dict_info["dataset"].append(info[0])
                dict_info["util_function"].append(info[2].split("-")[0])
                dict_info["hyp_sampling"].append(info[3])
                dict_info["hyp_count"].append(info[4].split("-")[0])
                dict_info["ref_sampling"].append(info[4].split("-")[-1])
                dict_info["ref_count"].append(info[5])

                if existing_df is not None:
                    existing_row = existing_df.loc[
                        (existing_df["task"] == dir) &
                        (existing_df["dataset"] == info[0]) &
                        (existing_df["util_function"] == info[2].split("-")[0]) &
                        (existing_df["hyp_sampling"] == info[3]) &
                        (existing_df["hyp_count"] == int(info[4].split("-")[0])) &
                        (existing_df["ref_sampling"] == info[4].split("-")[-1]) &
                        (existing_df["ref_count"] == int(info[5]))
                    ]
                    if not existing_row.empty and not existing_row.iloc[0]["cache_file"] == "line count mismatch":
                        dict_info["cache_file"].append(existing_row.iloc[0]["cache_file"])
                        continue

                with open(os.path.join(cache_path, dir, filename), "r") as f:
                    cache_line_count = len(f.readlines())
                f.close()
                if cache_line_count >= dataset_counts[f"{dir}/{info[0]}"]:
                    dict_info["cache_file"].append(filename)
                else:
                    dict_info["cache_file"].append("line count mismatch")
    return pd.DataFrame(dict_info)

def create_score_metadata(existing_df=None):
    dict_info = {
        "task": [],
        "dataset": [],
        "hyp_sampling": [],
        "hyp_count": [],
        "metric_name": [],
        "score_file": []
    }
    for dir in tqdm(os.listdir(score_path), desc="Processing Score Files"):
        for filename in tqdm(os.listdir(os.path.join(score_path, dir)), desc=f"Processing {dir}"):
            if filename.split(".")[-1] == "yaml":
                continue
            info = filename.split(".")
            dict_info["task"].append(dir)
            dict_info["dataset"].append(info[0])
            dict_info["hyp_sampling"].append(info[2])
            dict_info["hyp_count"].append(info[3])
            dict_info["metric_name"].append(info[4])

            if existing_df is not None:
                existing_row = existing_df.loc[
                    (existing_df["task"] == dir) &
                    (existing_df["dataset"] == info[0]) &
                    (existing_df["hyp_sampling"] == info[2]) &
                    (existing_df["hyp_count"] == int(info[3])) &
                    (existing_df["metric_name"] == info[4])
                ]
                if not existing_row.empty and not existing_row.iloc[0]["score_file"] == "line count mismatch":
                    dict_info["score_file"].append(existing_row.iloc[0]["score_file"])
                    continue

            with open(os.path.join(score_path, dir, filename), "r") as f:
                score_line_count = len(f.readlines())
            f.close()
            if score_line_count >= dataset_counts[f"{dir}/{info[0]}"] * int(info[3]):
                dict_info["score_file"].append(filename)
            else:
                dict_info["score_file"].append("line count mismatch")
    return pd.DataFrame(dict_info)

def create_validated_metadata(existing_df=None):
    dict_info = {
        "task": [],
        "dataset": [],
        "util_function":[],
        "decompose_mode":  [],
        "hyp_sampling": [],
        "hyp_count": [],
        "ref_sampling":[],
        "ref_count":[],
        "params":[],
        "selected_text_file": [],
        "supporting_files": []
    }
    for dir in tqdm(os.listdir(validated_path), desc="Processing Validated Files"):
        for filename in tqdm(os.listdir(os.path.join(validated_path, dir)), desc=f"Processing {dir}"):
            if filename.split(".")[-1].isnumeric() or filename.split(".")[-1] not in ["mbr_data", "pt", "logs", "yaml"]:
                info = filename.split(".")
                dict_info["task"].append(dir)
                dict_info["dataset"].append(info[0])
                dict_info["util_function"].append(info[2].split("-")[0])
                decompose_mode = info[2].split("-")[1]
                dict_info["decompose_mode"].append(decompose_mode)
                dict_info["hyp_sampling"].append(info[3])
                dict_info["hyp_count"].append(info[4].split("-")[0])
                dict_info["ref_sampling"].append(info[4].split("-")[1])
                dict_info["ref_count"].append(info[5])

                if "params-" in filename:
                    dict_info["params"].append(filename.split("params-")[-1])
                else:
                    dict_info["params"].append("")

                existing_row = pd.DataFrame()
                if existing_df is not None:
                    existing_row = existing_df.loc[
                        (existing_df["task"] == dir) &
                        (existing_df["dataset"] == info[0]) &
                        (existing_df["util_function"] == info[2].split("-")[0]) &
                        (existing_df["decompose_mode"] == decompose_mode) &
                        (existing_df["hyp_sampling"] == info[3]) &
                        (existing_df["hyp_count"] == int(info[4].split("-")[0])) &
                        (existing_df["ref_sampling"] == info[4].split("-")[1]) &
                        (existing_df["ref_count"] == int(info[5])) &
                        (existing_df["params"] == dict_info["params"][-1])
                    ]

                if existing_row.empty or not existing_row.iloc[0]["selected_text_file"] == "line count mismatch":
                    with open(os.path.join(validated_path, dir, filename), "r") as f:
                        validated_line_count = len(f.readlines())
                    f.close()
                    if validated_line_count >= dataset_counts[f"{dir}/{info[0]}"]:
                        dict_info["selected_text_file"].append(filename)
                    else:
                        dict_info["selected_text_file"].append("line count mismatch")
                else:
                    dict_info["selected_text_file"].append(existing_row.iloc[0]["selected_text_file"])
                
                if not existing_row.empty and not ("file not found" in existing_row.iloc[0]["supporting_files"].values()):
                    dict_info["supporting_files"].append(existing_row.iloc[0]["supporting_files"])
                    continue

                temp_supporting_files = {}
                if "svd_mbr" in decompose_mode:
                    for ext in ["decomposed_matrix.pt", "mbr_data", "singularvals.pt"]:
                        supporting_file = f"{filename}.{ext}"
                        if os.path.exists(os.path.join(validated_path, dir, supporting_file)):
                            temp_supporting_files[ext] = supporting_file
                        else:
                            temp_supporting_files[ext] = "file not found"
                elif decompose_mode == "nmf_mbr":
                    for ext in ["decomposed_matrix.pt", "mbr_data", "H.pt", "W.pt"]:
                        supporting_file = f"{filename}.{ext}"
                        if os.path.exists(os.path.join(validated_path, dir, supporting_file)):
                            temp_supporting_files[ext] = supporting_file
                        else:
                            temp_supporting_files[ext] = "file not found"
                elif decompose_mode in ["mbr", "normed_mbr"]:
                    for ext in ["original_matrix.pt"]:
                        supporting_file = f"{filename}.{ext}"
                        if os.path.exists(os.path.join(validated_path, dir, supporting_file)):
                            temp_supporting_files[ext] = supporting_file
                        else:
                            temp_supporting_files[ext] = "file not found"
                elif decompose_mode in ["model_mbr", "probabilistic_mbr", "model_norm_mbr"]:
                    temp_supporting_file = {}
                else:
                    raise ValueError(f"Unknown decompose mode {decompose_mode} in filename {filename}")
                dict_info["supporting_files"].append(temp_supporting_files)

    return pd.DataFrame(dict_info)

def main():
    get_dataset_count()
    existing_df = None

    # if os.path.exists("metadata/hyp_metadata.csv"):
    #     existing_df = pd.read_csv("metadata/hyp_metadata.csv")
    #     existing_df["supporting_files"] = existing_df["supporting_files"].apply(literal_eval)
    # hyp_metadata_df = create_hyp_metadata(existing_df)
    # hyp_metadata_df.to_csv("metadata/hyp_metadata.csv", index=False)

    # if os.path.exists("metadata/cache_metadata.csv"):
    #     existing_df = pd.read_csv("metadata/cache_metadata.csv")
    # cache_metadata_df = create_cache_metadata(existing_df)
    # cache_metadata_df.to_csv("metadata/cache_metadata.csv", index=False)

    # if os.path.exists("metadata/score_metadata.csv"):
    #     existing_df = pd.read_csv("metadata/score_metadata.csv")
    # score_metadata_df = create_score_metadata(existing_df)
    # score_metadata_df.to_csv("metadata/score_metadata.csv", index=False)

    if os.path.exists("metadata/validated_metadata.csv"):
        existing_df = pd.read_csv("metadata/validated_metadata.csv")
        existing_df["supporting_files"] = existing_df["supporting_files"].apply(literal_eval)
    validated_metadata_df = create_validated_metadata(existing_df)
    validated_metadata_df.to_csv("metadata/validated_metadata.csv", index=False)

    # if os.path.exists("metadata/ensemble_metadata.csv"):
    #     existing_df = pd.read_csv("metadata/ensemble_metadata.csv")
    #     existing_df["supporting_files"] = existing_df["supporting_files"].apply(literal_eval)
    # validated_metadata_df = create_validated_metadata(existing_df)
    # validated_metadata_df.to_csv("metadata/ensemble_metadata.csv", index=False)

if __name__ == "__main__":
    main()
