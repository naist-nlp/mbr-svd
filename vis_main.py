import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from argparse import ArgumentParser


# from vis_load_data import compile_decodes, compile_scores, update_decode_status, get_combination_result, get_scores
from vis_load_data import (compile_no_mbr_scores, 
                        compile_mbr_scores, 
                        compile_oracle_scores)

from vis_plots import main as visualize_plots

metadata_dir = "metadata"

def parse_args():
    parser = ArgumentParser(description="Visualization for SVD analysis results")
    parser.add_argument(
        "--decompose_mode",
        type=str,
        help="Decomposition mode to analyze (e.g., svd_mbr)",
    )
    parser.add_argument(
        "--process_type",
        type=str,
        help="Type of process to run",
    )
    parser.add_argument(
        "--cand_counts",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128, 256, 512, 1024],
        help="List of candidate counts to consider",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=metadata_dir,
        help="Path to the metadata CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the generated visualizations",
    )
    return parser.parse_args()

# def load_score_data(score_filename):
#     try:
#         scores_df = pd.read_csv(score_filename)
#         scores_df = scores_df.map(lambda x: literal_eval(x) if x.startswith("[") else x)
#     except:
#         scores_df = compile_scores()
#         scores_df.to_csv(score_filename, index=False)
#     return scores_df

# def update_data(decode_df, scores_series):

#     # Check for decoding results
#     print("Decode df shape before updating:", decode_df.shape)
#     condition = decode_df["decode_exist"] == True
#     result_df = decode_df.loc[condition].apply(get_combination_result, all_hyps=np.array(scores_series["hyps"].values[0]), axis=1, result_type="expand")
#     decode_df.loc[condition, result_df.columns] = result_df

#     print("Combination results updated.")

#     # Calculate the scores
#     condition = decode_df["selected_indexes"].apply(lambda x: len(x) > 0)
#     result_list = decode_df.loc[condition, "selected_indexes"].apply(get_scores, scores_series=scores_series)
#     result_df = pd.DataFrame(result_list.tolist(), index=decode_df.loc[condition].index)
#     decode_df.loc[condition, result_df.columns] = result_df

#     print("Scores updated.")

#     return decode_df

def main():
    args = parse_args()
    decompose_mode = args.decompose_mode
    process_type = args.process_type
    cand_counts = args.cand_counts
    metadata_dir = args.metadata
    output_dir = args.output_dir

    decompose_filename = f"{output_dir}/decompose_decode_status_{decompose_mode}.csv"

    if process_type == "prepare_data":
        hyp_metadata = pd.read_csv(f"{metadata_dir}/hyp_metadata.csv")
        score_metadata = pd.read_csv(f"{metadata_dir}/score_metadata.csv")
        validated_metadata = pd.read_csv(f"{metadata_dir}/validated_metadata.csv")

        hyp_metadata = hyp_metadata.loc[hyp_metadata["dataset"] == "wmt22-ende"]

        validated_metadata["supporting_files"] = validated_metadata["supporting_files"].apply(literal_eval)
        hyp_metadata["supporting_files"] = hyp_metadata["supporting_files"].apply(literal_eval)

        if decompose_mode == "no_mbr":
            results_df = compile_no_mbr_scores(hyp_metadata, score_metadata)
        elif decompose_mode == "oracle":
            results_df = compile_oracle_scores(hyp_metadata, score_metadata)
        else:
            results_df = compile_mbr_scores(hyp_metadata, score_metadata, validated_metadata, decompose_mode)
        results_df.to_csv(decompose_filename, index=False)
    elif process_type == "visualize":
        no_mbr_df = pd.read_csv(f"{output_dir}/decompose_decode_status_no_mbr.csv")
        
        oracle_df = pd.read_csv(f"{output_dir}/decompose_decode_status_oracle.csv")

        vanilla_df = pd.read_csv(f"{output_dir}/decompose_decode_status_mbr.csv")

        decomposed_df = pd.read_csv(f"{output_dir}/decompose_decode_status_{decompose_mode}.csv")
        validated_metadata = pd.read_csv(f"{metadata_dir}/validated_metadata.csv")
        validated_metadata.rename(columns={"decompose_mode": "method"}, inplace=True)

        vanilla_df = pd.merge(vanilla_df, validated_metadata.loc[validated_metadata["method"] == "mbr"], 
            left_on=["task", "dataset", "hyp_sampling", "hyp_count","ref_sampling", "ref_count", "method", "util_function"], 
            right_on=["task", "dataset", "hyp_sampling", "hyp_count","ref_sampling", "ref_count", "method", "util_function"], 
            how="inner")

        decomposed_df = pd.merge(decomposed_df, validated_metadata.loc[validated_metadata["method"] == decompose_mode], 
            left_on=["task", "dataset", "hyp_sampling", "hyp_count","ref_sampling", "ref_count", "method", "util_function", "params"], 
            right_on=["task", "dataset", "hyp_sampling", "hyp_count","ref_sampling", "ref_count", "method", "util_function", "params"], 
            how="inner")

        decomposed_df["supporting_files"] = decomposed_df["supporting_files"].apply(literal_eval)
        vanilla_df["supporting_files"] = vanilla_df["supporting_files"].apply(literal_eval)

        visualize_plots(decomposed_df, vanilla_df, no_mbr_df, oracle_df, output_dir=output_dir)

    # try:
    #     decode_df = pd.read_csv(decompose_filename)
    #     decode_df["param"] = decode_df["param"].map(lambda x: literal_eval(x) if isinstance(x, str) else x)
    #     decode_df["selected_indexes"] = decode_df["selected_indexes"].map(lambda x: literal_eval(x) if isinstance(x, str) else x)
    # except FileNotFoundError:
    #     decode_df = compile_decodes()
    #     decode_df["decode_exist"] = decode_df.apply(update_decode_status, axis=1)
    #     score_df = load_score_data(score_filename)
    #     decode_df = update_data(decode_df, score_df)
    # else:
    #     print("Entering else")
    #     sub_decode_df = decode_df.loc[decode_df["decompose_mode"] == "normed_svd_mbr"].copy()
    #     score_df = load_score_data(score_filename)
    #     sub_decode_df = update_data(sub_decode_df, score_df)
    #     decode_df.update(sub_decode_df)
    #     # condition = decode_df["decode_exist"] == False
    #     # if condition.sum() > 0:
    #     #     decode_df.loc[condition, "decode_exist"] = decode_df.loc[condition].apply(update_decode_status, axis=1)
    #     #     scores_series = load_score_data(score_filename, hyp_count)
    #     #     decode_df = update_data(decode_df, scores_series)
    # finally:
    #     decode_df.to_csv(decompose_filename, index=False)

    # print("Decode data loaded.")

    
    # data_df = decode_df.loc[(decode_df["decompose_mode"].isin([decompose_mode])) & 
    #                         (decode_df["hyp_type"].isin([f"eps.{count}" for count in cand_counts])) & 
    #                         (decode_df["ref_type"].isin([f"eps.{count}" for count in cand_counts]))].copy()
    # data_df["param"] = data_df["param"].apply(literal_eval)
    # data_df = data_df.loc[data_df["param"].apply(lambda x: x.get("is_reduced", None) == True)]
    # result_df = data_df.apply(get_data_by_row, axis=1, result_type="expand")
    # data_df[result_df.columns] = result_df

    # if visualization_type == "matrices":
    #     filtered_data = data_df.loc[data_df["param"].apply(lambda x: x.get("top_k_sv", None) != None)]
    #     for idx, row in tqdm(filtered_data.iterrows(), total=filtered_data.shape[0], desc="Matrices Visualization"):
    #         if row["param"]:
    #             file_name = f'{row["hyp_type"]}-{row["ref_type"]}.params-{"-".join([f"{k}-{v}" for k, v in row["param"].items()])}'
    #         else:
    #             file_name = f'{row["hyp_type"]}-{row["ref_type"]}'
    #         output_path = f"{analysis_path}/{decompose_mode}/matrices/{file_name}"
    #         os.makedirs(output_path, exist_ok=True)
    #         vis_matrices(row["decode_content"], output_path)
    # elif visualization_type == "singularvals":
    #     filtered_data = data_df.loc[data_df["param"].apply(lambda x: x.get("top_k_sv", None) == 0)]
    #     for idx, row in tqdm(filtered_data.iterrows(), total=filtered_data.shape[0], desc="Singular Values Visualization"):
    #         if row["param"]:
    #             file_name = f'{row["hyp_type"]}-{row["ref_type"]}.params-{"-".join([f"{k}-{v}" for k, v in row["param"].items()])}'
    #         else:
    #             file_name = f'{row["hyp_type"]}-{row["ref_type"]}'
    #         output_path = f"{analysis_path}/{decompose_mode}/singularvals/{file_name}"
    #         os.makedirs(output_path, exist_ok=True)
    #         vis_singularvals(row["decode_content"], min(int(row["hyp_type"].split(".")[-1]), int(row["ref_type"].split(".")[-1])), output_path)
    # elif visualization_type == "performance":
    #     for hyp_cnt in cand_counts:
    #         filtered_scores = scores_df.loc[scores_df["hyp_type"] == f"eps.{hyp_cnt}"].iloc[0]
    #         comparer_indexes = scores_df.loc[scores_df["hyp_type"] == f"eps.{hyp_cnt}", ["lprobs_idx", "bleu_oracle_idx", "chrf_oracle_idx", "comet_oracle_idx", "bleurt_oracle_idx", "cometkiwi_oracle_idx"]].rename(columns={
    #             "lprobs_idx": "lprobs",
    #             "bleu_oracle_idx": "bleu_oracle",
    #             "chrf_oracle_idx": "chrf_oracle",
    #             "comet_oracle_idx": "comet_oracle",
    #             "bleurt_oracle_idx": "bleurt_oracle",
    #             "cometkiwi_oracle_idx": "cometkiwi_oracle"
    #         }).to_dict(orient="records")[0]
    #         for ref_cnt in cand_counts:
    #             if decompose_mode == "normed_svd_mbr":
    #                 for norm_dim in [None, 0, 1]:
    #                     filtered_data = data_df.loc[(data_df["hyp_type"] == f"eps.{hyp_cnt}") & (data_df["ref_type"] == f"eps.{ref_cnt}") & (data_df["param"].apply(lambda x: x.get("norm_dim", None) == norm_dim))]
    #                     output_path = f"{analysis_path}/{decompose_mode}/performance/eps.{hyp_cnt}-eps.{ref_cnt}-norm_dim_{norm_dim}"
    #                     os.makedirs(output_path, exist_ok=True)
    #                     vis_performance(filtered_data[["param", "decode_content"]], filtered_scores, comparer_indexes, output_path)
    #             else:                    
    #                 filtered_data = data_df.loc[(data_df["hyp_type"] == f"eps.{hyp_cnt}") & (data_df["ref_type"] == f"eps.{ref_cnt}")]
    #                 output_path = f"{analysis_path}/{decompose_mode}/performance/eps.{hyp_cnt}-eps.{ref_cnt}"
    #                 os.makedirs(output_path, exist_ok=True)
    #                 vis_performance(filtered_data[["param", "decode_content"]], filtered_scores, comparer_indexes, output_path)



if __name__ == "__main__":
    main()