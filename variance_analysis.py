import sys
sys.path.append("../")
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from src.svd_mbr.modules import svd_decomposition, z_score_norm
from collections import Counter

def bootstrap_variance_test(pairwise_matrix: torch.Tensor, num_bootstraps=100):
    """
    Measures the stability of Naive MBR, Top-1, Top-2, and Top-k SVD-MBR by bootstrapping the references.
    Where k is num_hypotheses.
    
    Args:
        pairwise_matrix: torch.Tensor of shape [num_hypotheses, num_references]
        num_bootstraps: int, the number of resampling iterations
        
    Returns:
        A dictionary containing the stability statistics for both methods.
    """
    num_hyp, num_ref = pairwise_matrix.shape
    
    winners = {
        "naive": [],
        "svd_top1": [],
        "svd_top2": [],
        "svd_topk": []
    }
    
    for i in range(num_bootstraps):
        # 1. Sample reference indices with replacement
        # This simulates a slightly different LLM generation pool each time
        sampled_indices = torch.randint(0, num_ref, (num_ref,))
        boot_matrix = pairwise_matrix[:, sampled_indices]
        
        # --------------------------------------------------
        # 2. Run Naive MBR (Mean across references)
        # --------------------------------------------------
        naive_scores = boot_matrix.mean(dim=1)
        naive_winner_idx = torch.argmax(naive_scores).item()
        winners["naive"].append(naive_winner_idx)
        
        # --------------------------------------------------
        # 3. Run Top-1 SVD-MBR (norm_dim=0)
        # --------------------------------------------------
        # Normalize
        normed_boot_matrix = z_score_norm(boot_matrix, dim=None)
        
        # Decompose
        try:
            topk_boot_matrix, S, U, Vh  = svd_decomposition(normed_boot_matrix, top_k=min(num_ref, num_hyp), is_reduced=True)
        except Exception as e:
            print(f"SVD decomposition failed at bootstrap iteration {i} with error: {e}")
            fp64_matrix = normed_boot_matrix.to(torch.float64)
            jitter = torch.randn_like(fp64_matrix) * 1e-7
            safe_matrix = fp64_matrix + jitter
            topk_boot_matrix, S, U, Vh  = svd_decomposition(safe_matrix, top_k=min(num_ref, num_hyp), is_reduced=True)
        topk_scores = topk_boot_matrix.mean(dim=1)
        topk_winner_idx = torch.argmax(topk_scores).item()
        winners["svd_topk"].append(topk_winner_idx)
        
        for k in [1, 2]:
            U_temp = U.cpu()[:, :k]
            S_temp = S.cpu()[:k]
            Vh_temp = Vh.cpu()[:k, :]
            temp_boot_matrix = (U_temp * S_temp) @ Vh_temp
            temp_scores = temp_boot_matrix.mean(dim=1)
            temp_winner_idx = torch.argmax(temp_scores).item()
            winners[f"svd_top{k}"].append(temp_winner_idx)
        

    # 4. Calculate Stability Statistics
    def calculate_stats(winners_list):
        counts = Counter(winners_list)
        unique_count = len(counts)
        most_common_idx, most_common_freq = counts.most_common(1)[0]
        stability_pct = (most_common_freq / num_bootstraps) * 100
        return unique_count, stability_pct, counts

    bootstrap_results = {}
    for method in winners:
        unique, stability, counts = calculate_stats(winners[method])
        bootstrap_results[method] = {"unique": unique, "stability": stability, "most_common": counts.most_common(1)[0]}
        # print(f"  Unique Winners Picked: {unique}")
        # print(f"  Stability Score: {stability:.2f}% (Maintained same winner {counts.most_common(1)[0][1]}/{num_bootstraps} times)")
        # print("==================================")

    return bootstrap_results

metrics = {
    "translation": ["comet", "bleurt", "bleu", "chrf", "cometkiwi"],
    "summarization": ["bertscore", "rouge_1", "rouge_2", "rouge_l", "rouge_lsum"]
}
all_metrics = metrics["translation"] + metrics["summarization"]

result_dir = "results/20260414"
metadata_dir = "metadata"
validated_path = "/var/autofs/cl/home2/share/mbrs/validated"
no_mbr_df = pd.read_csv(f"{result_dir}/decompose_decode_status_no_mbr.csv")
no_mbr_df = no_mbr_df.dropna(subset=all_metrics)

oracle_df = pd.read_csv(f"{result_dir}/decompose_decode_status_oracle.csv")

vanilla_df = pd.read_csv(f"{result_dir}/decompose_decode_status_mbr.csv")

normed_svd_df = pd.read_csv(f"{result_dir}/decompose_decode_status_normed_svd_mbr.csv")
nmf_df = pd.read_csv(f"{result_dir}/decompose_decode_status_nmf_mbr.csv")
validated_metadata = pd.read_csv(f"{metadata_dir}/validated_metadata.csv")
validated_metadata.rename(columns={"decompose_mode": "method"}, inplace=True)

cache_metadata = pd.read_csv(f"{metadata_dir}/cache_metadata.csv")
score_metadata = pd.read_csv(f"{metadata_dir}/score_metadata.csv")

vanilla_df = pd.merge(vanilla_df, validated_metadata.loc[validated_metadata["method"] == "mbr"], 
    left_on=["task", "dataset", "hyp_sampling", "hyp_count","ref_sampling", "ref_count", "method", "util_function"], 
    right_on=["task", "dataset", "hyp_sampling", "hyp_count","ref_sampling", "ref_count", "method", "util_function"], 
    how="inner")

normed_svd_df = pd.merge(normed_svd_df, validated_metadata.loc[validated_metadata["method"] == "normed_svd_mbr"], 
    left_on=["task", "dataset", "hyp_sampling", "hyp_count","ref_sampling", "ref_count", "method", "util_function", "params"], 
    right_on=["task", "dataset", "hyp_sampling", "hyp_count","ref_sampling", "ref_count", "method", "util_function", "params"], 
    how="inner")

nmf_df = pd.merge(nmf_df, validated_metadata.loc[validated_metadata["method"] == "nmf_mbr"], 
    left_on=["task", "dataset", "hyp_sampling", "hyp_count","ref_sampling", "ref_count", "method", "util_function", "params"], 
    right_on=["task", "dataset", "hyp_sampling", "hyp_count","ref_sampling", "ref_count", "method", "util_function", "params"], 
    how="inner")

normed_svd_df["supporting_files"] = normed_svd_df["supporting_files"].apply(literal_eval)
vanilla_df["supporting_files"] = vanilla_df["supporting_files"].apply(literal_eval)
nmf_df["supporting_files"] = nmf_df["supporting_files"].apply(literal_eval)

temp_df = vanilla_df.copy()
temp_df = temp_df.loc[temp_df["task"] == "translation"]
temp_res = []
for idx, row in tqdm(temp_df.iterrows(), total=temp_df.shape[0], desc="Processing Rows"):
    # print(f"Processing row {idx} with dataset {row['dataset']} and util function {row['util_function']}")
    pairwise_matrix = torch.load(f'{validated_path}/{row["task"]}/{row["supporting_files"]["original_matrix.pt"]}')
    all_results = {}
    for data_sample in pairwise_matrix:
        results = bootstrap_variance_test(data_sample, num_bootstraps=100)
        for method in results:
            if method not in all_results:
                all_results[method] = {"unique": [], "stability": [], "most_common": []}
            all_results[method]["unique"].append(results[method]["unique"])
            all_results[method]["stability"].append(results[method]["stability"])
            all_results[method]["most_common"].append(results[method]["most_common"])
    # print(all_results)
    all_results_avg = {}
    for method in all_results:
        all_results_avg[method + "_unique"] = np.mean(all_results[method]["unique"])
        all_results_avg[method + "_stability"] = np.mean(all_results[method]["stability"])
        all_results_avg[method + "_most_common"] = all_results[method]["most_common"]
    temp_res.append(all_results_avg)
temp_res = pd.DataFrame(temp_res, index=temp_df.index)
temp_df[temp_res.columns] = temp_res
temp_df.to_csv(f"{result_dir}/bootstrap_variance_analysis.csv", index=False)