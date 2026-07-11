import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from scipy.stats import kendalltau
from itertools import combinations

from vis_load_data import (
    validated_path,
    score_path,
    hyps_path,
    translation_metrics, 
    summarization_metrics,
)

def _calculate_matrix_diff(ori_mat, dec_mat):
    """
    Expects 2D tensor of shape [num_hypotheses, num_references].
    """
    return ori_mat - dec_mat

def _calculate_l2_norm(diff_tensor):
    """
    Expects 2D tensor of shape [num_hypotheses, num_references].
    """
    return torch.linalg.norm(diff_tensor, ord='fro')

def _normalize_zscore(tensor, dim=None, epsilon=1e-8):
    """
    Expects 2D tensor of shape [num_hypotheses, num_references]. If dim is specified, it normalizes along that dimension; otherwise, it normalizes the entire tensor.
    """
    if dim is None:
        mean = torch.mean(tensor)
        std = torch.std(tensor)
    else:
        mean = torch.mean(tensor, dim=dim, keepdim=True)
        std = torch.std(tensor, dim=dim, keepdim=True)
    normalized_tensor = (tensor - mean) / (std + epsilon)
    return normalized_tensor

def process_matrices(original_matrix, reconstructed_matrix):
    """
    Expects two 3D tensors of shape [dataset_size, num_hypotheses, num_references].
    """
    diff = torch.vmap(_calculate_matrix_diff)(original_matrix, reconstructed_matrix)
    error_val = torch.vmap(_calculate_l2_norm)(diff)
    reconstructed_matrix = torch.mean(reconstructed_matrix, dim=0)
    error_val = error_val.mean().item()  # Average L2 norm across the dataset
    return error_val, reconstructed_matrix

def calculate_kendall_tau(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    res = kendalltau(x, y)
    tau, p_value = res.statistic, res.pvalue
    return tau, p_value

def get_expected_ranks(scores: torch.Tensor) -> torch.Tensor:
    """
    Given a 2D tensor of shape [dataset_size, num_cands], compute the expected ranks for each candidate across the dataset.
    """
    ranks = torch.argsort(scores, dim=1, descending=True) + 1  # Get ranks (1-based)
    return ranks

def save_figure(fig, output_file, output_ext="png"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.savefig(f"{output_file}.{output_ext}", format=output_ext, bbox_inches='tight')
    plt.close(fig)

def generate_top_idx_agreement_fig(top_idx_dict, output_dir=""):
    """
    Generates a Pairwise Agreement Heatmap and an Agreement Distribution plot.
    
    Args:
        top_idx_dict: Dictionary where keys are method names and values are lists of chosen indices.
        Example: {"original": [1, 5, 2], "rank1": [1, 5, 3]}
    """
    # Convert input dictionary to a DataFrame
    df = pd.DataFrame(top_idx_dict)
    methods = df.columns
    n_methods = len(methods)
    
    # ==========================================
    # Plot 1: Pairwise Agreement Heatmap
    # ==========================================
    # Initialize an identity matrix (100% agreement with themselves)
    agreement_matrix = pd.DataFrame(np.eye(n_methods), index=methods, columns=methods)
    agreement_matrix = agreement_matrix.sort_index().sort_index(axis=1)  # Ensure the order of methods is consistent in both axes
    
    # Calculate pairwise agreement
    for m1, m2 in combinations(methods, 2):
        # Calculate the proportion of times the two methods picked the same index
        agreement_rate = (df[m1] == df[m2]).mean()
        agreement_matrix.loc[m1, m2] = agreement_rate
        agreement_matrix.loc[m2, m1] = agreement_rate # It's a symmetric matrix
        
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(
        agreement_matrix * 100, # Convert to percentages
        annot=True if n_methods <= 15 else False,             # Show numbers in cells
        fmt=".1f",              # 1 decimal place
        cmap="Blues",           # Blue color scale
        cbar_kws={'label': 'Agreement Percentage (%)'},
        vmin=0, vmax=100        # Lock scale from 0 to 100
    )
    plt.title("Pairwise Agreement: % of Identical Hypothesis Choices")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_figure(fig, f"{output_dir}-top_idx_agreement")

    # ==========================================
    # Plot 2: Consensus Level (Unique Choices per Doc)
    # ==========================================
    # For each document, how many UNIQUE hypotheses were chosen across all methods?
    # If 1: Perfect consensus (all methods picked the same sentence)
    # If len(methods): Total disagreement (every method picked a different sentence)
    unique_choices_per_doc = df.nunique(axis=1)
    
    # Calculate percentages for the bar chart
    distribution = unique_choices_per_doc.value_counts(normalize=True).sort_index() * 100
    
    fig = plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=distribution.index, y=distribution.values, palette="viridis")
    
    plt.title("System Consensus: How fractured are the decisions?")
    plt.xlabel("Number of Unique Hypotheses Chosen per Document\n(1 = Perfect Agreement among all methods)")
    plt.ylabel("Percentage of Dataset (%)")
    
    # Add percentage labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 5), 
                    textcoords = 'offset points')
        
    plt.tight_layout()
    save_figure(fig, f"{output_dir}-consensus_level")


def generate_ranking_agreement_fig(expected_ranks_dict, p_value_threshold=0.05, output_dir=""):
    """
    Generate a heatmap comparing the Kendall Tau correlation of different ranking methods.
    Input:
        - expected_ranks_dict: key is method names and value is a 2d array with shape of [dataset_size, expected_ranks] of the reranking.
        - p_value_threshold: the threshold for significance of the Kendall Tau correlation.
    """
    methods = list(expected_ranks_dict.keys())
    n_methods = len(methods)

    tau_matrix = pd.DataFrame(np.eye(n_methods), index=methods, columns=methods)
    tau_matrix = tau_matrix.sort_index().sort_index(axis=1)  # Ensure the order of methods is consistent in both axes
    
    for combination in combinations(methods, 2):
        method1, method2 = combination
        taus = []
        for idx in range(expected_ranks_dict[method1].shape[0]):
            tau, p_value = calculate_kendall_tau(expected_ranks_dict[method1][idx], expected_ranks_dict[method2][idx])
            if p_value < p_value_threshold:
                taus.append(tau)
        avg_tau = np.mean(taus) if taus else 0.0
        tau_matrix.loc[method1, method2] = avg_tau
        tau_matrix.loc[method2, method1] = avg_tau

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(
        tau_matrix, 
        annot=True if n_methods <= 15 else False,
        fmt=".2f",             
        cmap="coolwarm",           
        cbar_kws={'label': 'Kendall Tau'},
        vmin=-1, vmax=1        # Lock scale from -1 to 1
    )
    plt.title("Kendall Tau Correlation of Ranking Methods")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    save_figure(fig, f"{output_dir}-kendall_tau_comparison")

def generate_performance_fig(metrics, decomposed_data, vanilla_scores, no_mbr_scores, oracle_scores, perf_hue_col, output_dir=""):
    """
    Generate a single fig image that contains the performance visualization of all metrics
    based on the top-k, but add horizontal lines
    for baselines.
    """
    nlim = 3
    nrows = len(metrics) // nlim + (1 if len(metrics) % nlim > 0 else 0)
    ncols = min(len(metrics), nlim)
    fig, axes = plt.subplots(figsize=(24, 10), nrows=nrows, ncols=ncols)

    # Prepare categorical x-axis
    numeric_sort = sorted(pd.to_numeric(decomposed_data['top_k'].unique()))
    str_categories = [str(x) for x in numeric_sort]
    decomposed_data['top_k'] = pd.Categorical(
        decomposed_data['top_k'].astype(str), 
        categories=str_categories, 
        ordered=True
    )

    shared_lines, shared_labels = [], []
    for metric_name, ax in zip(metrics, axes.flatten()):
        # --- PRIMARY AXIS (Blue) ---
        blue_color = 'tab:blue'
        if perf_hue_col:
            sns.lineplot(x=decomposed_data["top_k"], y=decomposed_data[metric_name], 
                marker="o", color=blue_color, hue=decomposed_data[perf_hue_col], ax=ax)
        else:
            sns.lineplot(x=decomposed_data["top_k"], y=decomposed_data[metric_name], 
                marker="o", color=blue_color, ax=ax)
        
        # Style the left axis
        ax.set_ylabel(metric_name, color=blue_color, fontweight='bold')
        ax.tick_params(axis='y', labelcolor=blue_color)
        ax.spines['left'].set_color(blue_color)

        # Add horizontal lines and right-side text
        def add_h_line(y_val, color, label_text):
            line = ax.axhline(y=y_val, color=color, linestyle="-.", label=label_text)
            ax.text(1.01, y_val, f"{y_val:.2f}", 
                    color=color, va='center', transform=ax.get_yaxis_transform())
            return line

        if vanilla_scores.shape[0] == 1 :
            add_h_line(vanilla_scores[metric_name].values[0] if metric_name in vanilla_scores else 0.0, "orange", "Vanilla MBR")
        if no_mbr_scores.shape[0] == 1:
            add_h_line(no_mbr_scores[metric_name].values[0] if metric_name in no_mbr_scores else 0.0, "green", "No MBR")
        add_h_line(oracle_scores.loc[oracle_scores["method"] == f"oracle_{metric_name}", metric_name].values[0] if not oracle_scores.empty else 0.0, "grey", "Oracle")

        # --- COMBINED LEGEND OUTSIDE ---
        lines, labels = ax.get_legend_handles_labels()
        ax.legend().remove()  # Remove the default legend from this subplot since we'll create a shared one outside
        
        # bbox_to_anchor=(1.15, 1) moves the legend to the right and top
        if metric_name == metrics[0]:  # Only add legend entries from the first subplot to avoid duplicates
            shared_lines.extend(lines)
            shared_labels.extend(labels)
        ax.set_xlabel("Top-k")
        ax.set_title(metric_name)

    # Remove empty subplots if metrics < nrows*ncols
    for i in range(len(metrics), nrows * ncols):
        fig.delaxes(axes.flatten()[i])

    axes[0][0].legend(shared_lines, shared_labels, 
        loc='upper left', bbox_to_anchor=(-0.1, 1.0))

    fig.suptitle(f"Performance Comparison Across Metrics")
    
    # Adjust layout: We need extra room on the right for both labels AND legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
        
    save_figure(fig, f"{output_dir}-metrics_perf_compare")

def generate_heatmap_matrices(tensor_list, title_list, output_dir="", main_title="MBR Matrices Drift by Top-K"):
    """
    Generates heatmaps from a list of tensors, wrapping after 3 plots per row.
    Dynamically hides annotations and ticks if matrix dimensions exceed 16x16.
    Shares a horizontal color scale placed underneath the main figure title.
    """
    num_tensors = len(tensor_list)
    
    if num_tensors != len(title_list):
        raise ValueError("The number of tensors must match the number of titles provided.")

    # 1. Calculate the global min and max across ALL tensors
    all_values = torch.cat([t.flatten() for t in tensor_list])
    global_vmin = all_values.min().item()
    global_vmax = all_values.max().item()

    # 2. Calculate grid dimensions
    ncols = min(3, num_tensors)
    nrows = int(np.ceil(num_tensors / 3))
    
    # Create the figure
    fig = plt.figure(figsize=(6 * ncols, 5 * nrows + 1.5))
    
    # Add a main title at the very top
    fig.suptitle(main_title, fontsize=18, y=0.98, fontweight='bold')

    # 3. Use GridSpec to design the layout
    height_ratios = [0.05] + [1] * nrows 
    gs = fig.add_gridspec(nrows=nrows + 1, ncols=ncols, height_ratios=height_ratios, hspace=0.4, wspace=0.3)
    
    # Create the dedicated axis for the shared horizontal colorbar
    cbar_ax = fig.add_subplot(gs[0, :])

    # 4. Loop through the lists and plot each heatmap
    for i, (tensor, title) in enumerate(zip(tensor_list, title_list)):
        
        r = (i // 3) + 1 
        c = i % 3
        ax = fig.add_subplot(gs[r, c])
        is_first = (i == 0)
        
        # --- NEW LOGIC: Dimension Check ---
        num_rows, num_cols = tensor.shape
        hide_details = max(num_rows, num_cols) > 16
        
        # Determine settings based on the size check
        show_annot = not hide_details
        x_labels = False if hide_details else 'auto'
        y_labels = False if hide_details else 'auto'
        
        # Plot the heatmap with dynamic labels/annotations
        sns.heatmap(
            tensor.cpu().numpy(), 
            annot=show_annot, 
            fmt=".2f", 
            cmap="viridis", 
            ax=ax, 
            vmin=global_vmin, 
            vmax=global_vmax, 
            cbar=is_first, 
            cbar_ax=cbar_ax if is_first else None,
            cbar_kws={"orientation": "horizontal"} if is_first else None,
            xticklabels=x_labels,
            yticklabels=y_labels
        )
        
        # Set titles and labels
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Pseudo-Ref")
        ax.set_ylabel("Hypothesis")
        
        # If we hid the text labels, also turn off the tiny tick line marks for a clean look
        if hide_details:
            ax.tick_params(left=False, bottom=False)
            
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_figure(fig, f"{output_dir}-heatmap_matrices")

def generate_param_effect_fig(df, param_name, metric_name="comet", output_dir=""):
    """
    Visualizes the effect of a specific parameter (e.g., 'beta' or 'l1_ratio')
    isolated strictly to rank 1 and rank 2 using a bar plot.
    """
    
    # 1. Helper function to safely parse the param string
    def parse_param(text):
        res_text = text.split(f"{param_name}-")[-1].split("-")
        if res_text[0] == "":
            return f'-{text.split(f"{param_name}-")[-1].split("-")[1]}' #Handle negative values (e.g., beta--1.0-l1_ratio-0.5)
        else:
            return res_text[0]
        
    # Work on a copy of the dataframe
    df_plot = df.copy()
    df_plot[param_name] = df_plot['params'].apply(parse_param)

    # Safely look for either 'rank' (NMF) or 'top_k_sv' (SVD) keys
    df_plot['rank_val'] = df_plot['top_k']
    
    # Drop rows missing the parameter or the rank entirely
    df_plot = df_plot.dropna(subset=[param_name, 'rank_val'])
    
    if df_plot.empty:
        print(f"No valid data found containing '{param_name}'.")
        return

    # Convert to numeric for accurate filtering and categorical sorting on the X-axis
    # df_plot[param_name] = pd.to_numeric(df_plot[param_name])
    df_plot['rank_val'] = pd.to_numeric(df_plot['rank_val'])
    
    # 3. FILTER: Keep ONLY rows where rank is exactly 1 or 2
    df_plot = df_plot[df_plot['rank_val'].isin([1, 2])]
    
    if df_plot.empty:
        print(f"No data found for '{param_name}' at rank 1 or 2.")
        return
        
    # Convert rank_val to a string so Seaborn treats it as a distinct category for colors
    df_plot['rank_val'] = "Top-k " + df_plot['rank_val'].astype(str)

    df_plot = df_plot.sort_values(by=[param_name, 'rank_val'])

    # 4. Set up the figure (1 row, 2 columns)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    fig.suptitle(f"Effect of '{param_name}' (Isolated to Top-k 1 & 2)", fontsize=16, fontweight='bold')

    # Plot 1: {metric_name} vs param_name (Left) - Changed to barplot
    sns.barplot(
        data=df_plot, x="rank_val", y=metric_name, hue=param_name, 
        palette='Set1', ax=axes[0]
    )
    axes[0].set_title(f'{metric_name.upper()} vs {param_name}')
    axes[0].set_xlabel("Top-k")
    axes[0].set_ylabel(f'{metric_name.upper()}')
    axes[0].grid(True, linestyle='--', alpha=0.6, axis='y') # Grid only on Y axis for bar plots
    axes[0].legend(title=param_name, loc='upper left')

    # Plot 2: avg_l2_norm vs param_name (Right) - Changed to barplot
    sns.barplot(
        data=df_plot, x="rank_val", y='avg_l2_norm', hue=param_name, 
        palette='Set1', ax=axes[1]
    )
    axes[1].set_title(f'Avg L2 Norm vs {param_name}')
    axes[1].set_xlabel("Top-k")
    axes[1].set_ylabel('Avg L2 Norm')
    axes[1].grid(True, linestyle='--', alpha=0.6, axis='y') # Grid only on Y axis for bar plots
    axes[1].legend(title=param_name, loc='upper left')
    # Adjust layout and display
    plt.tight_layout()
    
    save_figure(fig, f"{output_dir}-{param_name}_effect_plot")

def generate_eval_fig(metric_name, decomposed_data, output_dir=""):
    """
    Generate a single fig image that contains the performance visualization with the avg_l2_norm line plot 
    on the secondary y-axis. The legend is placed outside the plot area to the right.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare categorical x-axis
    numeric_sort = sorted(pd.to_numeric(decomposed_data['top_k'].unique()))
    str_categories = [str(x) for x in numeric_sort]
    decomposed_data['top_k'] = pd.Categorical(
        decomposed_data['top_k'].astype(str), 
        categories=str_categories, 
        ordered=True
    )

    # --- PRIMARY AXIS (Blue) ---
    score_min, score_max = decomposed_data[metric_name].min(), decomposed_data[metric_name].max()
    blue_color = 'tab:blue'
    sns.lineplot(x=decomposed_data["top_k"], y=decomposed_data[metric_name], 
                marker="o", color=blue_color, label=f"{metric_name} score", ax=ax)
    
    # Style the left axis
    ax.set_ylabel(metric_name, color=blue_color, fontweight='bold')
    ax.tick_params(axis='y', labelcolor=blue_color)
    ax.spines['left'].set_color(blue_color)

    ax.set_ylim(bottom=score_min * 0.99, top=score_max * 1.01)  # Add some padding to y-limits

    # --- SECONDARY AXIS (Red) ---
    ax2 = ax.twinx()
    red_color = 'tab:red'
    sns.lineplot(x=decomposed_data["top_k"], y=decomposed_data["avg_l2_norm"], 
                marker="x", linestyle="--", mec='red', color=red_color, label="Singular Value", ax=ax2)
    
    # Style the right axis
    ax2.set_ylabel("Avg L2 Norm of Diff Matrix", color=red_color, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=red_color)
    ax2.spines['right'].set_color(red_color)

    # --- COMBINED LEGEND OUTSIDE ---
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # bbox_to_anchor=(1.15, 1) moves the legend to the right and top
    ax.legend(lines + lines2, labels + labels2, 
            loc='upper left', bbox_to_anchor=(1.05, 1.0))
    
    # Remove the default legend from ax2
    if ax2.get_legend():
        ax2.get_legend().remove()

    plt.title(f"{metric_name} vs Top-k L2 Norm")
    ax.set_xlabel("Top-k")
    
    # Adjust layout: We need extra room on the right for both labels AND legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75) 
    
    save_figure(fig, f"{output_dir}-{metric_name}_eval_plot")

def generate_metric_eval_fig(metric_name, df, output_dir=""):
    """
    Generate a single fig image that contains the performance visualization with the avg_l2_norm line plot 
    on the secondary y-axis. The legend is placed outside the plot area to the right.
    """

    checking_cols = ["hyp_count", "ref_count"]
    unique_combination = df[checking_cols].drop_duplicates()
    unique_combination = unique_combination.sort_values(by=checking_cols)

    nlim = 5
    nrows, ncols = unique_combination.shape[0] // nlim + (1 if unique_combination.shape[0] % nlim > 0 else 0), min(unique_combination.shape[0], nlim)
    fig, axes = plt.subplots(figsize=(ncols*8, nrows*4),nrows=nrows, ncols=ncols)

    for idx, comb_row in unique_combination.reset_index(drop=True).iterrows():
        match_mask = np.all(df[checking_cols] == comb_row, axis=1)
        decomposed_data = df.loc[match_mask].copy()
        decomposed_data = decomposed_data.sort_values(by="top_k")

        # Prepare categorical x-axis
        numeric_sort = sorted(pd.to_numeric(decomposed_data['top_k'].unique()))
        str_categories = [str(x) for x in numeric_sort]
        decomposed_data['top_k'] = pd.Categorical(
            decomposed_data['top_k'].astype(str), 
            categories=str_categories, 
            ordered=True
        )
        
        ax = axes.flatten()[idx]

        # --- PRIMARY AXIS (Blue) ---
        score_min, score_max = decomposed_data[metric_name].min(), decomposed_data[metric_name].max()
        blue_color = 'tab:blue'
        sns.lineplot(x=decomposed_data["top_k"], y=decomposed_data[metric_name], 
                    marker="o", color=blue_color, label=f"{metric_name} score", ax=ax)
        
        # Style the left axis
        ax.set_ylabel(metric_name, color=blue_color, fontweight='bold')
        ax.tick_params(axis='y', labelcolor=blue_color)
        ax.spines['left'].set_color(blue_color)

        ax.set_ylim(bottom=score_min * 0.99, top=score_max * 1.01)  # Add some padding to y-limits

        # --- SECONDARY AXIS (Red) ---
        ax2 = ax.twinx()
        red_color = 'tab:red'
        sns.lineplot(x=decomposed_data["top_k"], y=decomposed_data["avg_l2_norm"], 
                    marker="x", linestyle="--", mec='red', color=red_color, label="Singular Value", ax=ax2)
        
        # Style the right axis
        ax2.set_ylabel("Avg L2 Norm of Diff Matrix", color=red_color, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=red_color)
        ax2.spines['right'].set_color(red_color)

        # --- COMBINED LEGEND OUTSIDE ---
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        # bbox_to_anchor=(1.15, 1) moves the legend to the right and top
        ax.legend(lines + lines2, labels + labels2, 
                loc='upper left', bbox_to_anchor=(1.05, 1.0))
        
        # Remove the default legend from ax2
        if ax2.get_legend():
            ax2.get_legend().remove()

        plt.title(f"{comb_row['hyp_count']} Hypotheses & {comb_row['ref_count']} References")
        ax.set_xlabel("Top-k")
        
        # Adjust layout: We need extra room on the right for both labels AND legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.75) 
    
    plt.suptitle(f"{metric_name} vs Top-k L2 Norm Across Different Hyp/Ref Counts", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
        
    save_figure(fig, f"{output_dir}-{metric_name}_all_eval_plot")

def main(decomposed_df, vanilla_df, no_mbr_df, oracle_df, output_dir=""):
    if "nmf" in decomposed_df["method"].unique()[0]:
        decomposed_df["top_k"] = decomposed_df["params"].apply(lambda x: int(x.split("rank-")[-1].split("-")[0]))
    else:
        decomposed_df["top_k"] = decomposed_df["params"].apply(lambda x: int(x.split("top_k_sv-")[-1].split("-")[0]))

    vanilla_df["ranks"] = vanilla_df[["task","supporting_files"]].apply(lambda x: get_expected_ranks(torch.load(f"{validated_path}/{x['task']}/{x['supporting_files']['original_matrix.pt']}").mean(dim=2)), axis=1)
    no_mbr_df["ranks"] = no_mbr_df[["task", "hyp_count", "supporting_files"]].apply(lambda x: get_expected_ranks(torch.tensor(np.array([float(line.strip()) for line in open(f"{hyps_path}/{x['task']}/{x['supporting_files']['lprobs_norm']}").readlines()]).reshape(-1, int(x['hyp_count'])))), axis=1)
    oracle_df["ranks"] = oracle_df[["task","score_file","hyp_count"]].apply(lambda x: get_expected_ranks(torch.tensor(np.array([json.loads(line)["score"][0] for line in open(f"{score_path}/{x['task']}/{x['score_file']}").readlines()]).astype(float).reshape(-1, int(x['hyp_count'])))), axis=1)
    decomposed_df["ranks"] = decomposed_df[["task", "supporting_files"]].apply(lambda x: get_expected_ranks(torch.load(f"{validated_path}/{x['task']}/{x['supporting_files']['decomposed_matrix.pt']}").mean(dim=2)), axis=1)

    avg_l2_norms, matrices_list = [], []
    for row in decomposed_df.itertuples():
        decomposed_path = f"{validated_path}/{row.task}/{row.supporting_files['decomposed_matrix.pt']}"
        original_path = decomposed_path.replace(f"{row.method}", "mbr").split("params-")[0] + "original_matrix.pt"
        original_matrix = torch.load(original_path)
        decomposed_matrix = torch.load(decomposed_path)

        if "normed" not in row.method:
            norm_dim = -1
        else:
            norm_dim = row.params.split("norm_dim-")[-1].split("-")[0]
            if norm_dim == "None":
                norm_dim = None
            else:
                norm_dim = int(norm_dim)
        
        original_matrix = _normalize_zscore(original_matrix, dim=norm_dim) if norm_dim != -1 else original_matrix
        temp_diff, temp_mat = process_matrices(original_matrix, decomposed_matrix)
        avg_l2_norms.append(temp_diff)
        matrices_list.append(temp_mat)
    decomposed_df["avg_l2_norm"] = avg_l2_norms
    decomposed_df["avg_matrices"] = matrices_list

    checking_types = {
        # "ref_type": ["task", "dataset", "method", "hyp_sampling", "hyp_count", "ref_count", "util_function"], # Check the effect of decoding method for references
        # "util_func": ["task", "dataset", "method", "hyp_sampling", "hyp_count", "ref_sampling", "ref_count"], # Check the effect of various metrics as utility function
        "each_comb": ["task", "dataset", "method", "hyp_sampling", "hyp_count", "ref_sampling", "ref_count", "util_function"], # Analyze isolated to one combination
        # "counts": ["task", "dataset", "method",  "hyp_sampling", "ref_sampling", "util_function"] # Analyze the effect of hyp/ref counts by aggregating other factors
    }
    for type_key, checking_cols in tqdm(checking_types.items(), desc="Processing different factors"):
        unique_combination = decomposed_df[checking_cols].drop_duplicates()
        unique_combination = unique_combination.sort_values(by=checking_cols)

        ranks_dict = {}

        for idx, comb_row in tqdm(unique_combination.iterrows(), total=unique_combination.shape[0], desc=f"Processing combinations for {type_key}"):
            if comb_row["task"] == "translation":
                metrics = translation_metrics
            else:
                metrics = summarization_metrics

            match_mask = np.all(decomposed_df[checking_cols] == comb_row, axis=1)
            filtered_df = decomposed_df.loc[match_mask].copy()
            filtered_df = filtered_df.sort_values(by="top_k")

            if type_key not in ["counts"]:
                vanilla_col = [col for col in checking_cols if col not in ["method"]]
                vanilla_mask = np.all(vanilla_df[vanilla_col] == comb_row[vanilla_col], axis=1)
                vanilla_scores = vanilla_df.loc[vanilla_mask, metrics]

                baseline_col = ["task", "dataset", "hyp_sampling", "hyp_count"]
                no_mbr_mask = np.all(no_mbr_df[baseline_col] == comb_row[baseline_col], axis=1)
                no_mbr_scores = no_mbr_df.loc[no_mbr_mask, metrics]

                oracle_mask = np.all(oracle_df[baseline_col] == comb_row[baseline_col], axis=1)
                oracle_scores = oracle_df.loc[oracle_mask, metrics+["method", "score_file"]]
    
                for row in vanilla_df.loc[vanilla_mask].itertuples():
                    prefix = row.util_function if type_key == "util_func" else row.ref_sampling
                    dict_key = f"{prefix}-original"
                    if dict_key not in ranks_dict:
                        ranks_dict[f"{prefix}-original"] = row.ranks
                for row in no_mbr_df.loc[no_mbr_mask].itertuples():
                    prefix = row.hyp_sampling
                    dict_key = f"{prefix}-no_mbr"
                    if dict_key not in ranks_dict:
                        ranks_dict[dict_key] = row.ranks
                for row in oracle_df.loc[oracle_mask].itertuples():
                    ranks_dict[row.method] = row.ranks 

            analysis_filename = f"{output_dir}/{type_key}/{'-'.join(map(str, comb_row[checking_cols].values))}"

            curr_param = "no_param"
            if "normed_svd" in comb_row['method']:
                curr_param = "is_reduced-True-norm_dim-0"
                # filtered_df = filtered_df.loc[~filtered_df["params"].str.contains("norm_dim-1")]
                if type_key == "each_comb":
                    generate_param_effect_fig(filtered_df, "is_reduced", metric_name=comb_row['util_function'], output_dir=analysis_filename)
                    generate_param_effect_fig(filtered_df, "norm_dim", metric_name=comb_row['util_function'], output_dir=analysis_filename)

                filtered_df = filtered_df.loc[filtered_df["params"].str.contains(curr_param)]
            elif "svd" in comb_row['method']:
                curr_param = "is_reduced-True"
                if type_key == "each_comb":
                    generate_param_effect_fig(filtered_df, "is_reduced", metric_name=comb_row['util_function'], output_dir=analysis_filename)

                filtered_df = filtered_df.loc[filtered_df["params"].str.contains(curr_param)]
            elif "nmf" in comb_row['method']:
                curr_param = "beta-0-l1_ratio-0.0"
                if type_key == "each_comb":
                    generate_param_effect_fig(filtered_df, "beta", metric_name=comb_row['util_function'], output_dir=analysis_filename)
                    generate_param_effect_fig(filtered_df, "l1_ratio", metric_name=comb_row['util_function'], output_dir=analysis_filename)

            filtered_df = filtered_df.loc[filtered_df["params"].str.contains(curr_param)]

            if type_key == "ref_type":
                generate_performance_fig(metrics, filtered_df, vanilla_scores, no_mbr_scores, oracle_scores, perf_hue_col="ref_sampling", output_dir=analysis_filename)
                for row in filtered_df.loc[filtered_df["top_k"].isin(['1','2','3',filtered_df["top_k"].max()])].itertuples():
                    ranks_dict[f"{row.ref_sampling}-top{row.top_k}"] = row.ranks
                generate_ranking_agreement_fig(ranks_dict, output_dir=analysis_filename)
                top_idx_dict = {key: val[:, 0] for key, val in ranks_dict.items()}
                generate_top_idx_agreement_fig(top_idx_dict, output_dir=analysis_filename)

            elif type_key == "util_func":
                generate_performance_fig(metrics, filtered_df, vanilla_scores, no_mbr_scores, oracle_scores, perf_hue_col="util_function", output_dir=analysis_filename)
                for row in filtered_df.loc[filtered_df["top_k"].isin(['1','2','3',filtered_df["top_k"].max()])].itertuples():
                    ranks_dict[f"{row.util_function}-top{row.top_k}"] = row.ranks
                generate_ranking_agreement_fig(ranks_dict, output_dir=analysis_filename)
                top_idx_dict = {key: val[:, 0] for key, val in ranks_dict.items()}
                generate_top_idx_agreement_fig(top_idx_dict, output_dir=analysis_filename)

            elif type_key == "each_comb":
                generate_performance_fig(metrics, filtered_df, vanilla_scores, no_mbr_scores, oracle_scores, perf_hue_col="", output_dir=analysis_filename)
                dec_method = comb_row["method"]
                original_path = filtered_df.head(1)["supporting_files"].values[0]["decomposed_matrix.pt"].replace(f"{dec_method}", "mbr").split("params-")[0] + "original_matrix.pt"
                original_matrix = torch.load(f"{validated_path}/{comb_row['task']}/{original_path}")

                if "normed" not in dec_method:
                    norm_dim = -1
                else:
                    norm_dim = filtered_df.head(1)["params"].values[0].split("norm_dim-")[-1].split("-")[0]
                    if norm_dim == "None":
                        norm_dim = None
                    else:
                        norm_dim = int(norm_dim)
                
                original_matrix = _normalize_zscore(original_matrix, dim=norm_dim) if norm_dim != -1 else original_matrix
                original_matrix = torch.mean(original_matrix, dim=0)
                matrix_list = [original_matrix] + filtered_df["avg_matrices"].tolist()
                title_list = ["Original Matrix"] + filtered_df["top_k"].apply(lambda x: f"Top {x}").tolist()
                generate_heatmap_matrices(matrix_list, title_list, output_dir=analysis_filename)
            elif type_key == "counts":
                for metric_name in metrics:
                    generate_metric_eval_fig(metric_name=metric_name, df=filtered_df, output_dir=analysis_filename)


    # checking_cols = ["task", "dataset", "hyp_sampling", "hyp_count", 
    #                 "ref_sampling", "ref_count", "method", "util_function"]
    # unique_combination = decomposed_df[checking_cols].drop_duplicates()
    # unique_combination = unique_combination.loc[unique_combination["hyp_count"].isin([64]) & (unique_combination["ref_count"].isin([4, 64, 256]))]
    # unique_combination = unique_combination.sort_values(by=checking_cols)

    # for idx, comb_row in tqdm(unique_combination.iterrows(), total=unique_combination.shape[0], desc="Processing unique combinations"):
    #     if comb_row["task"] == "translation":
    #         metrics = translation_metrics
    #     else:
    #         metrics = summarization_metrics

    #     match_mask = np.all(decomposed_df[checking_cols] == comb_row, axis=1)
    #     filtered_df = decomposed_df.loc[match_mask].copy()
    #     filtered_df = filtered_df.sort_values(by="top_k")

    #     vanilla_col = [col for col in checking_cols if col not in ["method"]]
    #     vanilla_mask = np.all(vanilla_df[vanilla_col] == comb_row[vanilla_col], axis=1)
    #     vanilla_scores = vanilla_df.loc[vanilla_mask, metrics]
    #     vanilla_support_files = vanilla_df.loc[vanilla_mask, "supporting_files"].values[0]

    #     baseline_col = ["task", "dataset", "hyp_sampling", "hyp_count"]
    #     no_mbr_mask = np.all(no_mbr_df[baseline_col] == comb_row[baseline_col], axis=1)
    #     no_mbr_scores = no_mbr_df.loc[no_mbr_mask, metrics]
    #     no_mbr_support_files = no_mbr_df.loc[no_mbr_mask, "supporting_files"].values[0]

    #     oracle_mask = np.all(oracle_df[baseline_col] == comb_row[baseline_col], axis=1)
    #     oracle_scores = oracle_df.loc[oracle_mask, metrics+["method", "score_file"]]
        
    #     original_matrix = torch.load(f"{validated_path}/{comb_row['task']}/{vanilla_support_files['original_matrix.pt']}")
    #     with open(f"{hyps_path}/{comb_row['task']}/{no_mbr_support_files['lprobs_norm']}") as f:
    #         lprobs = [line.strip() for line in f.readlines()]
    #     lprobs_idx = np.array(lprobs).astype(float).reshape(-1, int(comb_row["hyp_count"]))

    #     expected_ranks_dict = {
    #         "vanilla_mbr": get_expected_ranks(original_matrix.mean(dim=2)),
    #         "no_mbr": get_expected_ranks(torch.tensor(lprobs_idx))
    #     }
    #     for row in oracle_scores.itertuples():
    #         scores_file = f"{score_path}/{comb_row['task']}/{row.score_file}"
    #         with open(scores_file, "r") as f:
    #             row_score = [json.loads(line)["score"][0] for line in f.readlines()]
    #             row_score = np.array(row_score).astype(float).reshape(-1, int(comb_row["hyp_count"]))
    #         f.close()
    #         expected_ranks_dict[row.method] = get_expected_ranks(torch.tensor(row_score))

    #     avg_l2_norms, matrices_list, heatmap_title_list, expected_ranks_list = [], [], [], []
    #     for row in filtered_df.itertuples():
    #         decomposed_matrix = torch.load(f"{validated_path}/{row.task}/{row.supporting_files['decomposed_matrix.pt']}")
    #         if "normed" not in row.method:
    #             norm_dim = -1
    #         else:
    #             norm_dim = row.params.split("norm_dim-")[-1].split("-")[0]
    #             if norm_dim == "None":
    #                 norm_dim = None
    #             else:
    #                 norm_dim = int(norm_dim)
            
    #         if len(avg_l2_norms) == 0:  # Modify the original matrix as a baseline only for the first iteration
    #             original_matrix = _normalize_zscore(original_matrix, dim=norm_dim) if norm_dim != -1 else original_matrix
    #             temp_diff, temp_mat = process_matrices(original_matrix, original_matrix, norm_dim=norm_dim)
    #             avg_l2_norms.append(temp_diff)
    #             matrices_list.append(temp_mat)
    #             heatmap_title_list.append("Original Matrix")
    #         temp_diff, temp_mat = process_matrices(original_matrix, decomposed_matrix, norm_dim=norm_dim)
    #         avg_l2_norms.append(temp_diff)
    #         matrices_list.append(temp_mat)
    #         heatmap_title_list.append(f"Top-{row.top_k}")
    #         expected_ranks_list.append(get_expected_ranks(decomposed_matrix.mean(dim=2)))

    #     filtered_df["avg_l2_norm"] = avg_l2_norms[1:]
    #     filtered_df["matrices"] = matrices_list[1:]
    #     filtered_df["heatmap_title"] = heatmap_title_list[1:]
    #     filtered_df["expected_ranks"] = expected_ranks_list

    #     analysis_filename = f"{output_dir}/{'-'.join(map(str, comb_row[checking_cols].values))}"

    #     curr_param = "no_param"
    #     if "normed_svd" in comb_row['method']:
    #         curr_param = "is_reduced-True-norm_dim-0"
    #         generate_param_effect_fig(filtered_df, "is_reduced", metric_name=comb_row['util_function'], output_dir=analysis_filename)
    #         generate_param_effect_fig(filtered_df, "norm_dim", metric_name=comb_row['util_function'], output_dir=analysis_filename)

    #         filtered_df = filtered_df.loc[filtered_df["params"].str.contains(curr_param)]
    #     elif "svd" in comb_row['method']:
    #         curr_param = "is_reduced-True"
    #         generate_param_effect_fig(filtered_df, "is_reduced", metric_name=comb_row['util_function'], output_dir=analysis_filename)

    #         filtered_df = filtered_df.loc[filtered_df["params"].str.contains(curr_param)]
    #     elif "nmf" in comb_row['method']:
    #         curr_param = "beta-0-l1_ratio-0.0"
    #         generate_param_effect_fig(filtered_df, "beta", metric_name=comb_row['util_function'], output_dir=analysis_filename)
    #         generate_param_effect_fig(filtered_df, "l1_ratio", metric_name=comb_row['util_function'], output_dir=analysis_filename)

    #         filtered_df = filtered_df.loc[filtered_df["params"].str.contains(curr_param)]
        
    #     generate_heatmap_matrices([matrices_list[0]]+filtered_df["matrices"].tolist(), title_list=[heatmap_title_list[0]]+filtered_df["heatmap_title"].tolist(), 
    #         main_title=f"{comb_row['method']} - {comb_row['util_function']} - {curr_param} - Matrices Drift by Top-K",
    #         output_dir=analysis_filename)

    #     generate_performance_fig(metrics, filtered_df, vanilla_scores=vanilla_scores, no_mbr_scores=no_mbr_scores, oracle_scores=oracle_scores.loc[oracle_scores["method"] == f"oracle_{comb_row['util_function']}"].copy(), output_dir=analysis_filename)

    #     expected_ranks_dict.update({f"{row.method}_top{row.top_k}": row.expected_ranks for row in filtered_df.itertuples()})
    #     generate_ranking_agreement_fig(expected_ranks_dict, output_dir=analysis_filename)

    #     for metric_name in metrics:
    #         generate_eval_fig(metric_name, filtered_df, output_dir=analysis_filename)

    # checking_cols = ["task", "dataset", "hyp_sampling", 
    #                 "ref_sampling", "method", "util_function"]
    # unique_combination = decomposed_df[checking_cols].drop_duplicates()
    # unique_combination = unique_combination.sort_values(by=checking_cols)

    # for idx, comb_row in tqdm(unique_combination.iterrows(), total=unique_combination.shape[0], desc="Processing unique combinations"):
    #     if comb_row["task"] == "translation":
    #         metrics = translation_metrics
    #     else:
    #         metrics = summarization_metrics

    #     match_mask = np.all(decomposed_df[checking_cols] == comb_row, axis=1)
    #     filtered_df = decomposed_df.loc[match_mask].copy()
    #     filtered_df = filtered_df.sort_values(by="top_k")

    #     avg_l2_norms, matrices_list, heatmap_title_list = [], [], []
    #     for row in filtered_df.itertuples():
    #         decomposed_path = f"{validated_path}/{row.task}/{row.supporting_files['decomposed_matrix.pt']}"
    #         original_path = decomposed_path.replace(f"{row.method}", "mbr").split("params-")[0] + "original_matrix.pt"
    #         original_matrix = torch.load(original_path)
    #         decomposed_matrix = torch.load(decomposed_path)

    #         if "normed" not in row.method:
    #             norm_dim = -1
    #         else:
    #             norm_dim = row.params.split("norm_dim-")[-1].split("-")[0]
    #             if norm_dim == "None":
    #                 norm_dim = None
    #             else:
    #                 norm_dim = int(norm_dim)
            
    #         if len(avg_l2_norms) == 0:  # Modify the original matrix as a baseline only for the first iteration
    #             original_matrix = _normalize_zscore(original_matrix, dim=norm_dim) if norm_dim != -1 else original_matrix
    #             temp_diff, temp_mat = process_matrices(original_matrix, original_matrix, norm_dim=norm_dim)
    #             avg_l2_norms.append(temp_diff)
    #             matrices_list.append(temp_mat)
    #             heatmap_title_list.append("Original Matrix")
    #         temp_diff, temp_mat = process_matrices(original_matrix, decomposed_matrix, norm_dim=norm_dim)
    #         avg_l2_norms.append(temp_diff)
    #         matrices_list.append(temp_mat)
    #         heatmap_title_list.append(f"Top-{row.top_k}")

    #     filtered_df["avg_l2_norm"] = avg_l2_norms[1:]
    #     filtered_df["matrices"] = matrices_list[1:]
    #     filtered_df["heatmap_title"] = heatmap_title_list[1:]

    #     analysis_filename = f"{output_dir}/{'-'.join(map(str, comb_row[checking_cols].values))}"

    #     curr_param = "no_param"
    #     if "normed_svd" in comb_row['method']:
    #         curr_param = "is_reduced-True-norm_dim-0"
    #         filtered_df = filtered_df.loc[filtered_df["params"].str.contains(curr_param)]
    #     elif "svd" in comb_row['method']:
    #         curr_param = "is_reduced-True"
    #         filtered_df = filtered_df.loc[filtered_df["params"].str.contains(curr_param)]
    #     elif "nmf" in comb_row['method']:
    #         curr_param = "beta-0-l1_ratio-0.0"
    #         filtered_df = filtered_df.loc[filtered_df["params"].str.contains(curr_param)]

    #     for metric_name in metrics:
    #         generate_metric_eval_fig(metric_name, filtered_df, output_dir=analysis_filename)