import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from vis_load_data import validated_path, translation_metrics, summarization_metrics

def _calculate_matrix_diff(ori_mat, dec_mat):
    return ori_mat - dec_mat

def _calculate_l2_norm(diff_tensor):
    return torch.linalg.norm(diff_tensor, ord='fro').item()

def _normalize_zscore(tensor, dim=None, epsilon=1e-8):
    if dim is None:
        mean = torch.mean(tensor)
        std = torch.std(tensor)
    else:
        mean = torch.mean(tensor, dim=dim, keepdim=True)
        std = torch.std(tensor, dim=dim, keepdim=True)
    normalized_tensor = (tensor - mean) / (std + epsilon)
    return normalized_tensor

def process_matrices(original_matrix, reconstructed_matrix, norm_dim: int | None = -1):
    original_matrix = torch.mean(original_matrix, dim=0)
    reconstructed_matrix = torch.mean(reconstructed_matrix, dim=0)
    diff = _calculate_matrix_diff(original_matrix, reconstructed_matrix)
    error_val = _calculate_l2_norm(diff)
    return error_val, reconstructed_matrix

def save_figure(fig, output_file, output_ext="png"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.savefig(f"{output_file}.{output_ext}", format=output_ext, bbox_inches='tight')
    plt.close(fig)


def generate_performance_fig(metrics, decomposed_data, vanilla_scores, no_mbr_scores, oracle_scores, output_dir=""):
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
        sns.lineplot(x=decomposed_data["top_k"], y=decomposed_data[metric_name], 
                    marker="o", color=blue_color, label=f"Score", ax=ax)
        
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

        add_h_line(vanilla_scores[metric_name].values[0] if metric_name in vanilla_scores else 0.0, "orange", "Vanilla MBR")
        add_h_line(no_mbr_scores[metric_name].values[0] if metric_name in no_mbr_scores else 0.0, "green", "No MBR")
        add_h_line(oracle_scores[metric_name].values[0] if not oracle_scores.empty else 0.0, "grey", "Oracle")

        # --- COMBINED LEGEND OUTSIDE ---
        lines, labels = ax.get_legend_handles_labels()
        ax.legend().remove()  # Remove the default legend from this subplot since we'll create a shared one outside
        
        # bbox_to_anchor=(1.15, 1) moves the legend to the right and top
        shared_lines.extend(lines)
        shared_labels.extend(labels)
        ax.set_xlabel("Top-k")
        ax.set_title(metric_name)

    # Remove empty subplots if metrics < nrows*ncols
    for i in range(len(metrics), nrows * ncols):
        fig.delaxes(axes.flatten()[i])

    axes[0][0].legend(list(set(shared_lines)), list(set(shared_labels)), 
        loc='upper left', bbox_to_anchor=(1.05, 1.0))

    fig.suptitle(f"Performance Comparison Across Metrics")
    
    # Adjust layout: We need extra room on the right for both labels AND legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
        
    save_figure(fig, f"{output_dir}-metrics_perf_compare")


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
    isolated strictly to rank 1 and rank 2.
    """
    print(f"--- Preparing data to analyze '{param_name}' for Rank 1 & 2 ---")
    
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

    # Convert to numeric for accurate filtering and plotting
    # df_plot[param_name] = pd.to_numeric(df_plot[param_name])
    df_plot['rank_val'] = pd.to_numeric(df_plot['rank_val'])
    
    # 3. FILTER: Keep ONLY rows where rank is exactly 1 or 2
    df_plot = df_plot[df_plot['rank_val'].isin([1, 2])]
    
    if df_plot.empty:
        print(f"No data found for '{param_name}' at rank 1 or 2.")
        return
        
    # Convert rank_val to a string so Seaborn treats it as a distinct category for colors
    df_plot['rank_val'] = "Rank " + df_plot['rank_val'].astype(str)

    df_plot = df_plot.sort_values(by=[param_name, 'rank_val'])

    # 4. Set up the figure (1 row, 2 columns)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    fig.suptitle(f"Effect of '{param_name}' (Isolated to Rank 1 & 2)", fontsize=16, fontweight='bold')

    # Plot 1: {metric_name} vs param_name (Left)
    sns.lineplot(
        data=df_plot, x=param_name, y=metric_name, hue='rank_val', 
        marker='o', palette='Set1', ax=axes[0]
    )
    axes[0].set_title(f'{metric_name.upper()} vs {param_name}')
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel(f'{metric_name.upper()}')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: avg_l2_norm vs param_name (Right)
    sns.lineplot(
        data=df_plot, x=param_name, y='avg_l2_norm', hue='rank_val', 
        marker='s', palette='Set1', ax=axes[1]
    )
    axes[1].set_title(f'Avg L2 Norm vs {param_name}')
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel('Avg L2 Norm')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Adjust layout and display
    plt.tight_layout()
    
    save_figure(fig, f"{output_dir}-{param_name}_effect_plot")

def main(decomposed_df, vanilla_df, no_mbr_df, oracle_df, output_dir=""):
    if "nmf" in decomposed_df["method"].unique()[0]:
        decomposed_df["top_k"] = decomposed_df["params"].apply(lambda x: int(x.split("rank-")[-1].split("-")[0]))
    else:
        decomposed_df["top_k"] = decomposed_df["params"].apply(lambda x: int(x.split("top_k_sv-")[-1].split("-")[0]))

    checking_cols = ["task", "dataset", "hyp_sampling", "hyp_count", 
                    "ref_sampling", "ref_count", "method", "util_function"]
    unique_combination = decomposed_df[checking_cols].drop_duplicates()

    for idx, comb_row in tqdm(unique_combination.iterrows(), total=unique_combination.shape[0], desc="Processing unique combinations"):
        if comb_row["task"] == "translation":
            metrics = translation_metrics
        else:
            metrics = summarization_metrics

        match_mask = np.all(decomposed_df[checking_cols] == comb_row, axis=1)
        filtered_df = decomposed_df.loc[match_mask].copy()
        filtered_df = filtered_df.sort_values(by="top_k")

        vanilla_col = [col for col in checking_cols if col not in ["method"]]
        vanilla_mask = np.all(vanilla_df[vanilla_col] == comb_row[vanilla_col], axis=1)
        vanilla_scores = vanilla_df.loc[vanilla_mask, metrics]
        vanilla_support_files = vanilla_df.loc[vanilla_mask, "supporting_files"].values[0]

        baseline_col = ["task", "dataset", "hyp_sampling", "hyp_count"]
        no_mbr_mask = np.all(no_mbr_df[baseline_col] == comb_row[baseline_col], axis=1)
        no_mbr_scores = no_mbr_df.loc[no_mbr_mask, metrics]

        oracle_mask = np.all(oracle_df[baseline_col] == comb_row[baseline_col], axis=1)
        oracle_scores = oracle_df.loc[oracle_mask, metrics+["method"]]
        
        avg_l2_norms, matrices_list, heatmap_title_list = [], [], []
        original_matrix = torch.load(f"{validated_path}/{comb_row['task']}/{vanilla_support_files['original_matrix.pt']}")
        for row in filtered_df.itertuples():
            decomposed_matrix = torch.load(f"{validated_path}/{row.task}/{row.supporting_files['decomposed_matrix.pt']}")

            if "normed" not in row.method:
                norm_dim = -1
            else:
                norm_dim = row.params.split("norm_dim-")[-1].split("-")[0]
                if norm_dim == "None":
                    norm_dim = None
                else:
                    norm_dim = int(norm_dim)
            
            if len(avg_l2_norms) == 0:  # Modify the original matrix as a baseline only for the first iteration
                original_matrix = _normalize_zscore(original_matrix, dim=norm_dim) if norm_dim != -1 else original_matrix
                temp_diff, temp_mat = process_matrices(original_matrix, original_matrix, norm_dim=norm_dim)
                avg_l2_norms.append(temp_diff)
                matrices_list.append(temp_mat)
                heatmap_title_list.append("Original Matrix")
            temp_diff, temp_mat = process_matrices(original_matrix, decomposed_matrix, norm_dim=norm_dim)
            avg_l2_norms.append(temp_diff)
            matrices_list.append(temp_mat)
            heatmap_title_list.append(f"Top-{row.top_k}")

        filtered_df["avg_l2_norm"] = avg_l2_norms[1:]
        filtered_df["matrices"] = matrices_list[1:]
        filtered_df["heatmap_title"] = heatmap_title_list[1:]

        analysis_filename = f"{output_dir}/{'-'.join(map(str, comb_row[checking_cols].values))}"

        curr_param = "no_param"
        if "normed_svd" in comb_row['method']:
            curr_param = "is_reduced-True-norm_dim-0"
            generate_param_effect_fig(filtered_df, "is_reduced", metric_name=comb_row['util_function'], output_dir=analysis_filename)
            generate_param_effect_fig(filtered_df, "norm_dim", metric_name=comb_row['util_function'], output_dir=analysis_filename)

            filtered_df = filtered_df.loc[filtered_df["params"].str.contains(curr_param)]
        elif "svd" in comb_row['method']:
            curr_param = "is_reduced-True"
            generate_param_effect_fig(filtered_df, "is_reduced", metric_name=comb_row['util_function'], output_dir=analysis_filename)

            filtered_df = filtered_df.loc[filtered_df["params"].str.contains(curr_param)]
        elif "nmf" in comb_row['method']:
            curr_param = "beta-0-l1_ratio-0.0"
            generate_param_effect_fig(filtered_df, "beta", metric_name=comb_row['util_function'], output_dir=analysis_filename)
            generate_param_effect_fig(filtered_df, "l1_ratio", metric_name=comb_row['util_function'], output_dir=analysis_filename)

            filtered_df = filtered_df.loc[filtered_df["params"].str.contains(curr_param)]
        
        generate_heatmap_matrices([matrices_list[0]]+filtered_df["matrices"].tolist(), title_list=[heatmap_title_list[0]]+filtered_df["heatmap_title"].tolist(), 
            main_title=f"{comb_row['method']} - {comb_row['util_function']} - {curr_param} - Matrices Drift by Top-K",
            output_dir=analysis_filename)

        generate_performance_fig(metrics, filtered_df, vanilla_scores=vanilla_scores, no_mbr_scores=no_mbr_scores, oracle_scores=oracle_scores.loc[oracle_scores["method"] == f"oracle_{comb_row['util_function']}"].copy(), output_dir=analysis_filename)

        for metric_name in metrics:
            generate_eval_fig(metric_name, filtered_df, output_dir=analysis_filename)