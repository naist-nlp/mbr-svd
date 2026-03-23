import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

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

def generate_eval_fig(metric_name, decomposed_data, vanilla_score, no_mbr_score, oracle_score):
    """
    Generate a single fig image that contains the performance visualization with the avg_l2_norm line plot 
    on the secondary y-axis, and the horizontal lines for vanilla, no MBR, and oracle scores on the primary y-axis. 
    The legend is placed outside the plot area to the right.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare categorical x-axis
    numeric_sort = sorted(pd.to_numeric(decomposed_data['top_k_sv'].unique()))
    str_categories = [str(x) for x in numeric_sort]
    decomposed_data['top_k_sv'] = pd.Categorical(
        decomposed_data['top_k_sv'].astype(str), 
        categories=str_categories, 
        ordered=True
    )

    # --- PRIMARY AXIS (Blue) ---
    blue_color = 'tab:blue'
    sns.lineplot(x=decomposed_data["top_k_sv"], y=decomposed_data[metric_name], 
                marker="o", color=blue_color, label=f"{metric_name} score", ax=ax)
    
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

    add_h_line(vanilla_score, "orange", "Vanilla MBR")
    add_h_line(no_mbr_score, "green", "No MBR")
    add_h_line(oracle_score, "grey", "Oracle")

    # --- SECONDARY AXIS (Red) ---
    ax2 = ax.twinx()
    red_color = 'tab:red'
    sns.lineplot(x=decomposed_data["top_k_sv"], y=decomposed_data["avg_l2_norm"], 
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
    
    return fig

def generate_heatmap_matrices(tensor_list, title_list, main_title="MBR Matrices Drift by Top-K"):
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
        ax.set_xlabel("Hypothesis")
        ax.set_ylabel("Pseudo-Ref")
        
        # If we hid the text labels, also turn off the tiny tick line marks for a clean look
        if hide_details:
            ax.tick_params(left=False, bottom=False)
            
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

def generate_param_effect_fig(df, param_name, metric_name="comet"):
    """
    Visualizes the effect of a specific parameter (e.g., 'beta' or 'l1_ratio')
    isolated strictly to rank 1 and rank 2.
    """
    print(f"--- Preparing data to analyze '{param_name}' for Rank 1 & 2 ---")
    
    # 1. Helper function to safely parse the param string
    def parse_param(text):
        return text.split(f"{param_name}-")[-1].split("-")[0]
        
    # Work on a copy of the dataframe
    df_plot = df.copy()
    df_plot[param_name] = df_plot['params'].apply(parse_param)

    # Safely look for either 'rank' (NMF) or 'top_k_sv' (SVD) keys
    df_plot['rank_val'] = df_plot['rank'] if "nmf" in df_plot["method"].unique()[0] else df_plot["top_k_sv"]
    
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
    return fig

# def _calculate_matrix_diff(ori_mat, dec_mat):
#     return ori_mat - dec_mat

# def _calculate_l2_norm(diff_tensor):
#     return torch.linalg.norm(diff_tensor, ord='fro')

# def process_matrices(original_matrix, reconstructed_matrix):
#     original_matrix = torch.mean(original_matrix, dim=0)
#     reconstructed_matrix = torch.mean(reconstructed_matrix, dim=0)
#     diff = _calculate_matrix_diff(original_matrix, reconstructed_matrix)
#     error_val = _calculate_l2_norm(diff)
#     return {
#         "avg_original_matrix": original_matrix,
#         "avg_reconstructed_matrix": reconstructed_matrix,
#         "avg_diff_matrix": diff,
#         "avg_error_value": error_val
#     }

# def visualize_histogram(tensor, output_file):
#     if len(tensor) == 0:
#         return
#     error_values = _calculate_l2_norm(tensor)
    
#     plt.figure(figsize=(8, 6))
#     sns.histplot(error_values.numpy(), bins=50, color='teal', alpha=0.7)
#     plt.title("Distribution of Reconstruction Errors (L2 Norm)")
#     plt.xlabel("Error Score (Lower is Better)")
#     plt.ylabel("Count of Data Points")
#     plt.axvline(x=torch.mean(error_values).item(), color='red', linestyle='--', label='Mean Error')
#     plt.legend()
#     plt.savefig(output_file)
#     plt.close()

# def visualize_diff_heatmap(tensor, output_file):
#     if len(tensor) == 0:
#         return

#     # 1. Calculate Mean Absolute Difference (Magnitude of error)
#     # Good for spotting WHERE errors happen
#     avg_abs_diff = torch.mean(torch.abs(tensor), dim=0).cpu()

#     # 2. Calculate Mean Signed Difference (Bias)
#     # Good for spotting if you consistently overestimate (+) or underestimate (-)
#     avg_signed_diff = torch.mean(tensor, dim=0).cpu()
    
#     # Create a figure with 2 subplots side-by-side
#     plt.figure(figsize=(16, 6))
    
#     # --- Plot 1: Absolute Difference ---
#     sns.heatmap(avg_abs_diff.numpy(), 
#                 cmap='Reds')
#     plt.title("Mean ABSOLUTE Difference\n(How large is the error?)")
#     plt.xlabel("Matrix Columns")
#     plt.ylabel("Matrix Rows")
#     plt.tight_layout()
#     plt.savefig(output_file.replace(".png", "_abs.png"))
#     plt.close()

#     # --- Plot 2: Signed (Raw) Difference ---
#     plt.figure(figsize=(16, 6))
#     sns.heatmap(avg_signed_diff.numpy(), 
#                 cmap='RdBu_r', # Diverging color (Blue -> White -> Red)
#                 center=0)      # Forces White to be exactly 0
#     plt.title("Mean SIGNED Difference\n(Bias: Blue=Under / Red=Over)")
#     plt.xlabel("Matrix Columns")
#     plt.ylabel("Matrix Rows")
    
#     plt.tight_layout()
#     plt.savefig(output_file.replace(".png", "_signed.png"))
#     plt.close()

# def main(decode_content, output_path):
#     matrices_diff = _calculate_matrices_diff(decode_content)
#     print(matrices_diff.shape)
#     if not os.path.exists(os.path.join(output_path, "reconstruction_error_heatmaps.png")):
#         visualize_diff_heatmap(matrices_diff, os.path.join(output_path, "reconstruction_error_heatmaps.png"))
#     if not os.path.exists(os.path.join(output_path, "reconstruction_error_histogram.png")):
#         visualize_histogram(matrices_diff, os.path.join(output_path, "reconstruction_error_histogram.png"))