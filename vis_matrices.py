import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

def _calculate_matrices_diff(decode_content):
    original_matrices, reconstructed_matrices = [], []
    if decode_content[0]["original_matrix"] == None:
        return {}
    for row_res in decode_content:
        original_matrix = torch.tensor(row_res["original_matrix"]["data"])
        reconstructed_matrix = torch.tensor(row_res["decomposed_matrix"]["data"])
        original_matrices.append(original_matrix)
        reconstructed_matrices.append(reconstructed_matrix)
    original_matrices = torch.stack(original_matrices)
    reconstructed_matrices = torch.stack(reconstructed_matrices)
    diffs = original_matrices - reconstructed_matrices
    return diffs

def _calculate_l2_norm(tensor):
    return torch.linalg.norm(tensor, dim=(1,2), ord='fro')

def visualize_histogram(tensor, output_file):
    if len(tensor) == 0:
        return
    error_values = _calculate_l2_norm(tensor)
    
    plt.figure(figsize=(8, 6))
    sns.histplot(error_values.numpy(), bins=50, color='teal', alpha=0.7)
    plt.title("Distribution of Reconstruction Errors (L2 Norm)")
    plt.xlabel("Error Score (Lower is Better)")
    plt.ylabel("Count of Data Points")
    plt.axvline(x=torch.mean(error_values).item(), color='red', linestyle='--', label='Mean Error')
    plt.legend()
    plt.savefig(output_file)
    plt.close()

def visualize_diff_heatmap(tensor, output_file):
    if len(tensor) == 0:
        return

    # 1. Calculate Mean Absolute Difference (Magnitude of error)
    # Good for spotting WHERE errors happen
    avg_abs_diff = torch.mean(torch.abs(tensor), dim=0).cpu()

    # 2. Calculate Mean Signed Difference (Bias)
    # Good for spotting if you consistently overestimate (+) or underestimate (-)
    avg_signed_diff = torch.mean(tensor, dim=0).cpu()
    
    # Create a figure with 2 subplots side-by-side
    plt.figure(figsize=(16, 6))
    
    # --- Plot 1: Absolute Difference ---
    sns.heatmap(avg_abs_diff.numpy(), 
                cmap='Reds')
    plt.title("Mean ABSOLUTE Difference\n(How large is the error?)")
    plt.xlabel("Matrix Columns")
    plt.ylabel("Matrix Rows")
    plt.tight_layout()
    plt.savefig(output_file.replace(".png", "_abs.png"))
    plt.close()

    # --- Plot 2: Signed (Raw) Difference ---
    plt.figure(figsize=(16, 6))
    sns.heatmap(avg_signed_diff.numpy(), 
                cmap='RdBu_r', # Diverging color (Blue -> White -> Red)
                center=0)      # Forces White to be exactly 0
    plt.title("Mean SIGNED Difference\n(Bias: Blue=Under / Red=Over)")
    plt.xlabel("Matrix Columns")
    plt.ylabel("Matrix Rows")
    
    plt.tight_layout()
    plt.savefig(output_file.replace(".png", "_signed.png"))
    plt.close()

def main(decode_content, output_path):
    matrices_diff = _calculate_matrices_diff(decode_content)
    if not os.path.exists(os.path.join(output_path, "reconstruction_error_heatmaps.png")):
        visualize_diff_heatmap(matrices_diff, os.path.join(output_path, "reconstruction_error_heatmaps.png"))
    if not os.path.exists(os.path.join(output_path, "reconstruction_error_histogram.png")):
        visualize_histogram(matrices_diff, os.path.join(output_path, "reconstruction_error_histogram.png"))