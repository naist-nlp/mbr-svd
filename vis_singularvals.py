import torch
import matplotlib.pyplot as plt
import os

def avg_singular_values(decode_content):
    if decode_content[0]["singularvals"] == None:
        return torch.Tensor([])
    sv_lists = []
    for row_res in decode_content:
        sv = row_res["singularvals"]["data"]
        sv_lists.append(sv)
    sv_tensor = torch.tensor(sv_lists)
    avg_sv = torch.mean(sv_tensor, dim=0)
    return avg_sv

def visualize_singular_values(singular_values, output_file):
    if len(singular_values) == 0:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(singular_values) + 1), singular_values.numpy(), marker='o', linestyle='-')
    plt.title("Average Singular Values")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def main(decode_content, max_dim, output_path):
    avg_sv = avg_singular_values(decode_content)
    if avg_sv.shape[0] < max_dim:
        return
    if not os.path.exists(os.path.join(output_path, "average_singular_values.png")):
        visualize_singular_values(avg_sv, os.path.join(output_path, "average_singular_values.png"))
