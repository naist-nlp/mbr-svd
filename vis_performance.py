import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from vis_load_data import metrics

def compare_selection_indexes(all_indexes: dict, comparer_indexes: dict, title: str, output_file: str):
    """
    Args:
    """

    # ==========================================
    # 1. Pairwise Agreement Matrix
    # ==========================================
    # We create a 3x3 matrix where cell [i, j] is the % overlap
    strategies = list(all_indexes.keys())
    comparer = list(comparer_indexes.keys())
    data_map = all_indexes
    comparer_map = comparer_indexes
    
    agreement_matrix = np.zeros((len(comparer), len(strategies)))
    
    for i, name_a in enumerate(comparer):
        for j, name_b in enumerate(strategies):
            # Calculate mean equality (percentage of matches)
            match_pct = np.mean(np.array(comparer_map[name_a]) == np.array(data_map[name_b]))
            agreement_matrix[i, j] = match_pct

    # ==========================================
    # 3. Visualization
    # ==========================================
    fig = plt.figure(figsize=(20, 6))
    sns.heatmap(agreement_matrix, annot=True, fmt=".1%", cmap="Blues", 
                xticklabels=strategies, yticklabels=comparer)
    fig.suptitle(title)
    
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def focus_ylim(ax, margin=0.2):
    """
    Adjusts the y-limits of a single axis to zoom in on the data.
    margin: Percentage of padding to add above/below the data range (default 20%)
    """
    y_values = []
    
    # 1. Get heights of all bars
    # ax.patches contains the rectangles for the bar chart
    for bar in ax.patches:
        y_values.append(bar.get_height())
        
    # 2. Get positions of horizontal lines (Oracle, LProbs, etc.)
    # ax.get_lines() contains the plot() lines or axhline()
    for line in ax.get_lines():
        y_data = line.get_ydata()
        if len(y_data) > 0:
            y_values.extend(y_data)
    
    if not y_values:
        return

    # 3. Calculate Min/Max
    min_y = min(y_values)
    max_y = max(y_values)
    data_range = max_y - min_y
    
    # Handle edge case where all values are identical
    if data_range == 0:
        data_range = max_y * 0.1

    # 4. Set the new limits
    # We subtract margin from the bottom and add it to the top
    ax.set_ylim(bottom=min_y - (data_range * margin), 
                top=max_y + (data_range * margin))
    
def vis_performance(all_indexes: dict, scores: pd.Series, title: str):
    """
    Args:
    """

    # ==========================================
    # 1. Bar chart of performance comparison
    # ==========================================
    result = []
    for metric in metrics:
        oracle_idx = scores[f"{metric}_oracle_idx"]
        metric_score = np.array(scores[metric]).astype(float).reshape(-1, int(scores["hyp_type"].split(".")[-1]))
        oracle_score = np.mean(np.take_along_axis(metric_score, np.array(oracle_idx).reshape(-1,1), axis=1))
        vanilla_score = np.mean(np.take_along_axis(metric_score, np.array(scores["vanilla_mbr_idx"]).reshape(-1,1), axis=1))
        lprobs_score = np.mean(np.take_along_axis(metric_score, np.array(scores["lprobs_idx"]).reshape(-1,1), axis=1))
        for top_k in all_indexes.keys():
            decomp_idx = all_indexes[top_k]
            decomp_score = np.mean(np.take_along_axis(metric_score, np.array(decomp_idx).reshape(-1,1), axis=1))
            result.append({
                "metric": metric,
                "strategy": top_k,
                "score": decomp_score,
                "oracle": oracle_score,
                "lprobs": lprobs_score,
                "vanilla_mbr": vanilla_score,
            })
    result_df = pd.DataFrame(result)
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
    for metric, ax in zip(metrics, axes):
        ax_data = result_df.loc[result_df["metric"] == metric]
        sns.barplot(data=ax_data, x="strategy", y="score", ax=ax)
        # exponent = int(np.floor(np.log10(np.abs(ax_data["score"].min()))))
        # magnitude = 10 ** exponent
        # ax.set_ylim(np.floor(ax_data["score"].min() / magnitude) * magnitude, np.ceil(ax_data["oracle"].max() / magnitude) * magnitude)
        ax.axhline(y=ax_data["oracle"].values[0], color='r', linestyle='--', label='Oracle Score')
        ax.axhline(y=ax_data["lprobs"].values[0], color='g', linestyle='--', label='LProbs Score')
        ax.axhline(y=ax_data["vanilla_mbr"].values[0], color='b', linestyle='--', label='Vanilla MBR Score')
        ax.set_title(metric)
        handles, labels = ax.get_legend_handles_labels()
        focus_ylim(ax)
        # ax.get_legend().remove()
    fig.legend(handles, labels, loc='upper right')

    plt.suptitle(title)
    plt.show()

def extract_selected_idx(decode_content):
    if decode_content[0]["selected_idx"] == None:
        return {}
    selected_indexes = []
    for row_res in decode_content:
        selected_idx = row_res["selected_idx"]
        selected_indexes.append(selected_idx)
    assert len(selected_indexes) == len(decode_content), "Length mismatch: selected_indexes {}, decode_content {}".format(len(selected_indexes), len(decode_content))
    return selected_indexes

def main(data: pd.DataFrame, scores: pd.Series, comparer_indexes: dict, output_path):
    data["selected_idx"] = data["decode_content"].apply(extract_selected_idx)
    all_indexes = data[["param","selected_idx"]].set_index(data["param"].apply(lambda x: x.get("top_k_sv", None))).drop(["param"], axis=1).to_dict(orient="index")
    all_indexes = {k: v["selected_idx"] for k, v in all_indexes.items()}
    comparer_indexes["vanilla_mbr"] = all_indexes[list(all_indexes.keys())[0]]  # Assuming the first key corresponds to vanilla MBR
    del all_indexes[list(all_indexes.keys())[0]]
    compare_selection_indexes(all_indexes, comparer_indexes, 
                             title="Selection Index Agreement Comparison",
                             output_file=f"{output_path}/selection_index_agreement.png")