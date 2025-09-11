import os
import json
import math
import numpy as np


def find_cutoff_otsu(data: list, score_key: str = 'personalization_score', bins: int = 256):
    """
    Use the Otsu method to find the optimal split point, applying Otsu directly to all data.

    Args:
        data: A list of records sorted by score_key in descending order.
        score_key: The field representing the score.
        bins: Number of bins used for Otsu histogram calculation.

    Returns:
        cutoff_index: The cutoff index, where data[:cutoff_index] corresponds to the high-score data to be removed.
        threshold: The score threshold corresponding to the cutoff.
    """

    total_len = len(data)

    # Extract all scores
    scores = np.array([item[score_key] for item in data])

    # Compute histogram required by Otsu method
    hist, bin_edges = np.histogram(scores, bins=bins)
    total = scores.size
    prob = hist.astype(np.float32) / total

    # Compute cumulative probability and cumulative mean
    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(prob * bin_edges[:-1])
    global_mean = cum_mean[-1]

    best_between = -np.inf
    best_threshold = None

    # Find the optimal split point
    for i in range(bins):
        if cum_prob[i] == 0 or cum_prob[i] == 1:
            continue
        mean1 = cum_mean[i] / cum_prob[i]
        mean2 = (global_mean - cum_mean[i]) / (1 - cum_prob[i])
        between = cum_prob[i] * (1 - cum_prob[i]) * (mean1 - mean2) ** 2
        if between > best_between:
            best_between = between
            best_threshold = bin_edges[i]

    # Find cutoff index based on threshold
    cutoff_index = next((i for i, item in enumerate(data) if item[score_key] < best_threshold), total_len)

    threshold = best_threshold

    return cutoff_index, threshold


def process_client(client_index: int):
    """
    Process a single client:
      1. Read data from local_training_{i}_v1.json and local_training_{i}_v2.json,
         and compute personalization score (aggregated_perplexity / local_perplexity).
      2. Merge data and sort by personalization score in descending order.
      3. Find the optimal cutoff point within the top 50% so that the mean difference
         between the two parts before and after the cutoff is maximized.
      4. Save the full sorted data to the local folder,
         and save the data after the cutoff (excluding the first part) to the global folder.
    """
    # Define input file names
    local_json_path = f"local_training_{client_index}_v1.json"  # local_training_x_v1.json
    aggregated_json_path = f"local_training_{client_index}_v2.json"  # local_training_x_v2.json

    # Read data
    with open(local_json_path, 'r', encoding='utf-8') as f:
        local_data = json.load(f)
    with open(aggregated_json_path, 'r', encoding='utf-8') as f:
        aggregated_data = json.load(f)

    merged_data = []
    # Traverse both datasets (assuming they are aligned in order)
    for sample_local, sample_agg in zip(local_data, aggregated_data):
        local = sample_local.get("local_perplexity")
        aggregated = sample_agg.get("aggregated_perplexity")
        if local is None or aggregated is None:
            continue
        # Compute personalization score, avoid division by zero
        score = aggregated / local if local > 0 else float('inf')
        # Merge both dictionaries and add personalization_score field
        merged_sample = sample_local.copy()
        merged_sample.update(sample_agg)
        merged_sample["personalization_score"] = score
        merged_data.append(merged_sample)

    # Sort by personalization score in descending order
    merged_data.sort(key=lambda x: x["personalization_score"], reverse=True)

    # Path to save logs
    log_file = "client_data_log.txt"

    # Find the optimal cutoff point within the top 40%
    try:
        cutoff, gap = find_cutoff_otsu(merged_data, score_key="personalization_score")
    except ValueError as e:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Client {client_index}: {str(e)}\n")
        return

    total = len(merged_data)
    retained = total - cutoff
    retain_ratio = retained / total

    # Write logs
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"Client {client_index}: Best cutoff index = {cutoff}, variance = {gap:.4f}\n")
        f.write(
            f"Client {client_index}: Total = {total}, Retained = {retained}, Retention ratio = {retain_ratio:.2%}\n")

    keys_to_keep = ["instruction", "context", "response", "category", "personalization_score"]

    new_data = []
    for record in merged_data:
        new_record = {k: record.get(k) for k in keys_to_keep}
        new_data.append(new_record)

    # Output directories, create if not exist
    local_output_dir = "local1/8"
    global_output_dir = "global1/8"
    remain_output_dir = "remain1/8"
    os.makedirs(local_output_dir, exist_ok=True)
    os.makedirs(global_output_dir, exist_ok=True)
    os.makedirs(remain_output_dir, exist_ok=True)

    # Save full sorted data to local folder
    local_output_path = os.path.join(local_output_dir, f"local_training_{client_index}.json")
    with open(local_output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

    # Save data after cutoff to global folder
    global_data = new_data[cutoff:]
    global_output_path = os.path.join(global_output_dir, f"local_training_{client_index}.json")
    with open(global_output_path, 'w', encoding='utf-8') as f:
        json.dump(global_data, f, indent=4, ensure_ascii=False)

    # Save data before cutoff to remain folder
    remain_data = new_data[:cutoff]
    remain_output_path = os.path.join(remain_output_dir, f"local_training_{client_index}.json")
    with open(remain_output_path, 'w', encoding='utf-8') as f:
        json.dump(remain_data, f, indent=4, ensure_ascii=False)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"Client {client_index}: Full data saved to {local_output_path}, post-cutoff data saved to {global_output_path}\n\n")


def main():
    # Process 8 clients
    for i in range(8):
        process_client(i)


if __name__ == "__main__":
    main()
