import json
import os


def process_client_data_by_score(epoch, client_index, threshold=1.0):
    """
    Filter and save data based on personalization_score < threshold.

    Args:
    epoch (int): Current global epoch
    client_index (int): Client index
    threshold (float, optional): Filtering threshold, default is 1.0
    """
    print("epoch = ", epoch, "client_index = ", client_index)
    local_json_path = f"data_train/8/local_training_{client_index}_v1_{epoch-1}.json"
    aggregated_json_path = f"data_train/8/local_training_{client_index}_v2_{epoch-1}.json"

    log_file = f"data_train/8/client_data_log_{epoch}.txt"

    try:
        with open(local_json_path, 'r', encoding='utf-8') as f:
            local_data = json.load(f)
        with open(aggregated_json_path, 'r', encoding='utf-8') as f:
            aggregated_data = json.load(f)
    except FileNotFoundError as e:
        print(f"File read failed: {e}")
        return

    merged_data = []
    # Traverse the data of two files (assuming the same order)
    for sample_local, sample_agg in zip(local_data, aggregated_data):
        local = sample_local.get("local_perplexity")
        aggregated = sample_agg.get("aggregated_perplexity")
        if local is None or aggregated is None:
            continue
        # Calculate personalized score to avoid local being 0
        score = aggregated / local if local > 0 else float('inf')
        # Merge two dictionaries and add scores
        merged_sample = sample_local.copy()
        merged_sample.update(sample_agg)
        merged_sample["personalization_score"] = score
        merged_data.append(merged_sample)

    if not merged_data:
        print(f"Client {client_index}: No valid data, skip processing.")
        return

    # Descending by personalization_score (optional)
    merged_data.sort(key=lambda x: x["personalization_score"], reverse=True)

    # Filter by threshold
    selected_data = [item for item in merged_data if item.get('personalization_score', 0) < threshold]
    remain_data = [item for item in merged_data if item.get('personalization_score', 0) >= threshold]

    total = len(merged_data)
    selected = len(selected_data)
    retain_ratio = selected / total

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"Client {client_index}: Total = {total} Selected = {selected} Remain = {total - selected} 保留比例 = {retain_ratio:.2%}\n")
        if selected == 0:
            f.write(f"Client {client_index}: Warning, all data is filtered out and not retained.\n")
        if selected == total:
            f.write(f"Client {client_index}: Warning, all data is filtered out and not retained.\n")

    keys_to_keep = ["instruction", "context", "response", "category", "personalization_score"]

    new_selected_data = [{k: record.get(k) for k in keys_to_keep} for record in selected_data]
    new_remain_data = [{k: record.get(k) for k in keys_to_keep} for record in remain_data]
    new_all_data = [{k: record.get(k) for k in keys_to_keep} for record in merged_data]

    all_data_dir = f"data_train/8/global_epoch_{epoch}/all_data/8"
    selected_data_dir = f"data_train/8/global_epoch_{epoch}/selected_data/8"
    remain_data_dir = f"data_train/8/global_epoch_{epoch}/remain_data/8"
    os.makedirs(all_data_dir, exist_ok=True)
    os.makedirs(selected_data_dir, exist_ok=True)
    os.makedirs(remain_data_dir, exist_ok=True)

    all_data_path = os.path.join(all_data_dir, f"local_training_{client_index}.json")
    with open(all_data_path, 'w', encoding='utf-8') as f:
        json.dump(new_all_data, f, indent=4, ensure_ascii=False)

    selected_data_path = os.path.join(selected_data_dir, f"local_training_{client_index}.json")
    with open(selected_data_path, 'w', encoding='utf-8') as f:
        json.dump(new_selected_data, f, indent=4, ensure_ascii=False)

    remain_data_path = os.path.join(remain_data_dir, f"local_training_{client_index}.json")
    with open(remain_data_path, 'w', encoding='utf-8') as f:
        json.dump(new_remain_data, f, indent=4, ensure_ascii=False)

    print(f"Client {client_index} processing completed: Total={total}, Selected={selected}, Ratio={retain_ratio:.2%}")

    selected_data_path = os.path.join(selected_data_dir, f"local_training_{client_index}.json")
    other_data_dir = f"data_train/initialized_data/8/global1/8"
    other_data_path = os.path.join(other_data_dir, f"local_training_{client_index}.json")

    with open(selected_data_path, 'r', encoding='utf-8') as f:
        selected_data = json.load(f)
    with open(other_data_path, 'r', encoding='utf-8') as f:
        other_data = json.load(f)


    merged_datas = other_data + selected_data
    merged_datas.sort(key=lambda x: x.get('personalization_score', 0), reverse=True)

    with open(other_data_path, 'w', encoding='utf-8') as f:
        json.dump(merged_datas, f, indent=4, ensure_ascii=False)

    remain1_data_dir = f"data_train/initialized_data/8/remain1/8"
    remain1_data_path = os.path.join(remain1_data_dir, f"local_training_{client_index}.json")
    with open(remain1_data_path, 'w', encoding='utf-8') as f:
        json.dump(remain_data, f, indent=4, ensure_ascii=False)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"Client {client_index}: Complete data saved at {all_data_path}, truncated data saved at {selected_data_path}, discarded data saved at {remain_data_path}\n\n")
