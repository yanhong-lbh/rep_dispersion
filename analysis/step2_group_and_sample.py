import json
import os
import numpy as np
import bisect

# --- Configuration ---
MODELS = ["llama-3.2-1b"] # Add other models as needed
DATASET = 'wikitext'
CHUNK_LEN = 512
N_CHUNKS = 100000

# Input file pattern from Step 1 and output file for this script
PPL_GROUPED_FILE = f"all_models_grouped_ppl_{CHUNK_LEN}_{DATASET}.json"
UNIFORM_SAMPLED_FILE = f"all_models_grouped_ppl_uniform_{CHUNK_LEN}_{DATASET}.json"

# --- Part 1: Grouping (from group_ppl.py) ---
print("--- Starting Step 2, Part 1: Grouping PPL scores ---")
results = {}
for model in MODELS:
    model_dir = f"data/{DATASET}_{model}_{CHUNK_LEN}_{N_CHUNKS}_random_42"
    
    perplexity_data = []
    if not os.path.exists(model_dir):
        print(f"Warning: Directory not found: {model_dir}. Skipping model {model}.")
        continue

    for i in range(N_CHUNKS):
        filename = os.path.join(model_dir, f"ppl_{i}.json")
        if not os.path.isfile(filename):
            continue
        try:
            with open(filename, "r") as f:
                ppl_value = json.load(f)
                perplexity_data.append((i, ppl_value))
        except (json.JSONDecodeError, ValueError):
            continue
            
    print(f"Model: {model}, Loaded: {len(perplexity_data)} PPL files.")
    
    perplexity_data.sort(key=lambda x: x[1])
    
    groups = []
    group_size = 100
    for i in range(0, len(perplexity_data), group_size):
        chunk = perplexity_data[i:i+group_size]
        if len(chunk) < group_size:
            break # Discard incomplete groups
        
        indices = [x[0] for x in chunk]
        ppls = [x[1] for x in chunk]
        groups.append({
            "mean_ppl": float(np.mean(ppls)),
            "indices": indices
        })
    results[model] = {"groups": groups}

with open(PPL_GROUPED_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"Grouped PPL data saved to {PPL_GROUPED_FILE}")

# --- Part 2: Uniform Sampling (from sample.py) ---
print("\n--- Starting Step 2, Part 2: Uniform Sampling ---")
with open(PPL_GROUPED_FILE, "r") as f:
    results = json.load(f)

uniform_results = {}
for model, model_data in results.items():
    groups = model_data["groups"]
    groups_sorted = sorted(groups, key=lambda x: x["mean_ppl"])
    
    mean_ppls = [g["mean_ppl"] for g in groups_sorted]
    
    if len(groups_sorted) <= 100:
        selected_groups = groups_sorted
    else:
        min_ppl, max_ppl = mean_ppls[0], mean_ppls[-1]
        target_values = np.linspace(min_ppl, max_ppl, 100)
        
        selected_indices = set()
        for val in target_values:
            pos = bisect.bisect_left(mean_ppls, val)
            if pos == len(mean_ppls): pos -= 1
            
            if pos > 0 and abs(mean_ppls[pos-1] - val) < abs(mean_ppls[pos] - val):
                selected_indices.add(pos - 1)
            else:
                selected_indices.add(pos)
        
        selected_groups = [groups_sorted[i] for i in sorted(list(selected_indices))]

    uniform_results[model] = {"groups": selected_groups}

with open(UNIFORM_SAMPLED_FILE, "w") as f:
    json.dump(uniform_results, f, indent=2)

print(f"Uniformly sampled data saved to {UNIFORM_SAMPLED_FILE}")
print("--- Step 2 Complete ---")