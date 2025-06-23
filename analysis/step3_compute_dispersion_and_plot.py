import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import subprocess

# --- Configuration ---
MODELS = ["llama-3.2-1b"] # Must match models from Step 2
DATASET = 'wikitext'
CHUNK_LEN = 512
N_CHUNKS = 100000
LAYER_IDX = -1 # Final layer

# Input file from Step 2
SAMPLED_DATA_FILE = f"all_models_grouped_ppl_uniform_{CHUNK_LEN}_{DATASET}.json"

# --- Part 1: Get Hidden States for Sampled Chunks ---
print("--- Starting Step 3, Part 1: Calculating Hidden States for Sampled Chunks ---")
for model_name in MODELS:
    print(f"Running preprocess.py for {model_name} to get hidden states...")
    # This command is equivalent to the 'send_jobs_k_hs_only.sh' script's purpose
    subprocess.run([
        'python', 'preprocess.py',
        '--dataset_name', DATASET,
        '--model_name', model_name,
        '--chunk_len', str(CHUNK_LEN),
        '--n_chunks', str(N_CHUNKS),
        '--random_sample',
        '--random_seed', '42',
        '--layer_idx', str(LAYER_IDX),
        '--selected_indices_path', SAMPLED_DATA_FILE
    ])

# --- Part 2: Compute Dispersion from Hidden States ---
print("\n--- Starting Step 3, Part 2: Computing Representation Dispersion ---")
with open(SAMPLED_DATA_FILE, 'r') as f:
    sampled_data = json.load(f)

all_results = {}
for model_name in MODELS:
    print(f"Processing model: {model_name}")
    model_dir = f"data/{DATASET}_{model_name}_{CHUNK_LEN}_{N_CHUNKS}_random_42"
    model_groups = sampled_data.get(model_name, {}).get("groups", [])
    
    if not model_groups:
        print(f"No sampled groups found for {model_name}. Skipping.")
        continue

    model_results = []
    for group in model_groups:
        mean_ppl = group['mean_ppl']
        indices = group['indices']
        
        group_distances = []
        for index in indices:
            hs_path = f"{model_dir}/hidden_states_{LAYER_IDX}_{index}.npy"
            if not os.path.exists(hs_path):
                continue
            
            # Load hidden states and compute dispersion
            hidden_states = torch.from_numpy(np.load(hs_path)).squeeze(0) # Shape: (seq_len, hidden_dim)
            if hidden_states.shape[0] > 1:
                # Normalize to unit vectors
                normalized_hs = F.normalize(hidden_states.float(), p=2, dim=1)
                # Calculate pairwise cosine similarity, convert to distance
                cosine_sim = torch.matmul(normalized_hs, normalized_hs.t())
                # Get upper triangle, excluding diagonal, and average
                dist = 1 - cosine_sim[torch.triu(torch.ones(cosine_sim.shape), diagonal=1).bool()].mean().item()
                group_distances.append(dist)

        if group_distances:
            avg_cosine_distance = np.mean(group_distances)
            model_results.append({
                'mean_ppl': mean_ppl,
                'avg_cosine_distance': avg_cosine_distance
            })
    all_results[model_name] = model_results

# --- Part 3: Filter Outliers and Plot (from remove_outliers_then_plot.py) ---
print("\n--- Starting Step 3, Part 3: Filtering and Plotting Results ---")
os.makedirs('plots_postprocessed', exist_ok=True)

for model_name, results in all_results.items():
    plot_file = f'plots_postprocessed/{DATASET}_{model_name}_plot.png'
    
    # Define an outlier threshold for PPL
    ppl_threshold = 25 # Example threshold, adjust as needed per dataset
    
    filtered_results = [g for g in results if g['mean_ppl'] < ppl_threshold]
    
    if not filtered_results:
        print(f"No data left for {model_name} after filtering. Skipping plot.")
        continue

    mean_ppls = [g['mean_ppl'] for g in filtered_results]
    avg_dists = [g['avg_cosine_distance'] for g in filtered_results]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_ppls, avg_dists, alpha=0.7, label=f'Layer {LAYER_IDX}')
    plt.title(f'Representation Dispersion vs. Perplexity\nModel: {model_name} | Dataset: {DATASET}')
    plt.xlabel("Mean Perplexity (PPL)")
    plt.ylabel("Average Pairwise Cosine Distance (Dispersion)")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Plot saved to {plot_file}")

print("--- Step 3 Complete ---")