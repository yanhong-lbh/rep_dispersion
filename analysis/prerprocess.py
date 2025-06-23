
import os
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
from config import config
import torch
from tqdm import tqdm
import numpy as np
import random
from utils import (
    save_json, load_json, 
    load_dataset_text, load_model_and_tokenizer
)

def calculate_token_probabilities(sentence, tokenizer, model, model_name, chunk_id, layer_idx=-1):
    inputs = sentence.to('cuda')

    # Adjust input shape if not GPT
    if not model_name.startswith('gpt'):
        inputs = inputs.unsqueeze(0)

    labels = inputs.clone()

    if config.last_token_ppl_only:
        # Create a label tensor that is -100 everywhere except the last token.
        labels = inputs.clone()
        labels[:, :-1] = -100

    with torch.no_grad():
        # First, compute perplexity
        outputs = model(inputs, labels=labels)
        ppl = torch.exp(outputs.loss).item()

    if config.ppl_only or config.last_token_ppl_only:
        return ppl

    # If we need hidden states:
    with torch.no_grad():
        hidden_output = model(inputs, output_hidden_states=True)
    # hidden_output.hidden_states is a tuple of (layer_count, batch_size, seq_len, hidden_dim)
    # Default layer_idx = -1 means the last layer
    selected_hidden_states = hidden_output.hidden_states[layer_idx].to('cpu').float().numpy()
    return selected_hidden_states, ppl

def sample_chunks(tokenized, chunk_len, n_chunks, random_sample=False, random_seed=42):
    if random_sample:
        total_chunks = len(tokenized) // chunk_len
        random.seed(random_seed)
        sampled_indices = random.sample(range(total_chunks), n_chunks)
        chunks = [tokenized[i * chunk_len:(i + 1) * chunk_len] for i in sampled_indices]
    else:
        chunks = [tokenized[i:i + chunk_len] for i in range(0, n_chunks * chunk_len, chunk_len)]
    return chunks

lora_path_dict = {
    'llama-3.2-1b': 'lora_tuned_models/llama3.2-1b/lora/pretrain/checkpoint-381'
}

full_ft_path_dict = {
    'llama-3.2-1b': 'full_ft_models/yh_wiki_llama32_1b_full_sft/checkpoint-4100'
}
def main():

    cache_dir = config.cache_dir
    data_dir = config.data_dir
    dataset_name = config.dataset_name
    model_name = config.model_name
    chunk_len = config.chunk_len
    n_chunks = config.n_chunks
    random_sample = config.random_sample
    random_seed = config.random_seed
    ppl_only = config.ppl_only
    k = config.k
    selected_indices_path = config.selected_indices_path
    last_token_ppl_only = config.last_token_ppl_only
    layer_idx = config.layer_idx
    token_chosen = config.token_chosen
    use_lora_finetuned_model = config.use_lora_finetuned_model
    use_full_finetuned_model = config.use_full_finetuned_model
    ft_dataset = config.ft_dataset

    lora_path = lora_path_dict[model_name] if use_lora_finetuned_model else None
    full_ft_path = full_ft_path_dict[model_name] if use_full_finetuned_model else None

    os.environ['HF_HOME'] = cache_dir
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    all_text = load_dataset_text(dataset_name, cache_dir)

    tokenizer, model = load_model_and_tokenizer(
        model_name=model_name,
        lora_path=lora_path,
        full_ft_path=full_ft_path,
        cache_dir=cache_dir
    )

    selected_indices = None
    if selected_indices_path:
        # Load the selected indices from the given JSON file
        uniform_data = load_json(selected_indices_path)
        # Check if the current model is in the JSON
        if model_name in uniform_data:
            # Extract all indices from the groups
            groups = uniform_data[model_name]["groups"]
            selected_indices = []
            for g in groups:
                selected_indices.extend(g["indices"])
            selected_indices = set(selected_indices)
        else:
            print(f"No entry for model {config.model_name} in {args.selected_indices_path}. Proceeding without filtering.")

    tokenized_corpus_path = f'cache/{dataset_name}_{model_name}.json'

    if not os.path.exists(tokenized_corpus_path):
        if type(all_text).__name__ == 'list':
            tokenized = tokenizer.batch_encode_plus(all_text)['input_ids']
            save_json(tokenized, tokenized_corpus_path)
        else:
            tokenized = tokenizer.encode(all_text)

            save_json(tokenized, tokenized_corpus_path)
    else:
        tokenized = load_json(tokenized_corpus_path)

    all_chunks = []


    if isinstance(tokenized[0], list):
        # multiple documents
        if dataset_name == 'cc_news':
            for t in tokenized:
                if len(t) > 512:
                    # Keep the first 512 tokens as one chunk
                    all_chunks.append(t[:512])
        elif dataset_name in [
            'med_terms', 'python-github-code', 'cnn_dailymail',
            'pile_of_law_subset', 'pubmed-summarization'
        ]:
            for t in tokenized:
                for start_idx in range(0, len(t), chunk_len):
                    chunk = t[start_idx : start_idx + chunk_len]
                    if len(chunk) > 0:
                        all_chunks.append(chunk)
        else:
            for t in tokenized:
                for start_idx in range(0, len(t), chunk_len):
                    chunk = t[start_idx : start_idx + chunk_len]
                    if len(chunk) > 0:
                        all_chunks.append(chunk)
    else:
        # single list of tokens
        for start_idx in range(0, len(tokenized), chunk_len):
            chunk = tokenized[start_idx : start_idx + chunk_len]
            if len(chunk) > 0:
                all_chunks.append(chunk)

    print(len(all_chunks))

    # among the filtered chunks, pick exactly n_chunks
    if len(all_chunks) == 0:
        print("Warning: No chunks available!")
        chunks = []
    else:
        if len(all_chunks) <= n_chunks:
            chunks = all_chunks
        else:
            if random_sample:
                random.seed(random_seed)
                sampled_indices = random.sample(range(len(all_chunks)), n_chunks)
                chunks = [all_chunks[i] for i in sampled_indices]
            else:
                chunks = all_chunks[:n_chunks]

    prefix = f"{data_dir}/{dataset_name}_{model_name}_{chunk_len}"

    if random_sample:
        prefix += f"_{n_chunks}_random_{random_seed}"
    
    if use_lora_finetuned_model:
        prefix += f"_lora_ft_{ft_dataset}"
    elif use_full_finetuned_model:
        prefix += f"_full_ft_{ft_dataset}"

    os.makedirs(prefix, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for index in tqdm(range(len(chunks))):

        if selected_indices is not None and index not in selected_indices:
            continue

        hidden_states_path = f"{prefix}/hidden_states_{layer_idx}_{index}.npy"
        #token_probs_path = f"{prefix}/token_probs_{index}.json"
        if last_token_ppl_only:
            ppl_path = f"{prefix}/last_token_ppl_{index}.json"
        else:
            ppl_path = f"{prefix}/ppl_{index}.json"

        if os.path.exists(hidden_states_path) and os.path.exists(ppl_path):
            continue
        
        if os.path.exists(ppl_path) and (config.ppl_only or config.last_token_ppl_only):
            continue

        if config.ppl_only or config.last_token_ppl_only:
            ppl = calculate_token_probabilities(torch.tensor(chunks[index]), tokenizer, model, model_name, index)
            save_json(ppl, ppl_path)
        else:
            hidden_states, ppl = calculate_token_probabilities(torch.tensor(chunks[index]), tokenizer, model, model_name, index)
            np.save(hidden_states_path, hidden_states)


if __name__ == '__main__':
    main()