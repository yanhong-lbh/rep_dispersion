import json
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    AutoConfig
)
from config import config
import random
from peft import PeftModel



def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_dataset_text(dataset_name, cache_dir):
    if dataset_name == "python-github-code":
        raw_dataset = load_dataset('tomekkorbak/python-github-code', cache_dir=cache_dir)['train']
        all_text = [r['text'] for r in raw_dataset]
    elif dataset_name == 'cnn_dailymail':
        raw_dataset = load_dataset('cnn_dailymail', '3.0.0', cache_dir=cache_dir)['train']
        all_text = [r['article'] for r in raw_dataset]
    elif dataset_name == 'wikitext':
        raw_dataset = load_dataset('wikitext', 'wikitext-103-v1', cache_dir=cache_dir)['train']
        all_text = ' '.join([raw_dataset[i]['text'] for i in range(len(raw_dataset))])
    elif dataset_name == 'cc_news':
        raw_dataset = load_dataset('vblagoje/cc_news', cache_dir=cache_dir)['train']
        all_text = [r['title']+': '+r['text'] for r in raw_dataset]
    elif dataset_name == 'med_terms':
        raw_dataset = load_dataset('gamino/wiki_medical_terms', cache_dir=cache_dir)['train']
        all_text = [r['page_title']+': '+r['page_text'] for r in raw_dataset]
    elif dataset_name == 'pile_of_law_subset':
        all_text = []
        subsets = [
            'hhs_alj_opinions', 'cfpb_creditcard_contracts', 'irs_legal_advice_memos',
            'constitutions', 'acus_reports', 'doj_guidance_documents', 'eoir',
            'medicaid_policy_guidance','olc_memos', 'cc_casebooks', 'federal_register'
        ]
        for law_subset_name in subsets:
            raw_dataset = load_dataset(
                'pile-of-law/pile-of-law',
                law_subset_name,
                trust_remote_code=True, 
                cache_dir=cache_dir
            )['train']
            all_text.extend([r['text'] for r in raw_dataset])
    elif dataset_name == 'pubmed-summarization':
        raw_dataset = load_dataset('ccdv/pubmed-summarization', 'document', cache_dir=cache_dir)['train']
        all_text = [r['article'] for r in raw_dataset]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return all_text


def load_model_and_tokenizer(model_name, lora_path=None, full_ft_path=None, cache_dir=None, load_model_from_scratch=False):

    def load_model_from_scratch(hf_model_id, is_gpt2=False):
        """
        Creates a config (from_pretrained for convenience so it matches 
        the shape of the real model) but does NOT load actual weights.
        Then initializes a fresh model from that config.
        For GPT2, we can also handle a direct GPT2Config if desired.
        """
        if is_gpt2:
            # Use GPT2Config directly if you prefer:
            config = GPT2Config()
            model_local = GPT2LMHeadModel(config)
        else:
            # For generic models (like LLaMA, Mistral, etc.), load config from HF repos:
            config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True, cache_dir=cache_dir)
            model_local = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        return model_local

    if full_ft_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(full_ft_path, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(full_ft_path, 
                                                     torch_dtype=torch.bfloat16, 
                                                     cache_dir=cache_dir)
    else:
        if model_name in ['gpt2', 'gpt2_wiki']:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
            model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir)
            if model_name == 'gpt2_wiki':
                checkpoint = torch.load(config.gpt2_wiki_ft_path)
                # Adjust key names if needed
                checkpoint['state_dict'] = {
                    key.replace("model.", ""): value
                    for key, value in checkpoint['state_dict'].items()
                }
                model.load_state_dict(checkpoint['state_dict'])

        elif model_name in ['gpt2-xl', 'gpt2-xl_wiki']:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl', cache_dir=cache_dir)
            model = GPT2LMHeadModel.from_pretrained('gpt2-xl', cache_dir=cache_dir)
            if model_name == 'gpt2-xl_wiki':
                checkpoint = torch.load(config.gpt2_xl_wiki_ft_path)
                checkpoint['state_dict'] = {
                    key.replace("model.", ""): value
                    for key, value in checkpoint['state_dict'].items()
                }
                model.load_state_dict(checkpoint['state_dict'])

        elif model_name in ['llama-3-8b']:
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16, cache_dir=cache_dir)
        elif model_name in ['llama-3.2-1b']:
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B', cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B', torch_dtype=torch.bfloat16, cache_dir=cache_dir)
            if load_model_from_scratch:
                model = load_model_from_scratch('meta-llama/Llama-3.2-1B')
            else:
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B', cache_dir=cache_dir)
        elif model_name in ['llama-3.2-3b']:
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B', cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B', torch_dtype=torch.bfloat16, cache_dir=cache_dir)   
        elif model_name in ['llama-3.1-8b']:
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B', cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B', torch_dtype=torch.bfloat16, cache_dir=cache_dir)   
        elif model_name in ['phi-2']:
            tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', cache_dir=cache_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2', torch_dtype="auto", trust_remote_code=True, cache_dir=cache_dir)  
        elif model_name == 'gemma-2-2b':
            tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b', cache_dir=cache_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b',  cache_dir=cache_dir)  
        elif model_name in ['gemma-2-9b']:
            tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-9b', cache_dir=cache_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained('google/gemma-2-9b',  cache_dir=cache_dir)
        elif model_name in ['mistral-7b-v0.1']:
            tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1', cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1', cache_dir=cache_dir)
        elif model_name in ['qwen2.5-0.5b']:
            tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', cache_dir=cache_dir)
        elif model_name in ['qwen2.5-1.5b']:
            tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B', cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B', cache_dir=cache_dir)
        elif model_name in ['qwen2.5-3b']:
            tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B', cache_dir=cache_dir)
        elif model_name in ['qwen2.5-7b']:
            tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B', cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B', cache_dir=cache_dir)
        elif model_name == 'neulab_distilgpt2':
            tokenizer = AutoTokenizer.from_pretrained('neulab/distilgpt2-finetuned-wikitext103', cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained('neulab/distilgpt2-finetuned-wikitext103', cache_dir=cache_dir)
        elif model_name == 'neulab_gpt2-small':
            tokenizer = AutoTokenizer.from_pretrained('neulab/gpt2-finetuned-wikitext103', cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained('neulab/gpt2-finetuned-wikitext103', cache_dir=cache_dir)
        elif model_name == 'neulab_gpt2-medium':
            tokenizer = AutoTokenizer.from_pretrained('neulab/gpt2-med-finetuned-wikitext103', cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained('neulab/gpt2-med-finetuned-wikitext103', cache_dir=cache_dir)
        elif model_name == 'neulab_gpt2-large':
            tokenizer = AutoTokenizer.from_pretrained('neulab/gpt2-large-finetuned-wikitext103', cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained('neulab/gpt2-large-finetuned-wikitext103', cache_dir=cache_dir)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    if lora_path is not None:
        model = PeftModel.from_pretrained(
            model,
            lora_path,            
            torch_dtype=torch.bfloat16,
            device_map="auto" 
        )

    return tokenizer, model

