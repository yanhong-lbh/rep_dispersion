#!/bin/bash

# This script demonstrates the command to calculate perplexity (PPL) for data chunks.
# In practice, you would run this command in parallel for all 100,000 chunks.
# The original workflow used a Slurm sbatch script to manage this.

MODEL_NAME="llama-3.2-1b" # Example model
DATASET_NAME="wikitext"
N_CHUNKS=100000
CHUNK_LEN=512

echo "This step requires running preprocess.py for all chunks to get their PPL."
echo "The command is:"

# This is the core command that gets executed for each chunk.
# The original script created a job for each chunk or batches of chunks.
python preprocess.py \
    --dataset_name $DATASET_NAME \
    --model_name $MODEL_NAME \
    --chunk_len $CHUNK_LEN \
    --n_chunks $N_CHUNKS \
    --random_sample \
    --random_seed 42 \
    --ppl_only

echo "After this step, you should have a directory like data/${DATASET_NAME}_${MODEL_NAME}_${CHUNK_LEN}_${N_CHUNKS}_random_42/ containing thousands of 'ppl_*.json' files."