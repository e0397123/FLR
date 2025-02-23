#!/bin/bash

mkdir -p llama3_outputs/
export dimension="understanding"
for fname in "shp" "helpful_base" "helpful_online" "helpful_rejection" "alpaca_farm" "beavertails_holistic"; do
    
    /home/dialogue/miniconda3/envs/llama_factory/bin/python run_llama3.py \
        --model "meta-llama/Llama-3-8B-Instruct" \
        --input_fname "data/${fname}_accept.json" \
        --dimension ${dimension} \
        --output_fname "llama3_outputs/${fname}_accept_${dimension}.jsonl";

done
