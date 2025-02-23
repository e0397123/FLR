#!/bin/bash

mkdir -p qwen2_outputs/

for fname in "shp" "helpful_base" "helpful_online" "helpful_rejection" "alpaca_farm" "beavertails_holistic"; do

    /home/dialogue/miniconda3/envs/llama_factory/bin/python run_qwen2_pairwise.py \
        --model "Qwen/Qwen2-7B-Instruct" \
        --input_fname "data/${fname}_reject.json" \
        --output_fname "qwen2_outputs/${fname}_reject.jsonl";

done
