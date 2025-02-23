#!/bin/bash

export llama_model_path="meta-llama/Llama-3-8B-Instruct"
export fname='ultrafeedback_prompts'
export start_idx=0
export end_idx=5000
export output_folder="task111"

mkdir -p ${output_folder}

for num in 8; do
    python3 generate.py \
        --model_type=llama3 \
        --model_name_or_path=${llama_model_path} \
        --num_return_sequences=${num} \
        --temperature=0.7 \
        --fp16 \
        --length=2048 \
        --start_index=${start_idx} \
        --end_index=${end_idx} \
        --prompt_source ${fname}.json \
        --template llama3 \
        --output_path ${output_folder}/${fname}_output_k${num}-${start_idx}-${end_idx}.jsonl;
done
