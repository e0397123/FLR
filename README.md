# FLR
Repository for AAAI2025 Paper - Aligning Language Models Using Follow-up Likelihood as Reward Signal

## RM Evaluation / Output Demo Data (with Explanations)
Please find the data at [here](https://drive.google.com/drive/folders/15qrklDPHXcNEowHi-RWR7z9WlEtdPMax?usp=sharing); Please note the following:

(1) For pairwise preference benchmarks, A file name containing "accept" indicates the response is preferred while a file name containing "reject" means the response is not preferred. An "accept" file has a one-to-one data entry mapping to its corresponding "reject" file.

(2) The last word in a file name, such as "relevance", or "engagingness", etc, indicates the probabilities for follow-up utterances in those sub-categories of helpfulness. 

(3) For FLASK, MT-Bench-Single-Score, Feedback_Bench, and HelpSteer, the ground-truth overall scores are included in each data entry, and we can conduct correlation analysis for these four benchmarks.

(4) For wildbench_v2, please use the following LMSYS ELO Ratings for model ranking correlation analysis (as of June 7, 2024, copied from [here](https://huggingface.co/spaces/allenai/WildBench)):

`lmsys_elo = {
    'gpt-4o-2024-05-13': 1283.0,
    'gemini-1.5-pro': 1254.0,
    'gpt-4-turbo-2024-04-09': 1249.0,
    'gpt-4-0125-preview': 1239.0,
    'yi-large': 1234.0,
    'claude-3-opus-20240229': 1231.0,
    'Meta-Llama-3-70B-Instruct': 1214.0,
    'gemini-1.5-flash': 1214.0,
    'claude-3-sonnet-20240229': 1188.0,
    'Qwen2-72B-Instruct': 1184.0,
    'reka-core-20240501': 1176.0,
    'claude-3-haiku-20240307': 1170.0,
    'mistral-large-2402': 1158.0,
    'Yi-1.5-34B-Chat': 1155.0,
    'command-r-plus': 1154.0,
    'Qwen1.5-72B-Chat': 1143.0,
    'reka-flash-20240226': 1129.0,
    'Mixtral-8x7B-Instruct-v0.1': 1114.0,
    'Starling-LM-7B-beta': 1114.0,
    'dbrx-instruct@together': 1111.0,
    'gpt-3.5-turbo-0125': 1107.0,
    'command-r': 1107.0,
    'tulu-2-dpo-70b': 1101.0,
    'Mistral-7B-Instruct-v0.2': 1073.0,
    'Llama-2-70b-chat-hf': 1072.0,
    'Nous-Hermes-2-Mixtral-8x7B-DPO': 1047.0,
    'gemma-7b-it': 1047.0,
    'Phi-3-mini-128k-instruct': 1038.0,
    'Llama-2-7b-chat-hf': 1013.0,
    'gemma-2b-it': 978.0
}`
