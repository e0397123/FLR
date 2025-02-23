import torch
import argparse
import json
import logging
import os
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *

positive_feedback_dict = {"understanding": understanding_pos, 
                           "relevance": relevance_pos,
                           "correctness": correctness_pos,
                           "engagingness": engagingness_pos,
                           "informativeness": informativeness_pos,
                           "following": following_pos}

negative_feedback_dict = {"understanding": understanding_neg, 
                           "relevance": relevance_neg,
                           "correctness": correctness_neg,
                           "engagingness": engagingness_neg,
                           "informativeness": informativeness_neg,
                           "following": following_neg}


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--input_fname", type=str)
    parser.add_argument("--output_fname", type=str)
    parser.add_argument("--dimension", type=str)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)

    return parser.parse_args()
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def convert_to_messages(d):
    messages = []
    if len(d) % 2 == 0:
        new_d = d[1:]
    else:
        new_d = d
    for idx, utt in enumerate(new_d):
        if idx % 2 == 0:
            messages.append({'role': 'user', 'content': utt})
        else:
            messages.append({'role': 'assistant', 'content': utt})
    return messages

def get_outputs_withprobs(model, tokenizer, batch, positive_statements, negative_statements):
    texts = [x['conversation'] for x in batch]
    positive_all_probs = []
    for feedback in positive_statements:
        feedback_ids = tokenizer(feedback, return_tensors="pt").input_ids
        feedback_len = feedback_ids.shape[1]
        messages = convert_to_messages(texts[0] + [feedback])
        label_ids = tokenizer.apply_chat_template(messages, 
                                                  padding=False,
                                                  truncation=True, 
                                                  max_length=None,
                                                  return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model(
                input_ids=label_ids,
                labels=label_ids,  # skip BOS token
                return_dict=True
            )
        logits = output['logits'][0, -feedback_len:, :]
        target_side_probs = logits[0, feedback_ids[0]].mean().item()
        positive_all_probs.append(target_side_probs)

    negative_all_probs = []
    for feedback in negative_statements:
        feedback_ids = tokenizer(feedback, return_tensors="pt").input_ids
        feedback_len = feedback_ids.shape[1]
        messages = convert_to_messages(texts[0] + [feedback])
        label_ids = tokenizer.apply_chat_template(messages, 
                                                  padding=False,
                                                  truncation=True, 
                                                  max_length=None,
                                                  return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model(
                input_ids=label_ids,
                labels=label_ids,  # skip BOS token
                return_dict=True
            )
        logits = output['logits'][0, -feedback_len:, :]
        target_side_probs = logits[0, feedback_ids[0]].mean().item()
        negative_all_probs.append(target_side_probs)

    return positive_all_probs, negative_all_probs

def run_generation(args, prompts, positive_statements, negative_statements):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=access_token)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, 
                                                 device_map='auto', 
                                                 torch_dtype=torch.float16,
                                                 token=access_token)
    model = model.to(device)
    model.eval()
    
    batch_chunks = chunks(prompts, args.batch_size)
    
    with open(os.path.join(args.output_fname), 'w') as outfile:
        for _, batch in tqdm(enumerate(batch_chunks), total=len(prompts)//args.batch_size):
            pos, neg = get_outputs_withprobs(model, tokenizer, batch, positive_statements, negative_statements)
            for d, _ in enumerate(batch):
                batch[d]['positve'] = pos
                batch[d]['negative'] = neg
                json.dump(batch[d], outfile)
                outfile.write('\n')
                outfile.flush()

if __name__ == "__main__":
    args = read_args()
    print(args)
    random.seed(args.seed)
    
    positive_statements = positive_feedback_dict[args.dimension]
    negative_statements = negative_feedback_dict[args.dimension]
    
    data = json.load(open(args.input_fname))
    
    if type(data[0]) is list:
        prompts = [{'conversation': d} for d in data]
    else:
        prompts = data
    
    run_generation(args, prompts, positive_statements, negative_statements)
