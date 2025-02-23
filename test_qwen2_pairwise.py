import torch
import argparse
import json
import logging
import os
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)


negative_statements = ["Not really relevant here.", "You’re really confusing", "I don’t understand what you’re saying.", "That’s not really relevant here.", "You are so confusing.", "You’re really boring.", "That’s not very interesting.", "That was a really boring response.", "I am so confused right now.", " What are you trying to say?", "That makes no sense!", "I don’t understand at all!", "You’re not understanding me!", "That’s a very generic response.", "You’re making no sense at all. ", "I don’t want to talk about that!", "What does that even mean?", "That's not helpful.", "Your explanation was vague and hard to follow.", "This information isn't what I was looking for."]
positive_statements = [
    "This is highly relevant here.",
    "You’re very clear and easy to understand.",
    "I completely understand what you’re saying.",
    "This is highly pertinent to our discussion.",
    "You are very clear.",
    "You’re very engaging.",
    "This is very interesting.",
    "That was a really engaging response.",
    "I understand perfectly now.",
    "I see exactly what you’re trying to say.",
    "That makes perfect sense!",
    "I understand completely!",
    "You’re understanding me perfectly!",
    "That’s a very specific response.",
    "You’re making perfect sense.",
    "I definitely want to talk about that!",
    "That makes perfect sense!",
    "That's very helpful.",
    "Your explanation was clear and easy to follow.",
    "This information is exactly what I was looking for."
]

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--input_fname", type=str)
    parser.add_argument("--output_fname", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)

    return parser.parse_args()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def convert_to_messages(d):
  messages = [{"role": "system", "content": "You are a helpful assistant."}]
  for idx, utt in enumerate(d):
    if idx % 2 == 0:
      messages.append({'role': 'user', 'content': utt})
    else:
      messages.append({'role': 'assistant', 'content': utt})
  return messages


def get_outputs_withprobs(model, tokenizer, batch):
    texts = [x['conversation'] for x in batch]
    positive_all_probs = []
    for feedback in positive_statements:
        feedback_ids = tokenizer(feedback, return_tensors="pt").input_ids
        feedback_len = feedback_ids.shape[1]
        messages = convert_to_messages(texts[0] + [feedback])
        label_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
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
        label_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
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
  

def run_generation(args, prompts):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
  
    data_json_lines = [{'conversation': d} for d in prompts]
    batch_chunks = chunks(data_json_lines, args.batch_size)
    
    with open(os.path.join(args.output_fname), 'w') as outfile:
        for b, batch in tqdm(enumerate(batch_chunks), total=len(data_json_lines)//args.batch_size):
            pos, neg = get_outputs_withprobs(model, tokenizer, batch)
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
    
    data = json.load(open(args.input_fname))
    run_generation(args, data)
