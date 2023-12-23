import openai
import asyncio
from typing import Any
import base64
import requests
from io import BytesIO
from PIL import Image 
import json
import os
import re
import torch as th
from tqdm import tqdm
from openai import AsyncOpenAI
from torch.utils.data import DataLoader
from litellm import acompletion
import yaml
import click
import sys
import time
from litellm import Router
sys.path.append('../utils')
from reasoning_utils import * 
from prompts import * 
import litellm
litellm.vertex_project = "####" # Your Project ID
litellm.vertex_location = "####"  # proj location

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["TOGETHERAI_API_KEY"] = "TOGETHERAI_API_KEY"


class GSMDataset(th.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.ids = [ex["qid"] for ex in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn = self.qns[idx]
        ans = self.ans[idx]
        qid = self.ids[idx]
        
        return qid, qn, ans


@click.command()
@click.option("--task", default='gsm8k', type=str)
@click.option("--model", default='gpt-3.5-turbo', type=str)
@click.option("--lr", default=0, type=int)
@click.option("--rr", default=-1, type=int)
def main(task, model, lr, rr):
    
    # with open("gsm8k-cot.yaml", 'r') as stream:
    #     data_loaded = yaml.safe_load(stream)
    # prompt = data_loaded['doc_to_text']
    
    question_answer_list = []
    
    if task == 'gsm8k':
        test_examples = get_examples_gsm8k("gsm8k", lr=lr, rr=rr)
    elif task == 'svamp':
        test_examples = get_examples_svamp("svamp", lr=lr, rr=rr)
    elif task == 'asdiv':
        test_examples = get_examples_svamp("asdiv", lr=lr, rr=rr)
    elif task == 'mawpsmultiarith':
        test_examples = get_examples_svamp("mawpsmultiarith", lr=lr, rr=rr)
    
    test_dset = GSMDataset(test_examples)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)
    router = set_router(model)

    if not os.path.exists(f'gemini-benchmark/outputs/{task}'):
        os.makedirs(f'gemini-benchmark/outputs/{task}')
    
    if not os.path.exists(f'gemini-benchmark/outputs/{task}/{model}'):
        os.makedirs(f'gemini-benchmark/outputs/{task}/{model}')
        
    if not os.path.exists(f'gemini-benchmark/outputs/{task}/{model}/all_jsons'):
        os.makedirs(f'gemini-benchmark/outputs/{task}/{model}/all_jsons')
    
    
    for idx, (qid, qn, ans) in tqdm(enumerate(test_loader), total=len(test_loader)):
        
        result_path = f'gemini-benchmark/outputs/{task}/{model}/all_jsons/{qid[0]}.json'
        
        if os.path.isfile(result_path):
            continue
        
        predictions = []
        for q, qi in zip(qn, qid):
            # q_prompt = prompt.replace("{{question}}", "{question}").format(question=q)
            q_prompt = (PROMPT + '\n' + TEMPLATE.format(question=q))
            response = get_response(q_prompt)
            predictions.append(response)
        

        for i, (response, qi, q, a) in enumerate(zip(predictions, qid, qn, ans)):
            al = {'qid': qi.item(),
                  # 'prompt': prompt.replace("{{question}}", "{question}").format(question=q),
                  'prompt': (PROMPT + '\n' + TEMPLATE.format(question=q)),
                  'question': q}
            
            if isinstance(response, str):
                al['generated_text'] = response
            else:
                al['generated_text'] = response.choices[0].message.content
            
            if task == 'gsm8k': 
                only_a = extract_answer(a)
                al['answer'] = only_a
                al['answer_text'] = a
            else:
                al['answer'] = a
                
            question_answer_list.append(al)
            
            result_path = f'gemini-benchmark/outputs/{task}/{model}/all_jsons/{qi}.json'
            
            if os.path.isfile(result_path):
                continue
                
            with open(result_path, 'w') as fp:
                json.dump(al, fp)
    
    question_answer_list = return_predicted_answer(question_answer_list)
    
    with open(f'gemini-benchmark/outputs/{task}/{model}/output.jsonl', 'w') as f:
        for d in question_answer_list:
            json.dump(d, f)
            f.write('\n')
        
    return


if __name__ == "__main__":
    main()
