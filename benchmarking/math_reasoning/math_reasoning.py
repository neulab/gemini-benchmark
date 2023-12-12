# NOTE:
# You can find an updated, more robust and feature-rich implementation
# in Zeno Build
# - Zeno Build: https://github.com/zeno-ml/zeno-build/
# - Implementation: https://github.com/zeno-ml/zeno-build/blob/main/zeno_build/models/providers/openai_utils.py

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
from litellm import Router
sys.path.append('../utils')
from reasoning_utils import * 


os.environ["OPENAI_API_KEY"] = "######"


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
def main(task, model):
    
    with open("gsm8k-cot.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    prompt = data_loaded['doc_to_text']
    
    question_answer_list = []
    
    if task == 'gsm8k':
        test_examples = get_examples_gsm8k("gsm8k", N=10)
    elif task == 'svamp':
        test_examples = get_examples_svamp("svamp", N=10)
    elif task == 'asdiv':
        test_examples = get_examples_svamp("asdiv", N=10)
    elif task == 'mawpsmultiarith':
        test_examples = get_examples_svamp("mawpsmultiarith", N=10)
    
    test_dset = GSMDataset(test_examples)
    test_loader = DataLoader(test_dset, batch_size=8, shuffle=True)

    for idx, (qid, qn, ans) in tqdm(enumerate(test_loader), total=len(test_loader)):
        
        mlist = []
        for q in qn:
            q_prompt = prompt.replace("{{question}}", "{question}").format(question=q)
            
            mlist.append([{"role": "system", "content": "You are a helpful math assistant."},
                          {"role": "user", "content": q_prompt}])
        
        predictions = asyncio.run(
            dispatch_openai_requests(
                messages_list=mlist,
                model=model,
                temperature=0.3,
                max_tokens=512,
                top_p=1.0,
            )
        )

        for i, (response, qi, q, a) in enumerate(zip(predictions, qid, qn, ans)):
            al = {'qid': qi.item(),
                  'prompt': prompt.replace("{{question}}", "{question}").format(question=q),
                  'question': q,
                  'generated_text': response.choices[0].message.content}
            
            if task == 'gsm8k': 
                only_a = extract_answer(a)
                al['answer'] = only_a
                al['answer_text'] = a
            else:
                al['answer'] = a
                
            question_answer_list.append(al)
            
    if not os.path.exists(f'/home/sakter/courses/Fall_2023/openai/outputs/{task}'):
        os.makedirs(f'/home/sakter/courses/Fall_2023/openai/outputs/{task}')    
    
    question_answer_list = return_predicted_answer(question_answer_list)
    
    with open(f'/home/sakter/courses/Fall_2023/openai/outputs/{task}/output.jsonl', 'w') as f:
        for d in question_answer_list:
            json.dump(d, f)
            f.write('\n')
        
    return


if __name__ == "__main__":
    main()
