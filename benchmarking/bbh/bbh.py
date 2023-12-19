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
sys.path.append('../utils')
from reasoning_utils import * 
import litellm


os.environ["OPENAI_API_KEY"] = "##"
os.environ["TOGETHERAI_API_KEY"] = "##"
os.environ["HF_TOKEN"] = "##"

class BBHDataset(th.utils.data.Dataset):
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
@click.option("--task", default='object_counting', type=str)
@click.option("--model", default='gpt-3.5-turbo', type=str)
@click.option("--lr", default=0, type=int)
@click.option("--rr", default=-1, type=int)
def main(task, model, lr, rr):
    
    if task == 'all_tasks':
        all_tasks = TASKS
    else:
        all_tasks = [task]
    
    
    results_all_tasks = []
    router = set_router(model)
    
    for task in all_tasks:
        print('Starting task: ', task)
    
        with open(f"lm-evaluation-harness/lm_eval/tasks/bbh/cot_fewshot/{task}.yaml", 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        prompt = data_loaded['doc_to_text']

        question_answer_list = []

        test_examples = get_examples(task, model, lr=lr, rr=rr)

        test_dset = BBHDataset(test_examples)
        test_loader = DataLoader(test_dset, batch_size=8, shuffle=False)
        
        
        if not os.path.exists(f'gemini-benchmark/outputs/bbh/{model}'):
            os.makedirs(f'gemini-benchmark/outputs/bbh/{model}')
        
        if not os.path.exists(f'gemini-benchmark/outputs/bbh/{model}/{task}'):
            os.makedirs(f'gemini-benchmark/outputs/bbh/{model}/{task}')
            
        if not os.path.exists(f'gemini-benchmark/outputs/bbh/{model}/{task}/all_jsons'):
            os.makedirs(f'gemini-benchmark/outputs/bbh/{model}/{task}/all_jsons')
            
        
        for idx, (qid, qn, ans) in tqdm(enumerate(test_loader), total=len(test_loader)):

            mlist = []
            for q in qn:
                if task == 'dyck_languages': q_prompt = prompt.replace("{{input}}", q)
                else: q_prompt = prompt.replace("{{input}}", "{input}").format(input=q)
                 

                mlist.append([{"role": "system", "content": "Follow the given examples and answer the question."},
                              {"role": "user", "content": q_prompt}])

            predictions = asyncio.run(
                dispatch_openai_requests(
                    router=router,
                    messages_list=mlist,
                    model=model,
                    temperature=0.0,
                    max_tokens=512,
                    top_p=1.0,
                )
            )

            for i, (response, qi, q, a) in enumerate(zip(predictions, qid, qn, ans)):
                if task == 'dyck_languages': q_prompt = prompt.replace("{{input}}", q)
                else: q_prompt = prompt.replace("{{input}}", "{input}").format(input=q)
                al = {'task': task,
                     'qid': qi.item(),
                     'prompt': q_prompt,
                     'question': q,
                     'answer': a,
                     'generated_text': response.choices[0].message.content}
                question_answer_list.append(al)
                
                result_path = f'gemini-benchmark/outputs/bbh/{model}/{task}/all_jsons/{qi}.json'
            
                if os.path.isfile(result_path):
                    continue

                with open(result_path, 'w') as fp:
                    json.dump(al, fp)
                    
        question_answer_list = get_answer(question_answer_list)
        results_all_tasks.extend(question_answer_list)

        with open(f'gemini-benchmark/outputs/bbh/{model}/{task}/output.jsonl', 'w') as f:
            for d in question_answer_list:
                json.dump(d, f)
                f.write('\n')
        
    return


if __name__ == "__main__":
    main()
