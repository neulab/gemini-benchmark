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
from litellm import Router


os.environ["OPENAI_API_KEY"] = "######"

model_list = [{
    "model_name": "gpt-3.5-turbo", 
    "litellm_params": {
        "model": "gpt-3.5-turbo",
    }
}]

router = Router(model_list=model_list)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


async def dispatch_openai_requests(
    messages_list: list,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float
):
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    
    async_responses = [
        await router.acompletion(
            model="gpt-3.5-turbo",
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
        
    return async_responses


def get_examples_gsm8k(split, N=None):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    if N != None:
        examples = examples[:N]
        
    print(f"{len(examples)} {split} examples")
    return examples

def get_examples_svamp(split, N=None):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["input"] + "\n")
        ex.update(answer=str(ex["target"]))
        del ex['input']
        del ex["target"]

    if N != None:
        examples = examples[:N]
        
    print(f"{len(examples)} {split} examples")
    return examples


def get_examples(split, N=None):
    path = os.path.join("data/bbh/", f"{split}.jsonl")
    
    examples = read_jsonl(path)
        
    for ex in examples:
        ex.update(question=ex["input"]+ "\n")
        ex.update(answer=ex["target"])
        del ex['input']
        del ex["target"]
        
    if N != None:
        examples = examples[:N]

    print(f"{len(examples)} {split} examples")
    return examples



def return_predicted_answer(question_answer_list):
    for out in question_answer_list:
        soln = out['generated_text']
        exact = float(out['answer'])
        if 'The answer is' in soln:
            soln = soln.split('The answer is')[-1]
            prob_ans = re.findall(r"[-+]?(?:[0-9,]*\.*\d+)", soln)
            prob_ans = [float(x.replace(',', '')) for x in prob_ans]
            prob_ans = [float(x) for x in prob_ans]
            if len(prob_ans) > 0 and exact == prob_ans[0]:
                out['predict'] = out['answer']
                out['is_correct'] = 1
            else:
                if len(prob_ans) > 0: out['predict'] = str(prob_ans[0])
                else: out['predict'] = "-10000000000"
                out['is_correct'] = 0
        else:
            out['predict'] = "-10000000000"
            out['is_correct'] = 0
            
    return question_answer_list


def get_answer(question_answer_list):
    for out in question_answer_list:
        soln = out['generated_text']
        exact = out['answer']
        prob_ans = re.findall(r"(?<=the answer is )(.*)(?=.)", soln)
        if len(prob_ans) > 0 and exact == prob_ans[0]:
            out['predict'] = out['answer']
            out['is_correct'] = 1
        else:
            if len(prob_ans) > 0: out['predict'] = str(prob_ans[0])
            else: out['predict'] = "-10000000000"
            out['is_correct'] = 0

    return question_answer_list