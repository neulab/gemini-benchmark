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
import litellm
import time
# litellm.set_verbose = True


os.environ["OPENAI_API_KEY"] = "##"
os.environ["TOGETHERAI_API_KEY"] = "##"
os.environ["HF_TOKEN"] = "##"

def set_router(model):
    
    if model == 'mixtral':
        model="together_ai/DiscoResearch/DiscoLM-mixtral-8x7b-v2"
        # model = "huggingface/DiscoResearch/DiscoLM-mixtral-8x7b-v2"
    
    model_list = [{
        "model_name": model, 
        "litellm_params": {
            "model": model,
        }
    }]

    router = Router(model_list=model_list)
    return router

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
PATTERN = r"(?:\(|\s)([A-Z])\.?(?:\)|\s|$)"


TASKS = [
    "boolean_expressions",
    "geometric_shapes",
    "reasoning_about_colored_objects",
    "tracking_shuffled_objects_seven_objects",
    "movie_recommendation",
    "logical_deduction_five_objects",
    "salient_translation_error_detection",
    "logical_deduction_three_objects",
    "temporal_sequences",
    "disambiguation_qa",
    "object_counting",
    "causal_judgement",
    "ruin_names",
    "formal_fallacies",
    "penguins_in_a_table",
    "web_of_lies",
    "sports_understanding",
    "navigate",
    "date_understanding",
    "logical_deduction_seven_objects",
    "snarks",
    "tracking_shuffled_objects_five_objects",
    "word_sorting",
    "dyck_languages",
    "multistep_arithmetic_two",
    "hyperbaton",
    "tracking_shuffled_objects_three_objects"
]


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
    router: Router,
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
    if model == 'mixtral':
        model="together_ai/DiscoResearch/DiscoLM-mixtral-8x7b-v2"
        # model = "huggingface/DiscoResearch/DiscoLM-mixtral-8x7b-v2"
        # top_p = 0.9
        
    async_responses = [
        await router.acompletion(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        for x in messages_list
    ]

    return async_responses


def get_examples_gsm8k(split, lr=0, rr=-1):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    if rr == -1:
        examples = examples[lr:len(examples)]
        
    print(f"{len(examples)} {split} examples")
    return examples

def get_examples_svamp(split, lr=0, rr=-1):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["input"] + "\n")
        ex.update(answer=str(ex["target"]))
        del ex['input']
        del ex["target"]

    if rr == -1:
        examples = examples[lr:len(examples)]
        
    print(f"{len(examples)} {split} examples")
    return examples


def get_examples(split, model, lr=0, rr=-1):
    path = os.path.join("data/bbh/", f"{split}.jsonl")
    
    examples = read_jsonl(path)
        
    for ex in examples:
        ex.update(question=ex["input"]+ "\n")
        ex.update(answer=ex["target"])
        del ex['input']
        del ex["target"]
        
    if rr == -1:
        examples = examples[lr:len(examples)]

    new_examples = []
    
    for ex in examples:
        if os.path.exists(f'gemini-benchmark/outputs/bbh/{model}/{split}/all_jsons'):
            qid = ex['qid']
            result_path = f'gemini-benchmark/outputs/bbh/{model}/{split}/all_jsons/{qid}.json'
            if os.path.isfile(result_path):
                continue
        new_examples.append(ex)
    
    print(f"{len(new_examples)} {split} examples")
    return new_examples


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
            
        soln = out['generated_text']
        exact = float(out['answer'])
        prob_ans = re.findall(r"[-+]?(?:[0-9,]*\.*\d+)", soln)
        prob_ans = [float(x.replace(',', '')) for x in prob_ans]
        if len(prob_ans) > 0 and exact == prob_ans[-1]:
            out['predict_last'] = out['answer']
            out['is_correct_last'] = 1
        else:
            if len(prob_ans) > 0: out['predict_last'] = str(prob_ans[-1])
            else: out['predict_last'] = "-10000000000"
            out['is_correct_last'] = 0
            
    return question_answer_list




def get_answer(question_answer_list):
    for out in question_answer_list:
        soln = out['generated_text']
        exact = out['answer']
        prob_ans = re.findall(r"(?<=the answer is )(.*)(?=.)", soln)
        if len(prob_ans) > 0 and exact == prob_ans[-1]:
            out['predict'] = out['answer']
            out['is_correct'] = 1
        else:
            if len(prob_ans) > 0: out['predict'] = str(prob_ans[-1])
            else: out['predict'] = "-10000000000"
            out['is_correct'] = 0
            
        
        if exact.startswith('(') and exact.endswith(')'):
            prob_ans = re.findall(PATTERN, out['generated_text'])
            prob_ans = ['('+x+')' for x in prob_ans]
            if len(prob_ans) > 0 and exact == prob_ans[-1]:
                out['predict_last'] = out['answer']
                out['is_correct_last'] = 1
            else:
                if len(prob_ans) > 0: out['predict_last'] = str(prob_ans[-1])
                else: out['predict_last'] = "-10000000000"
                out['is_correct_last'] = 0
                
    for out in question_answer_list:
        if 'is_correct_last' not in out:
            out['is_correct_last'] = out['is_correct']
            out['predict_last'] = out['predict']
    
    return question_answer_list 