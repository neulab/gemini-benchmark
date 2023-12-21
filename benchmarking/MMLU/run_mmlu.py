from litellm import acompletion
from tqdm import tqdm
from utils import *
import pandas as pd
import asyncio
import litellm
import json

# parse arguments
import argparse
import os


async def get_response(prompt: str, model: str):
    if "gemini" in model:
        response = await acompletion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Follow the given examples and answer the question.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ],
        )
    else:
        response = await acompletion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Follow the given examples and answer the question.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
    return response


def main(args, tasks=TASKS):
    if "gpt" in args.model_name:
        # gpt evaluation
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    elif "gemini" in args.model_name:
        # gemini evaluation
        litellm.vertex_project = ""  # Your Project ID
        litellm.vertex_location = ""  # Your Project Location
        litellm.drop_params = True
    else:
        args.model_name = "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"
    if args.cot:
        mmlu_cot_prompt = json.load(open("data/mmlu-cot.json"))
    all_acc = 0
    all_number = 0
    accs_json = {}
    method_name = "cot" if args.cot else "simple"
    outputs_file = open(f"outputs/{args.model_name}_{method_name}_outputs.json", "a")
    for task in tasks:
        print("Testing %s ..." % task)
        acc = 0
        dev_df = pd.read_csv(
            os.path.join("data", "dev", task + "_dev.csv"), header=None
        )[: args.num_examples]
        test_df = pd.read_csv(
            os.path.join("data", "test", task + "_test.csv"), header=None
        )
        for i in tqdm(range(test_df.shape[0])):
            if args.cot:
                # chain of thought
                q = test_df.iloc[i, 0] + "\n"
                for j, letter in enumerate(["A", "B", "C", "D"]):
                    q += "(" + letter + ") " + str(test_df.iloc[i, j + 1]) + " "
                q += "\nA: Let's think step by step."

                prompt = mmlu_cot_prompt[task] + "\n\n" + q
                label = test_df.iloc[i, test_df.shape[1] - 1]

                response = asyncio.run(get_response(prompt, args.model_name))
                ans_model = response["choices"][0]["message"]["content"]
                ans_, residual = extract_ans(ans_model)

                ans_model, correct = test_answer_mmlu_(ans_, label)
                if correct:
                    acc += 1
            else:
                # simple prompting
                k = args.num_examples
                prompt_end = format_example(test_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, task, k)
                prompt = train_prompt + prompt_end
                label = test_df.iloc[i, test_df.shape[1] - 1]

                response = asyncio.run(get_response(prompt, args.model_name))
                # 0 means the answer character [A, B, C, D] (sometimes model will output more)
                ans_model = response["choices"][0]["message"]["content"][0]

                correct = ans_model == label
                if correct:
                    acc += 1
            outputs_file.write(
                json.dumps(
                    {
                        "task": task,
                        "correct": correct,
                        "prediction": ans_model,
                        "label": label,
                        "response": response["choices"][0]["message"]["content"],
                        "question": test_df.iloc[i, 0],
                        "A": test_df.iloc[i, 1],
                        "B": test_df.iloc[i, 2],
                        "C": test_df.iloc[i, 3],
                        "D": test_df.iloc[i, 4],
                        "prompt": prompt,
                    }
                )
                + "\n"
            )
        print("%s acc %.4f" % (task, acc / test_df.shape[0]))
        accs_json[task] = acc / test_df.shape[0]
        all_acc += acc
        all_number += test_df.shape[0]
    accs_json["all"] = all_acc / all_number
    json.dump(
        accs_json, open(f"outputs/{args.model_name}_{method_name}_accs.json", "w")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo", "gpt-4-1106-preview", "gemini-pro", "mixtral"],
    )
    parser.add_argument("--cot", action="store_true")
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of examples included in the current prompt input. ",
    )
    args = parser.parse_args()
    main(args)
