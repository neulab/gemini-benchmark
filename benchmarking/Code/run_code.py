"""Evaluate Codex performance on NL-to-Code generation. """

from utils import get_test_path, get_prediction_path, load_testset
from prompt import create_fewshot_prompt_nl2code
from verify import get_valid_solutions, wrap_check
from litellm import acompletion
from typing import Dict, List
from tqdm import tqdm
import json, argparse
import os, random
import asyncio


async def get_response(
    prompt: str,
    sample: Dict,
    verbose: bool = False,
):
    if verbose:
        print(f"[prompt] \n{prompt}\n------")
    response = await acompletion(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        # suffix=sample["suffix"],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["###"],
    )
    return response


def select_fewshot_examples(
    sample: Dict,
    candidates: List[Dict],
    num_examples: int = 1,
    method: str = "random",
) -> List[Dict]:
    """Select example as prefix to the prompt of the current sample."""
    if method == "random":
        num_examples = min(num_examples, len(candidates))
        return random.sample(candidates, num_examples)


def main():
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    # load source dataset
    dataset = load_testset(args.input_path)

    predset = []
    scores_dict = {f"pass@{idx}": [] for idx in range(1, args.n + 1)}
    outputs_file = open(f"{args.output_path}_outputs.json", "a")

    for i, sample in tqdm(enumerate(dataset)):
        if "suffix" in sample:
            # current gpts cannot handle suffix
            if sample["suffix"] != "":
                continue
        # create model input -- prompt
        examples = select_fewshot_examples(
            sample=sample,
            candidates=dataset[:i] + dataset[i + 1 :],
            num_examples=args.num_examples,
            method=args.fewshot_method,
        )
        prompt = create_fewshot_prompt_nl2code(
            sample=sample,
            examples=examples,
            num_tests=args.num_tests,
            function_name=args.function_name,
        )

        # collect code predictions
        response = asyncio.run(
            get_response(prompt=prompt, sample=sample, verbose=args.verbose)
        )
        predictions = [
            response["choices"][i]["message"]["content"]
            for i in range(len(response["choices"]))
        ]

        # simple cleansing of predicions
        valid_predictions = get_valid_solutions(predictions, deduplicate=False)
        num_valid = len(valid_predictions)
        assert num_valid == args.n, f"# num_valid"
        scores, outputs = wrap_check(
            sample,
            valid_predictions,
            k=[i + 1 for i in range(num_valid)],
            num_workers=args.n,
            max_num_tests=args.num_tests_eval,
            verbose=args.verbose,
            exclude_suffix=True,
            function_name=args.function_name,
        )
        if i % 10 == 0:
            print(f"[scores@{i:3d}] {scores}")

        for idx in range(num_valid):
            key = f"pass@{idx+1}"
            if key in scores:
                scores_dict[key].append(scores[key])
        outputs_file.write(
            json.dumps(
                {
                    "scores": scores,
                    "output": outputs,
                    "predictions": valid_predictions,
                    "task_id": sample["task_id"],
                    "question": sample["prompt"],
                    "canonical_solution": sample["canonical_solution"],
                    "test": sample["test"],
                    "entry_point": sample["entry_point"],
                    "prompt": prompt,
                }
            )
            + "\n"
        )

    for idx in range(args.n):
        key = f"pass@{idx+1}"
        scores = scores_dict[key]
        scores_dict[key] = sum(scores) / len(scores)
        print(f"[{key}] {sum(scores)/len(scores):.3f} ({len(scores)})")
    json.dump(scores_dict, open(f"{args.output_path}_accs.json", "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", type=str, default="en", choices=["en", "es", "ja", "ru"]
    )
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)

    parser.add_argument("--num_tests", type=int, default=0)
    parser.add_argument("--num_tests_eval", type=int, default=100)

    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo"],
    )
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of predictions required for each api call.",
    )

    parser.add_argument(
        "--sleep_time",
        type=int,
        default=60,
        help="Specify a positive integer if enable time sleep.",
    )

    parser.add_argument(
        "--function_name",
        type=str,
        default="id",
        choices=["id", "constant", "intent"],
        help="Method to construct the function name. ",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=0,
        help="Number of examples included in the current prompt input. ",
    )
    parser.add_argument(
        "--fewshot_method",
        type=str,
        default="random",
        choices=["random"],
        help="Method to select the prefix examples for prompt creation.",
    )

    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if (not args.input_path) or (not args.output_path):
        if not args.language:
            raise Exception(f"Need to specify [language] or [i/o path]")
        if not args.input_path:
            args.input_path = get_test_path(args.language)
        if not args.output_path:
            args.output_path = get_prediction_path(
                args.model_name,
                args.language,
                args.num_examples,
                args.num_tests,
            )

    main()
