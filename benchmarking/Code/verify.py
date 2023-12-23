"""Verify the correctness of model predictions. """

import os, random
from typing import Dict, List
from processor import CodeProcessor
from prompt import get_entry_point

import evaluate

from prompt import add_indent

# bleu_eval_metric = load_metric("bleu")
code_eval_metric = evaluate.load("code_eval")
# os.environ["HF_ALLOW_CODE_EVAL"] = "0"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def get_valid_solutions(
    predictions: List[str],
    deduplicate: bool = False,
) -> List[str]:
    processor = CodeProcessor()
    solutions = []
    for pred in predictions:
        valid_sol = processor.code_extract(pred)
        solutions.append(valid_sol)

    if deduplicate:
        solutions = list(set(solutions))
    return solutions


def wrap_check(
    sample: Dict,
    solution_list: List[str],
    k: List[int],
    num_workers: int = 1,
    max_num_tests: int = 1,
    verbose: bool = False,
    exclude_suffix: bool = False,
    function_name: str = "id",
):
    if exclude_suffix:
        if "intent" in sample:
            wrapped_solution_list = [
                f"{sample['prompt']}{solution}".replace("\t", " " * 4)
                for solution in solution_list
            ]
        else:
            wrapped_solution_list = [
                f"{sample['prompt']}    {solution}".replace("\t", " " * 4)
                for solution in solution_list
            ]
    else:
        wrapped_solution_list = [
            f"{sample['prompt']}{solution}{sample['suffix']}".replace("\t", " " * 4)
            for solution in solution_list
        ]
    if isinstance(sample["test"], list):
        # ODEX
        max_num_tests = min(len(sample["test"]), max_num_tests)
        test_case = random.sample(sample["test"], max_num_tests)
        entry_point = get_entry_point(sample, function_name)
        check_function = "\n".join(
            [
                sample["test_start"],
                "".join(test_case),
                "",
                f"check({entry_point})",
            ]
        )
    else:
        # HumanEval
        entry_point = sample["entry_point"]
        check_function = "\n".join(
            [
                sample["test"],
                "",
                f"check({entry_point})",
            ]
        )
    scores, outputs = code_eval_metric.compute(
        predictions=[wrapped_solution_list],
        references=[check_function],
        k=k,
        num_workers=num_workers,
    )
    if verbose:
        print(f"[predic] {wrapped_solution_list}")
        print(f"[fcheck] {check_function}")
        print(f"[scores] {scores}")
        print(f"[output] {outputs[0]}")
    return scores, outputs[0]


def wrap_check_test(
    prompt: str,
    suffix: str,
    solution_list: List[str],
    test_start: str,
    test_case: str,
    entry_point: str,
    k: List[int] = [1],
    num_workers: int = 1,
    add_indent_test: bool = True,
    verbose: bool = False,
):
    wrapped_solution_list = [
        f"{prompt}{solution}{suffix}".replace("\t", " " * 4)
        for solution in solution_list
    ]

    if add_indent_test == True:
        test_case = add_indent(test_case)
    check_function = "\n".join(
        [
            test_start,
            test_case,
            "",
            f"check({entry_point})",
        ]
    )
    if verbose:
        print(f"[solution list] \n{wrapped_solution_list}")
        print(f"[check function] \n{check_function}")
    scores, outputs = code_eval_metric.compute(
        predictions=[wrapped_solution_list],
        references=[check_function],
        k=k,
        num_workers=num_workers,
    )
    if verbose:
        print(f"[scores] {scores}")
        print(f"[output] {outputs[0]}")
        print("-" * 25)
    return scores, outputs[0]
