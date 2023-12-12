"""Create prompt for different settings: 
 - NL-to-Code generation 
 - More test case generation
"""

import random, string
from typing import Dict, List, Union


def remove_indent(test: str) -> str:
    """Remove the first level of indentation inside unit tests."""
    lines = test.split("\n")
    lines = [l[4:] for l in lines]
    return "\n".join(lines)


def add_indent(test: str) -> str:
    """add 1-level of indentation to be wrapped into the check function."""
    lines = test.split("\n")
    lines = [(" " * 4 + l) for l in lines]
    return "\n".join(lines)


def create_entry_point(
    intent: str,
    first_nwords: int = 4,
    stop_words: List[str] = ["in", "of", "a", "to", "and", "for", "with", "that"],
    delimiters: List[str] = ["`", "'"],
    verbose: bool = False,
) -> str:
    """Heuristically assign (meaningful) function name from the rewritten-intent."""
    words = [w.lower() for w in intent.split()[:first_nwords]]
    if verbose:
        print(f"[words 0] {words}")

    for idx, word in enumerate(words):
        try:
            word_num = float(word)
            break
        except:
            continue
    else:
        idx += 1
    words = words[:idx]
    if verbose:
        print(f"[words 1] {words}")

    for idx, word in enumerate(words):
        if any([word == sw for sw in stop_words]) and idx > 1:
            break
    else:
        idx += 1
    words = words[:idx]
    if verbose:
        print(f"[words 2] {words}")

    for idx, word in enumerate(words):
        if any([word.startswith(de) for de in delimiters]):
            break
    else:
        idx += 1
    words = words[:idx]
    if verbose:
        print(f"[words 3] {words}")

    words = [
        "".join([c for c in word if (not c in string.punctuation)]) for word in words
    ]
    words = [word for word in words if word.strip()]
    if verbose:
        print(f"[words 4] {words}")
    if len(words) < 2:
        words = ["f"] + words[:first_nwords]

    if words[0].startswith("¿"):
        words[0] = words[0][1:]

    return "_".join(words)


def get_default_entry_point(task_id: Union[int, str]) -> str:
    return f"f_{task_id}"


def replace_entry_point(function_head: str, alternative_name: str) -> str:
    """Replace the default function name to semantically meaningful one.
    E.g., "f_12345" -> "count_items"
    args:
        function_head: "def f_3844801(myList):"
        description: e.g., "check if all elements in list are identical"
    rets:
        sema_function_head: e.g. "def check_elements_identical(myList):"
    """
    arguments = function_head[function_head.index("(") :]
    return f"def {alternative_name}{arguments}"


def replace_function_name_test(test_case: str, function_name: str) -> str:
    return test_case.replace("candidate", function_name)


def get_test_body(
    function_name: str,
    tests: List[str],
    num_tests: int = 0,
    padding: bool = False,
    clean_indent: bool = True,
) -> str:
    if num_tests == 0:
        return ""

    if len(tests) >= num_tests:
        selected_tests = random.sample(tests, num_tests)
    else:
        if padding:
            selected_tests = []
            for _ in range(num_tests):
                selected_tests.extend(random.sample(tests, 1))
        else:
            selected_tests = tests

    if clean_indent:
        selected_tests = [remove_indent(t) for t in selected_tests]

    selected_tests = [
        replace_function_name_test(t, function_name) for t in selected_tests
    ]
    return "".join(selected_tests)


def get_entry_point(sample: Dict, method: str) -> str:
    if method == "id":
        return get_default_entry_point(sample["task_id"])
    elif method == "constant":
        return "function"
    elif method == "intent":
        return create_entry_point(sample["intent"])
    else:
        raise ValueError("FuncName Method [{method}] Not Supported!")


def create_prompt_nl2code(
    sample: Dict,
    num_tests: int = 0,
    function_name: str = "id",
) -> str:
    """Create Codex prompt for NL-to-Code generation.
    args:
        sample: annotated {
            'task_id': str, 'intent': str, 'canonical_solution': str,
            'prompt': str, 'suffix': str, 'entry_point': str,
            'test_start': str, 'test': List[str],
        }
        num_tests: number of unit tests included in the prompt.
    rets:
        prompt: input to codex
    """
    function_head, function_prefix = [p for p in sample["prompt"].split("\n")]

    assert (
        sample["intent"] is not None
    ), f"NL intent is None for sample {sample['task_id']}"
    if function_name == "task_id":
        function_name = get_default_entry_point(task_id=sample["task_id"])
    elif function_name == "constant":
        function_name = "function"
        function_head = replace_entry_point(function_head, function_name)
    elif function_name == "intent":
        function_name = create_entry_point(intent=sample["intent"])
        function_head = replace_entry_point(function_head, function_name)
    else:
        raise ValueError("Method [{args.function_name}] Not Supported!")
    test_str = get_test_body(
        function_name, sample["test"], num_tests, padding=False, clean_indent=False
    )
    docstr = f'    """{sample["intent"]}\n{test_str}    """'
    code_body = function_prefix.replace("\t", " " * 4)
    prompt = "\n".join([function_head, docstr, code_body])
    return prompt


def create_prompt_example(
    example: Dict, num_tests: int = 0, function_name: str = "id"
) -> str:
    prompt = create_prompt_nl2code(example, num_tests, function_name)
    return f"{prompt}{example['canonical_solution']}{example['suffix']}"


def create_fewshot_prompt_nl2code(
    sample: Dict,
    examples: List[Dict],
    num_tests: int = 0,
    function_name: str = "id",
) -> str:
    """Create few-shot prompt for NL-to-Code generation.
    args:
        sample: dict {'task_id', 'intent', 'canonical_solution', 'entry_point',
                      'prompt', 'suffix', 'test_start', 'test'}
        examples: list[dict], list of prefix examples
        num_tests: int, number of tests cases in the docstring
        function_name: str, method to build function name
                       {"id": f_12345, "constant": "function", "intent": "get_max_item"}
    rets:
        prompt: str, include examples if specified
    """
    if "intent" in sample:
        function_head, function_prefix = [p for p in sample["prompt"].split("\n")]
    if function_name == "id":
        if "intent" in sample:
            function_name = get_default_entry_point(task_id=sample["task_id"])
        else:
            function_name = sample["entry_point"]
    elif function_name == "constant":
        function_name = "function"
        function_head = replace_entry_point(function_head, function_name)
    elif function_name == "intent":
        function_name = create_entry_point(intent=sample["intent"])
        function_head = replace_entry_point(function_head, function_name)
    else:
        raise ValueError("Method [{args.function_name}] Not Supported!")
    test_str = get_test_body(
        function_name, sample["test"], num_tests, padding=False, clean_indent=False
    )
    if "intent" in sample:
        docstr = f'    """{sample["intent"]}\n{test_str}    """'
        code_body = function_prefix.replace("\t", " " * 4)
        prompt = "\n".join([function_head, docstr, code_body])
    else:
        prompt = sample["prompt"]

    if len(examples) == 0:
        return prompt

    examples = "\n\n".join(
        [
            create_prompt_example(example=ex, num_tests=0, function_name=function_name)
            for ex in examples
        ]
    )
    return f"{examples}\n\n{prompt}"


def create_prompt_test_gen(
    sample: Dict,
    add_solution: bool = False,
    num_tests: int = 0,
    function_name_option: str = "id",
) -> str:
    """Create Codex prompt for NL-to-Code generation.
    args:
        sample: annotated {
            'task_id': str, 'intent': str, 'canonical_solution': str,
            'prompt': str, 'suffix': str, 'entry_point': str,
            'test_start': str, 'test': List[str],
        }
        num_tests: number of unit tests included in the prompt.
        function_name: "default": "function"; "intent": extract from nl; "id": question id.
    rets:
        prompt: input to codex
    """
    function_head, function_prefix = [p for p in sample["prompt"].split("\n")]
    doc_str = f'    """{sample["intent"]}\n    """'
    if add_solution:
        solution = sample["canonical_solution"] + sample["suffix"]
    else:
        solution = "pass"
    code_body = function_prefix + solution
    code_body = code_body.replace("\t", " " * 4)

    assert (
        sample["intent"] is not None
    ), f"NL intent is None for sample {sample['task_id']}"
    if function_name_option == "intent":
        function_name = create_entry_point(intent=sample["intent"])
        function_head = replace_entry_point(function_head, function_name)
    elif function_name_option == "default":
        function_name = "function"
        function_head = replace_entry_point(function_head, function_name)
    else:  # function_name_option == "id"
        function_name = get_default_entry_point(task_id=sample["task_id"])

    instruction = f"\n# check the correctness of `{function_name}`"
    test_str = get_test_body(
        function_name, sample["test"], num_tests, padding=False, clean_indent=True
    )

    prompt = "\n".join([function_head, doc_str, code_body, instruction, test_str])
    return prompt
