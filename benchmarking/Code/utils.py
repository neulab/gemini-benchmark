"""Utility functions. 
 - data path: testset, prediction set 
 - logger initialization
 - write output to file(s)
"""

import os
import sys
import json
import torch
import logging
from typing import Dict, List


def get_test_path(language: str) -> str:
    return os.path.join("data", f"{language}_test.jsonl")


def get_prediction_dir(model_name: str) -> str:
    return "outputs"


def get_prediction_name(
    language: str, num_examples: int = 0, num_tests: int = 0
) -> str:
    return f"{language}_{num_examples}-{num_tests}"


def get_prediction_path(
    model_name: str = "codegen",
    language: str = "en",
    num_examples: int = 0,
    num_tests: int = 0,
) -> str:
    pdir = get_prediction_dir(model_name)
    pname = get_prediction_name(language, num_examples, num_tests)
    return os.path.join(pdir, pname)


logger = logging.getLogger(__name__)


def init_logger(is_main=True, is_distributed=False, filename=None):
    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    return logger


def write_output(glob_path, output_path, file_type: str = "jsonl"):
    files = list(glob_path.glob(f"*.{file_type}"))
    print(f"FIles: {files}")
    files.sort()
    with open(output_path, "w") as outfile:
        for path in files:
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    outfile.write(line)
            path.unlink()
    glob_path.rmdir()


def get_testgen_path(language: str, num_tests: int) -> str:
    return os.path.join("data", "testgen", f"{language}_tgen-{num_tests}.json")


def load_testset(path: str) -> List[Dict]:
    dataset = [json.loads(l.strip()) for l in open(path, "r")]
    print(f"load dataset #{len(dataset)}")
    return dataset


def print_scores_dict(scores_dict: Dict, n: int = 10, multi_row: bool = True) -> None:
    scores_text = []
    for idx in range(n):
        key = f"pass@{idx+1}"
        scores = scores_dict[key]
        scores_text.append(f"[{key}] {sum(scores)/len(scores):.4f} ({len(scores)})")

    if multi_row:
        for stext in scores_text:
            print(stext)
    else:
        print(" ".join(scores_text))


def adaptive_sleep_time(
    base_time: int = 10, num_return_sequences: int = 1, round_idx: int = 0
):
    return base_time * (round_idx + 1) * num_return_sequences


def get_libs(text: str) -> List[str]:
    libraries = []
    lines = text.split("\n")
    for line in lines:
        if "from " in line:
            assert " import " in line
            libs_text = line[line.index("from ") + 5 : line.index(" import ")]
            libs = [libs_text.split(".")[0].strip()]
        elif "import " in line:
            libs_text = line[line.index("import ") + 7 :]
            if " as " in line:
                libs_text = libs_text[: libs_text.index(" as ")]
            libs = [l.strip() for l in libs_text.split(",")]
            libs = [lb.split(".")[0].strip() for lb in libs]
        else:
            libs = []
        libraries.extend(libs)
    return libraries
