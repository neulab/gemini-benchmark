"""The main entry point for performing comparison on chatbots."""

from __future__ import annotations

import sys

import argparse
import json
import logging
import os
from dataclasses import asdict
import pandas as pd
import config as chatbot_config
from modeling import make_predictions, process_data
from zeno_build.experiments import search_space
from zeno_build.experiments.experiment_run import ExperimentRun
from zeno_build.optimizers import standard
from zeno_build.prompts.chat_prompt import ChatMessages
from zeno_build.reporting import reporting_utils
from zeno_build.reporting.visualize import visualize

from flores200_utils import *

os.environ["OPENAI_API_KEY"] = "####"
os.environ['INSPIREDCO_API_KEY'] = ""
os.environ["TOGETHERAI_API_KEY"] = ""


def chatbot_main(
        dataset: str,
        dataset_config_preset: str,
        results_dir: str,
        do_prediction: bool = True,
        do_visualization: bool = True,
):
    """Run the chatbot experiment."""
    # Get the dataset configuration
    dataset_preset = chatbot_config.space.dimensions["dataset_preset"]
    if not isinstance(dataset_preset, search_space.Constant):
        raise ValueError("All experiments must be run on a single dataset.")
    dataset_config = chatbot_config.dataset_configs[dataset_config_preset]
    print(chatbot_config.space.dimensions["prompt_preset"])

    # Define the directories for storing data and predictions
    data_dir = os.path.join(results_dir, "data")
    split_folder = dataset.split("/")  # --dataset "fewshot_texts/tt-zero/eng_Latn
    prompt = split_folder[-2]
    lang = split_folder[-1]
    predictions_dir = os.path.join(results_dir, "predictions", prompt, lang)

    # Load and standardize the format of the necessary data. The resulting
    # processed data will be stored in the `results_dir/data` directory
    # both for browsing and for caching for fast reloading on future runs.
    contexts_and_labels = process_data(
        dataset=dataset,  # TODO - changed from the original
        split=dataset_config.split,
        data_format=dataset_config.data_format,
        data_column=dataset_config.data_column,
        output_dir=data_dir,
    )
    # TODO --changed this
    labels: list[str] = load_labels(dataset)
    contexts: list[ChatMessages] = []

    for x in contexts_and_labels:
        contexts.append(x)

    assert len(contexts) == len(labels), "The contexts and labels should be the same length"

    if do_prediction:
        # Perform the hyperparameter sweep
        optimizer = standard.StandardOptimizer(
            space=chatbot_config.space,
            distill_functions=chatbot_config.sweep_distill_functions,
            metric=chatbot_config.sweep_metric_function,
            num_trials=chatbot_config.num_trials,
        )

        while not optimizer.is_complete(predictions_dir, include_in_progress=True):
            parameters = optimizer.get_parameters()
            if parameters is None:
                break
            predictions = make_predictions(
                contexts=contexts,
                dataset_preset=parameters["dataset_preset"],
                prompt_preset=parameters["prompt_preset"],
                model_preset=parameters["model_preset"],
                temperature=parameters["temperature"],
                max_tokens=parameters["max_tokens"],
                top_p=parameters["top_p"],
                context_length=parameters["context_length"],
                output_dir=predictions_dir,
            )
            if predictions is None:
                print(f"*** Skipped run for {parameters=} ***")
                continue
            #Uncomment below to calculate metrics using zeno - takes some time.
            # eval_result = optimizer.calculate_metric(contexts, labels, predictions)
            # print("*** Iteration complete. ***")
            # print(f"Parameters: {parameters}")
            # print(f"Eval: {eval_result}")
            print("***************************")

    if do_visualization:
        param_files = chatbot_config.space.get_valid_param_files(
            predictions_dir, include_in_progress=False
        )
        if len(param_files) < chatbot_config.num_trials:
            logging.getLogger().warning(
                "Not enough completed but performing visualization anyway."
            )
        results: list[ExperimentRun] = []

        dir = "exp_outputs_tsv/" + prompt

        try:
            os.makedirs(dir)
        except FileExistsError:
            pass

        for param_file in param_files:
            assert param_file.endswith(".zbp")
            with open(param_file, "r") as f:
                loaded_parameters = json.load(f)
            with open(f"{param_file[:-4]}.json", "r") as f:
                predictions = json.load(f)
            name = reporting_utils.parameters_to_name(
                loaded_parameters, chatbot_config.space
            )
            name = name + "_" + prompt + "_" + lang
            print(name)
            results.append(
                ExperimentRun(
                    parameters=loaded_parameters, predictions=predictions, name=name
                )
            )
            results_df = pd.DataFrame(
                {
                    "messages": [[asdict(y) for y in x.messages] for x in contexts],
                    "label": labels,
                    "predictions": predictions
                }
            )
            results_df.to_csv(os.path.join(dir, name + ".tsv"), index=False, sep="\t")


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to run",
    )
    parser.add_argument(
        "--dataset_config_preset",
        type=str,
        required=True,
        help="The dataset preset name in the config file. Should be something like local-fls for local datasets",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="The directory to store the results in.",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Skip prediction and just do visualization.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization and just do prediction.",
    )
    args = parser.parse_args()

    if args.skip_prediction and args.skip_visualization:
        raise ValueError(
            "Cannot specify both --skip-prediction and --skip-visualization."
        )

    chatbot_main(
        dataset=args.dataset,
        dataset_config_preset=args.dataset_config_preset,
        results_dir=args.results_dir,
        do_prediction=not args.skip_prediction,
        do_visualization=not args.skip_visualization,
    )
