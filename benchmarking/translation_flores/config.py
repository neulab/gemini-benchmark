"""Various configuration options for the chatbot task.

This file is intended to be modified. You can go in and change any
of the variables to run different experiments.
"""
from __future__ import annotations

import transformers

from zeno_build.evaluation.text_features.exact_match import avg_exact_match, exact_match
from zeno_build.evaluation.text_features.length import (
    chat_context_length,
    input_length,
    label_length,
    output_length,
)
from zeno_build.evaluation.text_metrics.critique import (
    avg_bert_score,
    avg_chrf,
    avg_length_ratio,
    bert_score,
    chrf,
    length_ratio,
)
from zeno_build.experiments import search_space
from zeno_build.models.dataset_config import DatasetConfig
from zeno_build.models.lm_config import LMConfig
from zeno_build.prompts.chat_prompt import ChatMessages, ChatTurn

# Define the space of hyperparameters to search over.
space = search_space.CombinatorialSearchSpace(
    {
        "dataset_preset": search_space.Constant("local-fra"),
        "model_preset": search_space.Categorical(
            [
                "gpt-3.5-turbo",
            ]
        ),
        "prompt_preset": search_space.Discrete(
            ["tt-def"]
        ),
        "temperature": search_space.Discrete([0.3]),
        "context_length": search_space.Discrete([-1]),
        "max_tokens": search_space.Constant(500),
        "top_p": search_space.Constant(1.0),
    }
)

# The number of trials to run
num_trials = 1

# The details of each dataset
dataset_configs = {
    "flores200": DatasetConfig(
        dataset=("facebook/flores", "swh_Latn-eng_Latn"),
        split="devtest",
        data_column="sentence_swh_Latn",
        data_format="flores",
    ),
    "local-ssw": DatasetConfig(  # This is the format we use.
        dataset="",
        split="devtest",
        data_column="",  # not relevant as we have plain txt files as inputs
        data_format="local",
    ),
}

# The details of each model
model_configs = {
    "text-davinci-003": LMConfig(provider="openai", model="text-davinci-003"),
    "gpt-3.5-turbo": LMConfig(provider="openai_chat", model="gpt-3.5-turbo"),
    "gpt-4-turbo": LMConfig(provider="openai_chat", model="gpt-4-1106-preview"),
    "gemini-pro": LMConfig(provider="litellm", model="gemini-pro"),
}

# The details of the prompts - we incorporated the prompts with the dataset so we use the default which is just empty
prompt_messages: dict[str, ChatMessages] = {
    "tt-def": ChatMessages(
        messages=[
            ChatTurn(
                role="user",
                content="",
            ),
        ]
    ),
}

# The functions to use to calculate scores for the hyperparameter sweep
sweep_distill_functions = [chrf]
sweep_metric_function = avg_chrf

# The functions used for Zeno visualization
zeno_distill_and_metric_functions = [
    output_length,
    input_length,
    label_length,
    chat_context_length,
    chrf,
    length_ratio,
    bert_score,
    exact_match,
    avg_chrf,
    avg_length_ratio,
    avg_bert_score,
    avg_exact_match,
]
