import dotenv

from collections import Counter
import re
import math

import pandas as pd
import csv
import os
import evaluate
from zeno_client import ZenoClient, ZenoMetric

dotenv.load_dotenv("../.env", override=True)

# Directory paths
data_dir = './data/flores200_dataset/devtest'
trg_langs = ["lvs_Latn", "tpi_Latn", "ukr_Cyrl", "lim_Latn", "kat_Geor", "tam_Taml", "mag_Deva", "hau_Latn", "fra_Latn",
             "acm_Arab", "ssw_Latn", "kmr_Latn", "war_Latn", "ajp_Arab", "pbt_Arab", "gle_Latn", "ron_Latn", "sna_Latn",
             "ckb_Arab", "ibo_Latn"]

# Create general (non-system specific) dataframe

# Initialize Zeno client
client = ZenoClient(os.environ.get("ZENO_API_KEY"))

# Load reference data
src_data, trg_data, trg_lang_data = [], [], []
src_lens, tgt_lens = [], []
lang_families = []
for trg_lang in trg_langs:
    with open(os.path.join(data_dir, "devtest.eng_Latn"), "r") as f_src, \
            open(os.path.join(data_dir, f"devtest.{trg_lang}"), "r") as f_trg:
        src_lines = [line.strip() for line in f_src.readlines()]
        src_len = [len(line) for line in src_lines]
        trg_lines = [line.strip() for line in f_trg.readlines()]
        tgt_len = [len(line) for line in trg_lines]

        src_data.extend(src_lines)
        trg_data.extend(trg_lines)
        src_lens.extend(src_len)
        tgt_lens.extend(tgt_len)
        trg_lang_data.extend([trg_lang] * len(src_lines))
        lang_families.extend([trg_lang.split('_')[1]] * len(src_lines))

# Create the reference DataFrame
df_ref = pd.DataFrame(
    {"source": src_data, "target": trg_data, "trg_lang": trg_lang_data, "src_len": src_lens, "tgt_len": tgt_lens,
     "lang_script": lang_families})
df_ref["id"] = df_ref.index

# Upload reference data to Zeno
project = client.create_project(
    name="Flores Translation Evaluation",
    view="text-classification",
    metrics=[ZenoMetric(name="CHRF", type="mean", columns=["chrf_score"])]
)
project.upload_dataset(df_ref, id_column="id", data_column='source', label_column="target")

# Create non-system specific dataframe

# Run for each system
system_outputs_dir = '~/gemini/gpt_mt_benchmark/exp_outputs_tsv-gemini'
# system_outputs_dir = '~/home/amuhamed/gemini/gpt_mt_benchmark/exp_outputs_tsv-3.5'
# system_outputs_dir = '~/home/amuhamed/gemini/gpt_mt_benchmark/exp_outputs_tsv-4'

# Load CHRF evaluator
chrf_evaluator = evaluate.load("chrf")


# Function to load predictions from TSV
def load_predictions_from_tsv(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # Skip header
        return [row[2] for row in reader]


def calculate_chrf(predictions, references, trg_lang):
    filtered_references = references[references['trg_lang'] == trg_lang]['target']
    chrf_scores = []
    for pred, ref in zip(predictions, filtered_references):
        score = chrf_evaluator.compute(predictions=[pred], references=[[ref]])["score"]
        if pd.isnull(score) or math.isnan(score):
            chrf_scores.append(0)
        else:
            chrf_scores.append(score)
    return chrf_scores


# Function to compute the maximum number of repeated words in predictions
def max_repeated_words(predictions):
    max_repeats = []
    for prediction in predictions:
        words = re.findall(r'\b\w+\b', prediction.lower())
        word_counts = Counter(words)
        max_count = max(word_counts.values()) if word_counts else 0
        max_repeats.append(max_count)
    return max_repeats


# Load and upload system outputs

for system_name in os.listdir(system_outputs_dir):
    system_path = os.path.join(system_outputs_dir, system_name)
    df_system_all_langs = pd.DataFrame()
    for trg_lang in trg_langs:

        if not os.path.isdir(system_path):
            continue

        for filename in os.listdir(system_path):
            if filename.endswith(f"{trg_lang}.tsv"):
                tsv_file = os.path.join(system_path, filename)
                break

        predictions = load_predictions_from_tsv(tsv_file)
        predictions_len = [len(pred) for pred in predictions]

        # Calculate CHRF scores
        chrf_scores = calculate_chrf(predictions, df_ref, trg_lang)

        # Calculate max repeated words for each prediction
        max_repeats = max_repeated_words(predictions)

        # Create DataFrame for system predictions
        df_system = pd.DataFrame({
            "output": predictions,
            "chrf_score": chrf_scores,
            "max_repeats": max_repeats,
            "pred_len": predictions_len,
        })

        # Replace all empty string outputs with the word 'empty'
        # df_system['output'] = df_system['output'].replace('', 'empty')

        df_system_all_langs = pd.concat([df_system_all_langs, df_system])

    # Reset index of the concatenated DataFrame
    df_system_all_langs.reset_index(drop=True, inplace=True)
    df_system_all_langs["id"] = df_system_all_langs.index

    # Upload system predictions to Zeno

    if '3.5' in system_path:
        if 'zero' in system_path:
            upload_system_name = 'gpt-3.5-turbo-0-shot'
        else:
            upload_system_name = 'gpt-3.5-turbo-5-shot'
    elif '4' in system_path:
        if 'zero' in system_path:
            upload_system_name = 'gpt-4-turbo-0-shot'
        else:
            upload_system_name = 'gpt-4-turbo-5-shot'
    elif 'gemini' in system_path:
        if 'zero' in system_path:
            upload_system_name = 'gemini-pro-0-shot'
        else:
            upload_system_name = 'gemini-pro-5-shot'
    else:
        raise ValueError("No dir found")
    project.upload_system(df_system_all_langs, name=upload_system_name, id_column="id", output_column="output")
