import pandas as pd
import csv
import os
import evaluate
from zeno_client import ZenoClient, ZenoMetric

# Initialize Zeno client with your API key
client = ZenoClient("API-KEY")

# Directory paths
data_dir = './data/flores200_dataset/devtest'
system_outputs_dir = './exp_outputs_tsv'
trg_langs = ["ssw_Latn"]

# Load CHRF evaluator
chrf_evaluator = evaluate.load("chrf")

# Load reference data
src_data, trg_data, trg_lang_data = [], [], []
for trg_lang in trg_langs:
    with open(os.path.join(data_dir, "devtest.eng_Latn"), "r") as f_src, \
            open(os.path.join(data_dir, f"devtest.{trg_lang}"), "r") as f_trg:
        src_lines = [line.strip() for line in f_src.readlines()]
        trg_lines = [line.strip() for line in f_trg.readlines()]

        src_data.extend(src_lines)
        trg_data.extend(trg_lines)
        trg_lang_data.extend([trg_lang] * len(src_lines))

# Create the reference DataFrame
df_ref = pd.DataFrame({"source": src_data, "target": trg_data, "trg_lang": trg_lang_data})
df_ref["id"] = df_ref.index

# Upload reference data to Zeno
project = client.create_project(
    name="Translation Evaluation",
    view="text-classification",
    metrics=[ZenoMetric(name="CHRF", type="mean", columns=["chrf_score"])]
)
project.upload_dataset(df_ref, id_column="id", data_column='source', label_column="target")


# Function to load predictions from TSV
def load_predictions_from_tsv(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # Skip header
        return [row[2] for row in reader]


# Function to calculate CHRF for each prediction
def calculate_chrf(predictions, references):
    return [chrf_evaluator.compute(predictions=[pred], references=[[ref]])["score"] for pred, ref in
            zip(predictions, references)]


# # Function to extract language and model details from filename
# def extract_details_from_filename(filename):
#     pattern = r'gpt-[\d.]+-[^_]+_([^_]+)'
#     match = re.search(pattern, filename)
#     if match:
#         return match.group(1)  # Returns the language code
#     return "unknown"


# Load and upload system outputs
for trg_lang in trg_langs:
    for system_name in os.listdir(system_outputs_dir):
        system_path = os.path.join(system_outputs_dir, system_name)

        if not os.path.isdir(system_path):
            continue

        for filename in os.listdir(system_path):
            if filename.endswith(f"{trg_lang}.tsv"):
                tsv_file = os.path.join(system_path, filename)
                break
        # tsv_file = os.path.join(system_path, f"{trg_lang}.tsv")

        predictions = load_predictions_from_tsv(tsv_file)

        # Calculate CHRF scores
        chrf_scores = calculate_chrf(predictions, df_ref["target"])

        # Create DataFrame for system predictions
        df_system = pd.DataFrame({
            "output": predictions,
            "chrf_score": chrf_scores,
            "id": range(len(predictions)),
        })

        # Upload system predictions to Zeno
        project.upload_system(df_system, name=system_name, id_column="id", output_column="output")
