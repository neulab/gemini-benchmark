from zeno_client import ZenoClient, ZenoMetric
import pandas as pd
import os
import dotenv
import re
dotenv.load_dotenv(override=True)

zeno_client = ZenoClient("ZENO_API_KEY")

OUTPUT_DIR = "../../gemini-benchmark/outputs/bbh"
models = os.listdir(OUTPUT_DIR)

if ".ipynb_checkpoints" in models:
    models.remove(".ipynb_checkpoints")


df = pd.read_json(os.path.join(OUTPUT_DIR, models[0], "output.jsonl"), lines=True)
df['task_qid'] = df['task'] +'_'+ df['qid'].astype(str)
base_df = pd.DataFrame({
    "qid": df["task_qid"],
    "question": df["question"],
    "answer": df['answer'] 
})


project = zeno_client.create_project(
    name="Gemini Evaluation - BBH",
    description="Evaluation of Gemini, GPT-4, and Mixtral on BBH dataset",
    view={
        "data": {
            "type": "text"
        },
        "label": {
            "type": "text"
        },
        "output": {
            "type": "text"
        }
    },
    public=True,
    metrics=[
        ZenoMetric(name="Accuracy Strict Match", type="mean", columns=["is_correct"]),
        ZenoMetric(name="Accuracy", type="mean", columns=["is_correct_last"])
    ],
)


project.upload_dataset(base_df, id_column="qid", data_column="question", label_column="answer")


def answer_type(answer):
    pattern = '[-+]?(?:[0-9,]*\.*\d+)'
    soln = re.findall(pattern, answer) 
    if answer.startswith('(') and answer.endswith(')'):
        return 'MCQ'
    elif answer == 'yes' or answer == 'Yes' or answer == 'No' or answer == 'no':
        return 'Yes/No'
    elif answer == 'true' or answer == 'True' or answer == 'False' or answer == 'false':
        return 'True/False'
    elif answer == 'valid' or answer == 'Valid' or answer == 'invalid' or answer == 'Invalid':
        return 'Valid/Invalid'
    elif len(soln) > 0:
        return 'Digit'
    else:
        return 'Other'

for model in models:
    df = pd.read_json(os.path.join(OUTPUT_DIR, model, "output.jsonl"), lines=True)
    df['task_qid'] = df['task'] +'_'+ df['qid'].astype(str)
    output_df = pd.DataFrame({
        "qid": df["task_qid"],
        "task": df["task"],
        "output": df.apply(lambda x: f"{x['generated_text']}\n\n{x['predict']}", axis=1),
        "output_last": df.apply(lambda x: f"{x['predict_last']}", axis=1),
        "output_type": df.apply(lambda x: f"{answer_type(x['answer'])}", axis=1),
        "question_length": df.apply(lambda x: len(x['question'].split(' ')), axis=1),
        "output_length": df.apply(lambda x: len(x['generated_text'].split(' ')), axis=1),
        "is_correct": df["is_correct"].astype(bool),
        "is_correct_last": df["is_correct_last"].astype(bool)
    })
    if model == 'gpt-4-1106-preview':
        model = 'gpt-4-turbo'
    project.upload_system(output_df, name=model, id_column="qid", output_column="output")