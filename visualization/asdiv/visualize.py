from zeno_client import ZenoClient, ZenoMetric
import pandas as pd
import os
import dotenv

dotenv.load_dotenv(override=True)
zeno_client = ZenoClient("ZENO_API_KEY")

OUTPUT_DIR = "../../gemini-benchmark/outputs/asdiv"

models = os.listdir(OUTPUT_DIR)

if ".ipynb_checkpoints" in models:
    models.remove(".ipynb_checkpoints")

def check_fraction(num):
    if num % 1 != 0:
        return 1
    return 0

df = pd.read_json(os.path.join(OUTPUT_DIR, models[0], "output.jsonl"), lines=True)
base_df = pd.DataFrame({
    "qid": df["qid"],
    "question": df["question"],
    "answer": df["answer"].astype(str) 
})



project = zeno_client.create_project(
    name="Gemini Evaluation - ASDIV",
    description="Evaluation of Gemini, GPT-4, and Mixtral on ASDIV dataset",
    view={
        "data": {
            "type": "text"
        },
        "label": {
            "type": "text"
        },
        "output": {
            "type": "markdown"
        }
    },
    public=True,
    metrics=[
        ZenoMetric(name="Accuracy Strict Match", type="mean", columns=["is_correct"]),
        ZenoMetric(name="Accuracy", type="mean", columns=["is_correct_last"])
    ],
)


project.upload_dataset(base_df, id_column="qid", data_column="question", label_column="answer")


for model in models:
    df = pd.read_json(os.path.join(OUTPUT_DIR, model, "output.jsonl"), lines=True)
    if model == 'mixtral':
        output_df = pd.DataFrame({
            "qid": df["qid"],
            "output": df.apply(lambda x: f"{x['generated_text']}\n\n**{x['predict']}**", axis=1),
            "answer_str": df.apply(lambda x: str(x['answer']), axis=1),
            "fraction": df.apply(lambda x: check_fraction(float(x['answer'])), axis=1),
            "numeric_answer": df.apply(lambda x: float(x['answer']), axis=1),
            "question_length": df.apply(lambda x: len(x['question'].split(' ')), axis=1),
            "output_length": df.apply(lambda x: len(x['generated_text'].split('Q:')[0].split(' ')), axis=1),
            "is_correct": df["is_correct"].astype(bool),
            "is_correct_last": df["is_correct_last"].astype(bool)
        })
    else:
        output_df = pd.DataFrame({
            "qid": df["qid"],
            "output": df.apply(lambda x: f"{x['generated_text']}\n\n**{x['predict']}**", axis=1),
            "answer_str": df.apply(lambda x: str(x['answer']), axis=1),
            "fraction": df.apply(lambda x: check_fraction(float(x['answer'])), axis=1),
            "numeric_answer": df.apply(lambda x: float(x['answer']), axis=1),
            "question_length": df.apply(lambda x: len(x['question'].split(' ')), axis=1),
            "output_length": df.apply(lambda x: len(x['generated_text'].split(' ')), axis=1),
            "is_correct": df["is_correct"].astype(bool),
            "is_correct_last": df["is_correct_last"].astype(bool)
        })
    if model == 'gpt-4-1106-preview':
        model = 'gpt-4-turbo'
    project.upload_system(output_df, name=model, id_column="qid", output_column="output")