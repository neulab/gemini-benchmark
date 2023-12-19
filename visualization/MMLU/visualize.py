from zeno_client import ZenoClient, ZenoMetric
import pandas as pd
import os
import dotenv

dotenv.load_dotenv("../.env", override=True)

OUTPUT_DIR = "../../outputs/MMLU"

models = os.listdir(OUTPUT_DIR)

# Upload base dataset
df = pd.read_json(os.path.join(OUTPUT_DIR, models[0], "output.jsonl"), lines=True)
base_df = pd.DataFrame(
    {
        "task": df["task"],
        "qid": df.index,
        "question": df["prompt"].apply(lambda x: x.split("\n\n")[-1]),
        "answer": df["label"],
    }
)

zeno_client = ZenoClient(os.environ.get("ZENO_API_KEY"))

project = zeno_client.create_project(
    name="Gemini Evaluation - MMLU",
    description="Evaluation of Gemini-Pro, GPT-4, and GPT-3.5 on MMLU dataset",
    view={
        "data": {"type": "text"},
        "label": {"type": "text"},
        "output": {"type": "text"},
    },
    public=True,
    metrics=[
        ZenoMetric(name="Accuracy", type="mean", columns=["correct"]),
        ZenoMetric(name="Answered", type="mean", columns=["answered"]),
    ],
)

project.upload_dataset(
    base_df, id_column="qid", data_column="question", label_column="answer"
)

for model in models:
    df = pd.read_json(os.path.join(OUTPUT_DIR, model, "output.jsonl"), lines=True)
    output_df = pd.DataFrame(
        {
            "qid": df.index,
            "output": df["prediction"] if "cot" not in model else df["response"],
            "output_length": df["prediction"].apply(lambda x: len(x))
            if "cot" not in model
            else df["response"].apply(lambda x: len(x)),
            "answered": df["prediction"].apply(lambda x: int(bool(len(x))))
            if "cot" not in model
            else df["response"].apply(lambda x: int(bool(len(x)))),
            "correct": df["correct"].astype(bool),
        }
    )
    project.upload_system(
        output_df, name=model, id_column="qid", output_column="output"
    )
