import os

import dotenv
import pandas as pd
from zeno_client import ZenoClient, ZenoMetric

dotenv.load_dotenv("../.env", override=True)

OUTPUT_DIR = "../../outputs/mawpsmultiarith"

models = os.listdir(OUTPUT_DIR)

# Upload base dataset
df = pd.read_json(os.path.join(OUTPUT_DIR, models[0], "output.jsonl"), lines=True)
base_df = pd.DataFrame(
    {"qid": df["qid"], "question": df["question"], "answer": df["answer"].astype(str)}
)

zeno_client = ZenoClient(os.environ.get("ZENO_API_KEY"))

project = zeno_client.create_project(
    name="Gemini Evaluation - MawpsMultiArith",
    description="Evaluation of Gemini, GPT-4, and Mixtral on MawpsMultiArith dataset",
    view={
        "data": {"type": "text"},
        "label": {"type": "text"},
        "output": {"type": "markdown"},
    },
    public=True,
    metrics=[
        ZenoMetric(name="Accuracy", type="mean", columns=["is_correct"]),
    ],
)

project.upload_dataset(
    base_df, id_column="qid", data_column="question", label_column="answer"
)

for model in models:
    df = pd.read_json(os.path.join(OUTPUT_DIR, model, "output.jsonl"), lines=True)
    output_df = pd.DataFrame(
        {
            "qid": df["qid"],
            "output": df.apply(
                lambda x: f"{x['generated_text']}\n\n**{x['predict']}**", axis=1
            ),
            "is_correct": df["is_correct"].astype(bool),
        }
    )
    project.upload_system(
        output_df, name=model, id_column="qid", output_column="output"
    )
