from zeno_client import ZenoClient, ZenoMetric
import pandas as pd
import os
import dotenv

dotenv.load_dotenv("../.env", override=True)

dataset = "HumanEval"
OUTPUT_DIR = f"../../outputs/{dataset}"

models = os.listdir(OUTPUT_DIR)

# Upload base dataset
df = pd.read_json(os.path.join(OUTPUT_DIR, models[0], "output.jsonl"), lines=True)
base_df = pd.DataFrame(
    {
        "qid": df.index,
        "question": df["prompt"],
        "answer": df["canonical_solution"],
    }
)

zeno_client = ZenoClient(os.environ.get("ZENO_API_KEY"))

project = zeno_client.create_project(
    name=f"Gemini Evaluation - {dataset}",
    description=f"Evaluation of Gemini, GPT-4, and Mixtral on {dataset} dataset",
    view={
        "data": {"type": "code"},
        "label": {"type": "code"},
        "output": {
            "type": "vstack",
            "keys": {"output": {"type": "code"}, "result": {"type": "text"}},
        },
    },
    public=True,
    metrics=[
        ZenoMetric(name="Pass@1", type="mean", columns=["correct"]),
    ],
)

project.upload_dataset(
    base_df, id_column="qid", data_column="question", label_column="answer"
)


def process_row(row):
    return {
        "output": row["predictions"][0],
        "result": row["output"][0][1]["result"],
    }


for model in models:
    df = pd.read_json(os.path.join(OUTPUT_DIR, model, "output.jsonl"), lines=True)
    output_df = pd.DataFrame(
        {
            "qid": df.index,
            "output": df.apply(process_row, axis=1),
            "correct": df["scores"].apply(lambda x: bool(x["pass@1"])),
        }
    )
    project.upload_system(
        output_df, name=model, id_column="qid", output_column="output"
    )
