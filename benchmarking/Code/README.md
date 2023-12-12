# Evaluation of code abilities

This code is mainly built upon [ODEX](https://github.com/zorazrw/odex) and [LiteLLM](https://github.com/BerriAI/litellm). We implemented additional code to support the [HumanEval](https://github.com/openai/human-eval) data format.

## How to evaluate ODEX (default)

```bash
python run_code.py
```

## How to evaluate ODEX HumanEval

```bash
python run_code.py --input_path data/HumanEval.jsonl --output_path outputs/HumanEval
```