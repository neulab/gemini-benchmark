# Evaluation of MMLU

This code is mainly built upon [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub) and [LiteLLM](https://github.com/BerriAI/litellm). We implemented additional code to support the original [MMLU](https://github.com/hendrycks/test) data format.

## How to evaluate MMLU with few-shot prompting (default)

```bash
python run_mmlu.py
```

## How to evaluate MMLU with chain-of-thought prompting

```bash
python run_mmlu.py --cot
```