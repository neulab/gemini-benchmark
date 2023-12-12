# Evaluation of MATH Tasks

This code is mainly built upon [EleutherAI](https://github.com/EleutherAI/lm-evaluation-harness/issues?q=gsm8k) and [LiteLLM](https://github.com/BerriAI/litellm). The script handles 4 math tasks (math_reasoning.py):

* GSM-8K
* SVAMP
* ASDIV
* MAWPS


## How to evaluate BBH with chain-of-thought prompting

```bash
python math_reasoning.py --task gsm8k --model gpt-3.5-turbo
```