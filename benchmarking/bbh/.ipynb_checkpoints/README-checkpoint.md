# Evaluation of BBH Tasks

This code is mainly built upon [EleutherAI](https://github.com/EleutherAI/lm-evaluation-harness/issues?q=gsm8k) and [LiteLLM](https://github.com/BerriAI/litellm). We implemented additional code to support the original [BBH](https://github.com/suzgunmirac/BIG-Bench-Hard) data format.


## How to evaluate BBH with chain-of-thought prompting

```bash
python bbh.py --task object_counting --model gpt-3.5-turbo
```