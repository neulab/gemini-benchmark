# An In-depth Look at Gemini's Language Abilities
Repo for the paper [An In-depth Look at Gemini's Language Abilities](https://arxiv.org/abs/2312.11444)
by [CMU](https://cmu.edu), [Zeno](https://zenoml.com), and [BerriAI LiteLLM](https://github.com/BerriAI/litellm)

In this paper, we do an in-depth exploration of [Google Gemini](https://blog.google/technology/ai/google-gemini-ai/)'s language abilities, making two contributions: 
- We provide a third-party, objective comparison of the abilities of the OpenAI GPT and Google Gemini models with reproducible code and fully transparent results. 
- we take a closer look at the results, identifying areas where one of the two model classes excels.

## Results

We perform this analysis over 10 datasets testing a variety of language abilities, including reasoning, answering knowledge-based questions, solving math problems, translating between languages, generating code, and acting as instruction-following agents. From this analysis, we find that (as of this writing on December 18th, 2023):

- Gemini Pro achieves accuracy that is close but slightly inferior to the corresponding GPT 3.5 Turbo on all tasks that we benchmarked.
- Gemini fails in mathematical reasoning with many digits, and is sensitive to multiple-choice answer ordering, aggressive content filtering, and others.
- Gemini demonstrates comparably high performance in areas such as generation into non-English languages, handling longer and more complex reasoning chains, and word sorting/rearrangement problems.

The overall results table can be found below:

| Task                          | Dataset             | Gemini Pro | GPT 3.5 Turbo | GPT 4 Turbo | Mixtral |
|-------------------------------|---------------------|------------|---------------|-------------|---------|
| **Knowledge-based QA**        | **MMLU (5-shot)**   | 64.12      | 67.75         | **80.48**   | -       |
|                               | **MMLU (CoT)**      | 60.63      | 70.07         | **78.95**   | -       |
| **Reasoning**                 | **BIG-Bench-Hard**  | 65.58      | 71.02         | **83.90**   | 41.76   |
| **Mathematics**               | **GSM8K**           | 69.67      | 74.60         | **92.95**   | 58.45   |
|                               | **SVAMP**           | 79.90      | 82.30         | **92.50**   | 73.20   |
|                               | **ASDIV**           | 81.53      | 86.69         | **91.66**   | 74.95   |
|                               | **MAWPS**           | 95.33      | **99.17**     | 98.50       | 89.83   |
| **Code Generation**           | **HumanEval**       | 52.44      | 65.85         | **73.17**   | -       |
|                               | **ODEX**            | 38.27      | 42.60         | **46.01**   | -       |
| **Machine Translation**       | **FLORES (0-shot)** | 29.59      | 37.50         | **46.57**   | -       |
|                               | **FLORES (5-shot)** | 29.00      | 38.08         | **48.60**   | -       |
| **Web Agents**                | **WebArena**        | 7.09       | 8.75          | **15.16**   | 1.37    |


You can find more details on results from each task, and comprehensive at each of the below links:

- [Knowledge-based QA](https://hub.zenoml.com/report/2674/Gemini%20MMLU) (MMLU)
- [Reasoning](https://hub.zenoml.com/report/2575/Gemini%20BBH) (BIG-Bench Hard)
- [Mathematics](https://hub.zenoml.com/report/2773/Gemini%20Mathematics) (GSM8K, SVAMP, ASDIV, MAWPS)
- [Code Generation](https://hub.zenoml.com/report/2641/Gemini%20Code) (HumanEval, ODEX)
- [Machine Translation](https://hub.zenoml.com/report/2740/Gemini%3A%20Flores%20Translation%20Evaluation) (FLORES)
- [Web Navigation Agents](https://hub.zenoml.com/report/2608/Gemini%20Webarena) (WebArena)


## File Structure

- `/outputs/{dataset}/{model}`: contains the outputs of the systems, separated by dataset and model
- `/benchmarking/{dataset}`: contains the code for benchmarking, separated by dataset
- `/visualization`: contains the code for visualization, possibly separated by task type

## Setup

Create a `.env` file in the root of the repository with your Zeno API key:

```
ZENO_API_KEY=your_api_key
```

This is loaded by `dotenv` in the visualization files.
