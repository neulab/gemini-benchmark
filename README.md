# An In-depth Look at Gemini's Language Abilities
Repo for the paper [An In-depth Look at Gemini's Language Abilities](https://arxiv.org/abs/2312.11444)
by [CMU](https://cmu.edu), [Zeno](https://zenoml.com), and [BerriAI LiteLLM](https://github.com/BerriAI/litellm)

In this paper, we do an in-depth exploration of [Google Gemini](https://blog.google/technology/ai/google-gemini-ai/)'s language abilities, making two contributions: 
- We provide a third-party, objective comparison of the abilities of the OpenAI GPT and Google Gemini models with reproducible code and fully transparent results. 
- we take a closer look at the results, identifying areas where one of the two model classes excels.

## Results

We perform this analysis over 10 datasets testing a variety of language abilities, including reasoning, answering knowledge-based questions, solving math problems, translating between languages, generating code, and acting as instruction-following agents. From this analysis, we find that (as of this writing on December 18th, 2023):

- Gemini's Pro model achieved comparable but slightly inferior accuracy compared to the current version of OpenAI's GPT 3.5 Turbo for all English tasks, but superior ability to translate into other languages.
- Gemini fails in mathematical reasoning with many digits, and is sensitive to multiple-choice answer ordering, and others.
- Gemini demonstrates comparably high performance in areas such as generation into non-English languages, handling longer and more complex reasoning chains, and word sorting/rearrangement problems.

The overall results table can be found below:

| **Task**                     | **Dataset**             | Gemini Pro | GPT 3.5 Turbo | GPT 4 Turbo | Mixtral |
|------------------------------|-------------------------|------------|---------------|-------------|---------|
| **Knowledge-based QA**       | MMLU (5-shot)           | 65.22      | 67.75         | **80.48**   | *68.81* |
|                              | MMLU (CoT)              | 62.09      | *70.07*       | **78.95**   | 59.57   |
| **Reasoning**                | BIG-Bench-Hard          | 67.53      | *71.02*       | **83.90**   | 60.76   |
| **Mathematics**              | GSM8K                   | 76.42      | *78.01*       | **92.72**   | 71.65   |
|                              | SVAMP                   | 81.10      | *82.30*       | **92.60**   | 81.60   |
|                              | ASDIV                   | 85.31      | *89.07*       | **92.75**   | 83.16   |
|                              | MAWPS                   | 96.50      | *98.00*       | **98.67**   | 96.00   |
| **Code Generation**          | HumanEval               | 59.76      | *74.39*       | **76.83**   | 45.12   |
|                              | ODEX                    | 39.86      | **52.62**     | *45.79*     | 40.55   |
| **Machine Translation**      | FLORES (5-shot) Unblocked| *56.14*   | 55.78         | **57.15**   | 44.27   |
|                              | FLORES (5-shot) All     | 22.83      | *43.12*       | **51.63**   | 33.45   |
| **Web Agents**               | WebArena                | 7.12       | *8.87*        | **14.90**   | 1.39    |


You can find more details on results from each task, and comprehensive analysis at each of the below links:

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
