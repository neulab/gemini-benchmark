# An In-depth Look at Gemini's Language Abilities
Repo for the paper [An In-depth Look at Gemini's Language Abilities](https://arxiv.org/abs/2312.11444)
by [CMU](https://cmu.edu), [Zeno](https://zenoml.com), and [BerriAI LiteLLM](https://github.com/BerriAI/litellm)

In this paper, we do an in-depth exploration of Gemini's language abilities, making two contributions: 
- We provide a third-party, objective comparison of the abilities of the OpenAI GPT and Google Gemini models with reproducible code and fully transparent results. 
- we take a closer look at the results, identifying areas where one of the two model classes excels.

We perform this analysis over 10 datasets testing a variety of language abilities, including reasoning, answering knowledge-based questions, solving math problems, translating between languages, generating code, and acting as instruction-following agents. From this analysis, we find that:

- Gemini Pro achieves accuracy that is close but slightly inferior to the corresponding GPT 3.5 Turbo on all tasks that we benchmarked.
- Gemini fails in mathematical reasoning with many digits, and is sensitive to multiple-choice answer ordering, aggressive content filtering, and others.
- Gemini demonstrates comparably high performance in areas such as generation into non-English languages, handling longer and more complex reasoning chains, and word sorting/rearrangement problems.

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
