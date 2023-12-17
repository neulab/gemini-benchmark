# Gemini Benchmark

by [CMU](https://cmu.edu), [Zeno](https://zenoml.com), and [BerriAI LiteLLM](https://github.com/BerriAI/litellm)

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
