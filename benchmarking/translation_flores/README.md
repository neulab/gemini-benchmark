# Benchmarking Large Language Models (GPT) for Machine Translation
## Overview
In this work, we investigate the translation capabilities of GPT models across 20 diverse languages  selected based on chr++ differentials from the [FLORES 200 dataset](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)

This codebase is derived from [ChatGPT-MT](https://arxiv.org/abs/2309.07423) and codebase.

Also see our [Zeno browser](), with interactive visualizations of our results.

We have outputs for 6 systems:
  - ChatGPT (0-shot prompts) (gpt-3.5-turbo-1106): 20 target languages
  - ChatGPT (5-shot prompts) (gpt-3.5-turbo-1106): 20 target languages
  - GPT-4 (5-shot prompts) (gpt-4-1106-preview):  20 target languages
  - NLLB-MOE: 20 target languages
  - Google Translate: 20 target languages
  - Gemini: 20 target languages
  
  
## Reproducing the work
We used *gpt-3.5-turbo-1106*, *gpt-4-1106-preview*, and *gemini-pro* in Dec 2023. 

Find instructions below on how to use our codebase.

### Outputs and inputs
The outputs and inputs from this work is a tsv contains 3 columns:
- messages : Contains the prompts
- label : Is the reference
- predictions : The predictions from Open AI.

###  Querying OpenAI
This section has instructions on how to use our codebase to run the experiments.
- Download raw model inputs which can be found on [Zenodo](https://zenodo.org/records/8286649). 
- The input data is in text format and includes the prompts. Use the `create_data.py` script to create the processed input data for the experiments. Point `source_dir` to the appropriate prompt folder in the downloaded data and `target_dir` to the folder where you want to save the data.
- You will need to install Zeno and OpenAI libraries. Install them and other requirements by running `pip install -r requirements.txt`
- config.py contains the configuration for the models, change the model names and sampling parameters as needed.
- modelling.py: This script contains utilities. An important one is the call to `generate_from_chat_prompt` function. You may want to reduce the `requests_per_minute` parameter value especially for n-shot and non-latin scripts so as not to max out and get empty returns from the API.
- flores200_utils.py : Contains data processing utilities for the Flores dataset
- Have a file called `langs.txt` that contains the languages you want to run inference on. 
- Your source folders (created by `create_data.py` will  be named [prompt]/[lang]/ with the `inputs` and `refs` subdirectories inside.
- run.sh: This is the bash script that launches main.py. Run `bash run.sh`

### Evaluation
We have a script `eval_runs.py` that handles evaluation for BLEU, chrF, SLR and TER.
`python eval_runs.py --results_dir [folder] --langs_file [a file with line separated languages to be evaluated]  --tokenizer [tokenizer-this is optional]`
### Notebooks
*langid_classifier.ipynb* - for classifying the langauge of the predictions
*visualization/translation_flores/zeno_upload.py* - This script shows how to upload results to the Zeno library and visualize them.

