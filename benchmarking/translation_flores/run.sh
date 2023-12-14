#!/bin/bash

prompt_datasets=("tt-zero")
input_file=langs.txt
for prompt in "${prompt_datasets[@]}"
do
    while IFS= read -r line; 
    do
    # Remove the newline character from the line
    	lang=$(echo "$line" | tr -d '\n')
    # Do something with the line
    	echo "Processing lang: $lang"
        python main.py   --dataset "./main_texts/$prompt/$lang" --dataset_config_preset "local-${lang:0:3}"
    done< "$input_file"
done
