#!/bin/bash


# result_dir="cache/gemini_pro_no_na"
# model="gemini-pro"
# result_dir="cache/gemini_pro_na"
# model="gemini-pro"
# result_dir="cache/mixtral_no_na"
# model="together_ai/DiscoResearch/DiscoLM-mixtral-8x7b-v2"
result_dir="cache/gpt4_no_na"
model="gpt-4-1106-preview"
instruction_path="src/webarena/agent/prompts/jsons/p_cot_id_actree_2s_no_na.json"

SERVER=""
OPENAI_API_KEY=""
OPENAI_ORGANIZATION=""
CONDA_ENV_NAME="webarena"
ENV_VARIABLES="export SHOPPING='http://${SERVER}:7770';export SHOPPING_ADMIN='http://${SERVER}:7780/admin';export REDDIT='http://${SERVER}:9999';export GITLAB='http://${SERVER}:8023';export MAP='http://miniserver1875.asuscomm.com:3000';export WIKIPEDIA='http://${SERVER}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing';export HOMEPAGE='http://${SERVER}:4399';export OPENAI_API_KEY=${OPENAI_API_KEY};export OPENAI_ORGANIZATION=${OPENAI_ORGANIZATION}"


# get the number of tmux panes
num_panes=$(tmux list-panes | wc -l)

# calculate how many panes need to be created
let "panes_to_create = 5 - num_panes"

# array of tmux commands to create each pane
tmux_commands=(
    'tmux split-window -h'
    'tmux split-window -v'
    'tmux select-pane -t 0; tmux split-window -v'
    'tmux split-window -v'
    'tmux select-pane -t 3; tmux split-window -v'
)

# create panes up to 5
for ((i=0; i<$panes_to_create; i++)); do
    eval ${tmux_commands[$i]}
done

#!/bin/bash

# Function to run a job
run_job() {
    tmux select-pane -t $1
    tmux send-keys "conda activate ${CONDA_ENV_NAME}; ${ENV_VARIABLES}; until python src/webarena/run.py --test_start_idx $2 --test_end_idx $3 --model ${model} --mode openai_chat_api --instruction_path ${instruction_path} --result_dir ${result_dir}; do echo 'crashed' >&2; sleep 1; done" C-m
    sleep 3
}

TOLERANCE=2
run_batch() {
    args=("$@") # save all arguments in an array
    num_jobs=${#args[@]} # get number of arguments

    for ((i=1; i<$num_jobs; i++)); do
        run_job $i ${args[i-1]} ${args[i]}
    done

    # Wait for all jobs to finish
    while tmux list-panes -F "#{pane_pid} #{pane_current_command}" | grep -q python; do
        sleep 100  # wait for 10 seconds before checking again
    done

    # Run checker
    while ! python src/webarena/scripts/check_error_runs.py ${result_dir} --delete_errors --tolerance ${TOLERANCE}; do
        echo "Check failed, rerunning jobs..."
        for ((i=1; i<$num_jobs; i++)); do
            run_job $i ${args[i-1]} ${args[i]}
        done

        # Wait for all jobs to finish
        while tmux list-panes -F "#{pane_pid} #{pane_current_command}" | grep -q python; do
            sleep 100  # wait for 10 seconds before checking again
        done
    done

}

run_batch 0 30 60 90 120
run_batch 120 150 180 210 240
run_batch 240 270 300 330 360
run_batch 360 390 420 450 480
run_batch 480 510 540 570 600
run_batch 600 630 660 690 720
run_batch 720 750 780 812

