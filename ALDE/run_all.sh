#! /bin/bash
# Runs all the experiments used in the main article for the ALDE project
# For instructive purposes

# All statistical models
for budget in 50 384; do
    if [ $budget -eq 50 ]; then
        n_pseudorand_init=10 # 10 + 50 = 60 <- total budget
        batch_size=10
    else
        n_pseudorand_init=96 # 96 + 384 = 480 <- total budget
        batch_size=96
    fi
    python execute_simulation.py --batch_size $batch_size --budget $budget --n_pseudorand_init $n_pseudorand_init --output_path results/${batch_size}_${budget}_simulations/
done

# Agents
for budget in 50 384; do
    if [ $budget -eq 50 ]; then
        n_pseudorand_init=10 # 10 + 50 = 60 <- total budget
        batch_size=10
    else
        n_pseudorand_init=96 # 96 + 384 = 480 <- total budget
        batch_size=96
    fi
    python execute_simulation.py --acq_fn AGENT SIMPLEAGENT --batch_size $batch_size --budget $budget --n_pseudorand_init $n_pseudorand_init --output_path results/${batch_size}_${budget}_agents/
done

# Standalone
for model in qwen3 deepseek gpt-5; do
    for blind in "" "-blind"; do
    for budget in 60 480; do # high budgets could cause deepseek to throw errors
        if [ $budget -eq 60 ]; then
            batch_size=10 # standalone runs does not start with any initial data (n_pseudorand_init=0)
        else
            batch_size=96
        fi
        python execute_standalone.py --model $model$blind --batch_size $batch_size --total_budget $budget --output_path results/${batch_size}_${budget}_standalone${blind}/ --runs 10
done

# SFT, sft/DPO models start with 10 initial pseudorandom sequences
python execute_standalone.py --type sft --model ./models/Qwen2-7B-SFT-0731 --n_init_pseudorandom 10 --batch_size 10 --total_budget 50 --output_path results/10_50_sft/ --runs 10
python execute_standalone.py --type sft --model ./models/Qwen2-7B-DPO-0731 --n_init_pseudorandom 10 --batch_size 10 --total_budget 50 --output_path results/10_50_dpo/ --runs 10
