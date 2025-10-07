#! /bin/bash
# Runs all the experiments used in the main article for the mols project
# For instructive purposes, run each LLM-benchmark by itself rather than in a loop

for data in TYK2 D2R; do
    for bad_start in "" "--bad_start"; do
        for selector in Random Exploit UCB Thompson EIS EpsExploit LLMFat LLMFeaturiseQwen LLMFeaturiseGPT LLMWorkflow LLMWorkflowSimple; do
            python bin/benchmark.py --model GPRegOracle --selector ${selector}Selector --data data/$data.csv --batch_size 60 --initial_size 60 --num_repeats 10 $bad_start
        done
    done
done


for disable in ucb smarts tanimoto; do
    python bin/benchmark.py  --selector_kwargs '{"disable_${disable}": true}' --model GPRegOracle --selector LLMWorkflowSimpleSelector --data data/TYK2.csv --bad_start --batch_size 60 --initial_size 60 --num_repeats 10
done
for disable in ucb smarts tanimoto; do
    python bin/benchmark.py  --selector_kwargs '{"disable_${disable}": true}' --model GPRegOracle --selector LLMWorkflowSimpleSelector --data data/D2R.csv --batch_size 60 --initial_size 60 --num_repeats 10
done