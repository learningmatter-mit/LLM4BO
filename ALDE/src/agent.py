"""
LLM Agent-based Active Learning Selector Module

This module implements an intelligent active learning selector that uses Large Language Models (LLMs)
to analyze protein sequence space and select the most promising sequences for labeling. The selector employs
a multi-agent system with strategy generation and implementation phases.

The module provides:
- Strategy generation using LLM analysis of current AL state
- Multi-agent implementation of different selection strategies
- Structured output validation for reliable selection results

Key Components:
- StrategyState: TypedDict for managing AL strategy workflow
- StrategyOutput: Pydantic model for strategy generation output
- init_selector_builder: Main function to create selector workflow

Dependencies:
- langchain_anthropic: For LLM interactions
- langgraph: For multi-agent workflow orchestration
- pandas/numpy: For data manipulation and analysis
"""

import pandas as pd
import numpy as np
import os, getpass, re, yaml, json, re 
import warnings
import time, random
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import traceback
warnings.filterwarnings('ignore')

from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from typing import TypedDict, List, Annotated, Optional, Tuple, Any, Dict
import operator
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
import anthropic
from langchain_openai import ChatOpenAI

# Import prompt and summary functions from separate module
from src.prompts import (
    get_prompts, 
    save_agent_strategies,
    update_agent_performance,
    compact_df  # For formatting data displays
)

# Import implementer from separate module
from src.implementer import init_implementer_builder

def retry_node(fn):
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff
        stop=stop_after_attempt(5),  # Retry up to 5 times
        retry=retry_if_exception_type(anthropic._exceptions.OverloadedError)
        )
    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapped

def _set_env(var: str):
    """Set environment variable if not already set.
    
    Args:
        var: Environment variable name to set
    """
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("LANGSMITH_API_KEY")
os.environ['LANGSMITH_TRACING'] = "true"
os.environ["LANGSMITH_PROJECT"] = "ALDE"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

class StrategyState(TypedDict):
    AL_task: str
    report: str
    final_selection : List[List[int]]
    selected_indices: Annotated[List[List[int]], operator.add]
    strategies : List[str]
    analysis : str
    reports : Annotated[List[str], operator.add]
    batch_size : int
    n_iter : int

class StrategyOutput(BaseModel):
    strategies: List[str] = Field(description="List of strategies to implement including all details, criteria and number of sequences to select.")
    analysis : str = Field(description="Analysis of the current state of the AL campaign and rationale for the strategies")

def init_selector_builder(implement_builder, selection_prompt, pred_df : pd.DataFrame):
    """Initialize the main selector builder with strategy generation and implementation workflow.
    
    Creates a StateGraph that orchestrates the complete AL selection process:
    1. Strategy generation using LLM analysis
    2. Parallel implementation of multiple strategies
    3. Final summarization and selection consolidation
    
    Args:
        implement_builder: Compiled implementer workflow graph
        selection_prompt: Formatted prompt template for strategy implementation
        pred_df: Prediction DataFrame with sequence, predictions, and std_predictions
    Returns:
        StateGraph: Compiled selector workflow graph
        
    Example:
        >>> builder = init_selector_builder(implement_builder, selection_prompt, pred_df)
        >>> result = builder.compile().invoke({"AL_task": "...", "batch_size": 10})
    """
    """"llm_strategy = ChatOpenAI(
            openai_api_key=os.environ.get('LAMBDA_API_KEY'),
            openai_api_base="https://api.lambda.ai/v1",
            model_name='qwen3-32b-fp8',
            max_tokens=32000,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}}
        )"""
    llm_strategy = ChatOpenAI(
        openai_api_key=os.environ.get('OPENAI_API_KEY'),
        model_name='gpt-5',
        max_tokens=32000,
        reasoning={"effort": "medium"}
    )
    llm_selection = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)

    strategy_system_prompt = """/think Show your reasoning process clearly.

EXAMPLE OUTPUT, follow this format, but freely create your own analysis and strategies: 

**ANALYSIS:**
[Provide situation-specific analysis of current state, progress, key findings, and hypotheses for the next cycle]

**STRATEGIES:**
[Generate a suitable number of strategies tailored to the specific problem and data, output a clear list with a clear number of sequences to select for each strategy]
"""
    def generate_strategy(state:StrategyState):
        """Generate selection strategies using LLM analysis.
        
        Analyzes the current AL state and generates multiple diverse strategies
        for compound selection based on the provided task description.
        
        Args:
            state: StrategyState containing AL task and batch size
            
        Returns:
            dict: Generated strategies and analysis of current AL state
        """
        response = llm_strategy.invoke([HumanMessage(content=state['AL_task'])] + [SystemMessage(content=strategy_system_prompt)])
        structured_llm_output = llm_selection.with_structured_output(StrategyOutput)
        print(response.content)
        sysmsg = SystemMessage(content="Extract the strategies and analysis from the following text. Do not include any other text in your response. Ensure each strategy clearly states number of selections to be made.")
        response = structured_llm_output.invoke([sysmsg, HumanMessage(content=response.content)])
        return {"strategies": response.strategies, "analysis": response.analysis}
    
    
    def strategy_iterator(state : StrategyState):
        """Route strategies to parallel implementers.
        Iteratively assign strategies to implementers until all strategies are implemented.
        
        Args:
            state: StrategyState containing generated strategies
        Returns:
            list: List of Send objects for parallel strategy implementation
        """
        n_done = len(state['reports'])
        prev_selected = [item for sublist in state['selected_indices'] for item in sublist]
        if n_done < len(state['strategies']):
            return Send("do_implementation", {
                "messages": [HumanMessage(content=selection_prompt.format(strategy=str(state['strategies'][n_done])))], 
                "strategy": state['strategies'][n_done], "prev_selected": prev_selected})
        elif len(prev_selected) < state['batch_size']:
            strategies_text = '\n'.join([f"Strategy {i+1}: {strategy}" for i, strategy in enumerate(state['strategies'])])
            strategy = f"""Your colleages have selected {len(prev_selected)} compounds by the following stratgies: 
        {strategies_text}

        Based on your colleages' strategies, select exactly{state['batch_size'] - len(prev_selected)} more compounds. Make sure there are no duplicates. You may release any constraints on the selection to achieve this. The goal is to find high fitness sequences."""

            implementer = Send("do_implementation", {
                "messages": [HumanMessage(content=selection_prompt.format(strategy=str(strategy)))], 
                "strategy": "Other strategies didnt meet the batch size. Enforce batch size by selecting more compounds.",
                "prev_selected": prev_selected})
            return implementer
        return "summarise_implementers"

    @retry_node
    def summarise_implementers(state : StrategyState):
        """Summarize and consolidate results from all implementers.
        
        Aggregates reports and selected indices from all strategy implementers, then prompts the LLM
        to make a final selection and write a concise summary report. Ensures batch size and uniqueness constraints.
        
        Args:
            state: StrategyState containing reports, strategies, and selected indices
            
        Returns:
            dict: Final report and list of selected compound indices
        """
        reports = state['reports']
        strategies = state['strategies']
        if len(reports) > len(strategies):
            strategies.append("Other strategies didnt meet the batch size. Enforce batch size by selecting more compounds.")
        analysis = state['analysis']
        prev_selected = [item for sublist in state['selected_indices'] for item in sublist]
        unique_selected_indices = list(set(prev_selected))
        if len(unique_selected_indices) != len(prev_selected):
            print(f"Warning: Duplicate indices in selected_indices: {prev_selected}")
            print(f"Duplicate indices: {list(set(prev_selected).difference(set(unique_selected_indices)))}")
        stratsum = '\n'.join([
            f"Strategy {i+1}: {strategy}\nReport {i+1}: {report}\n" 
            for i, (strategy, report) in enumerate(zip(strategies, reports))
        ])
        prompt = f"""You are an expert protein scientist tasked with summarizing a directed evolution acquisition step in an Active Learning (AL) cycle. Multiple expert chemists have independently selected sequences using different strategies based on previous experimental results. Some have succeded with implementing their strategies, others may have failed.
## Context
<current_AL_cycle_analysis>
{analysis}
</current_AL_cycle_analysis>

<individual_expert_strategies_and_selection_rationales>
{stratsum}
</individual_expert_strategies_and_selection_rationales>

## Task
- Write in Markdown format
- Maximum 200 words
- Focus on actionable insights and strategic reasoning that can be used to inform future cycles. 
- List all hypotheses explored and all comments related to them.
- Clearly state if an implementer failed to meet batch size or had to relax constraints to meet batch size. THESE ARE CONSIDERED FAILURES.
- Output only the final report (no preamble or explanations)

<report_structure>
# AL Campaign Acquisition Summary
## Implementation
[Highlight most important compound choices and if the selection targets were achieved]

## Campaign Impact
[How selections advance AL objectives, what was exploit/explore split?]
</report_structure>
"""
        validated = llm_selection.invoke([SystemMessage(content=prompt)] + [HumanMessage(content="Write a report based upon these memos.")])
        return {"report" : validated.content, "final_selection" : state['selected_indices'], "n_iter" : state['n_iter']}
    
    builder = StateGraph(StrategyState)
    builder.add_node("generate_strategy", generate_strategy)
    builder.add_node("do_implementation", implement_builder.compile())
    builder.add_node("summarise_implementers", summarise_implementers)
    builder.add_edge(START, "generate_strategy")
    builder.add_conditional_edges("generate_strategy", strategy_iterator, ['do_implementation'])
    builder.add_conditional_edges("do_implementation", strategy_iterator, ['do_implementation', 'summarise_implementers'])
    builder.add_edge("summarise_implementers", END)

    return builder

def _build_complete_training_data(train_df: pd.DataFrame, summaries_file: str) -> pd.DataFrame:
    """Build complete training data by collecting all validated sequences from summaries.
    
    Args:
        train_df: Current training data from acquisition (used as fallback)
        summaries_file: Path to JSON file containing historical summaries
        
    Returns:
        Complete training DataFrame with all historical sequences and fitness values
    """
    from src.prompts import load_agent_summaries
    
    # Load historical summaries
    summaries = load_agent_summaries(summaries_file)
    
    all_sequences = []
    all_fitness = []
    
    # Collect all validated sequences from all cycles
    for cycle_data in summaries:
        if cycle_data and 'strategies' in cycle_data:
            for strategy_data in cycle_data['strategies']:
                if 'selected' in strategy_data and 'validated_fitness' in strategy_data:
                    selected_seqs = strategy_data['selected']
                    validated_fit = strategy_data['validated_fitness']
                    
                    # Only include sequences that have validated fitness
                    for seq, fit in zip(selected_seqs, validated_fit):
                        if seq and fit is not None:
                            all_sequences.append(seq)
                            all_fitness.append(fit)
    
    # Create complete training DataFrame
    if all_sequences:
        complete_train_df = pd.DataFrame({
            'sequence': all_sequences,
            'fitness': all_fitness
        })
        # Remove duplicates, keeping the last occurrence (most recent validation)
        complete_train_df = complete_train_df.drop_duplicates(subset=['sequence'], keep='last')
    else:
        # Fallback to the passed train_df if no historical data found
        complete_train_df = train_df.copy()
    
    return complete_train_df


def run_selector_agent(oracle_model : str, batch_size : int, total_cycles : int, cycle : int, train_df : pd.DataFrame, pred_df : pd.DataFrame, protein : str, summaries_file : str = 'summaries.json', include_sequences : bool = True):
    """Run the full LLM-based selector agent workflow for active learning.
    
    Orchestrates the entire AL selection process by generating prompts, initializing builders,
    and executing the multi-agent workflow. Returns the final selected indices and a summary report.
    Also saves the cycle summary for future reference.
    
    Args:
        oracle_model: Name of the oracle model
        batch_size: Number of compounds to select in this cycle
        total_cycles: Total number of AL cycles
        cycle: Current cycle number
        train_df: Training data DataFrame (from acquisition interface)
        pred_df: Prediction DataFrame
        protein: Name of the target protein
        summaries_file: Path to JSON file for storing cycle summaries
        include_sequences: Whether to include sequence data in prompts (default True)
        
    Returns:
        tuple: (final_selection, summary, strategies) where final_selection is a list of selected indices, 
               summary is a markdown report, and strategies is a list of implemented strategies
    """
    # if cycle = 1, empty summaries_file
    if cycle == 1:
        with open(summaries_file, 'w') as f:
            json.dump([], f)
    else:
        # Update summary with performance metrics using new function
        update_agent_performance(cycle-1, train_df, summaries_file, include_sequences)

    # Build complete training data from historical summaries
    complete_train_df = _build_complete_training_data(train_df, summaries_file)
    
    strategy_prompt, selection_prompt, implementer_system_prompt = get_prompts(
        oracle_model, batch_size, total_cycles, cycle, complete_train_df, pred_df, protein, summaries_file, include_sequences
    )

    implement_builder = init_implementer_builder(complete_train_df, pred_df, implementer_system_prompt)
    selector_builder = init_selector_builder(implement_builder, selection_prompt, pred_df)

    memory = MemorySaver()
    graph = selector_builder.compile(checkpointer=memory)
    thread = {"configurable": {"thread_id": "1"}}
    input = {'AL_task': strategy_prompt, 'batch_size': batch_size, 'n_iter': 0}
    MAX_RETRIES, BASE_DELAY = 5, 30
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = graph.invoke(input, thread, stream_mode="values")
            break  # success, exit loop
        except Exception as e:
            print(f"[Attempt {attempt}] graph.invoke failed: {e}")
            print(f"Trace: {traceback.format_exc()}")
            if attempt == MAX_RETRIES:
                raise  # re-raise the last exception
            # Exponential backoff with jitter
            delay = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            time.sleep(delay)
    
    try: 
        # Save the agent strategies and predictions immediately after selection
        save_agent_strategies(cycle, response['strategies'], response['final_selection'], 
                            pred_df, response['report'], summaries_file)
    except Exception as e:
        print(f"Error saving agent strategies and predictions: {e}")
        
    if len(response['final_selection']) < batch_size:
        print(f"Warning: Selected {len(response['final_selection'])} sequences instead of {batch_size}")
        print(f"Selected indices: {response['final_selection']}")
        print(f"Response: {response}")
    return response['final_selection'], response['report'], response['strategies']

# Export functions for external use
__all__ = [
    'run_selector_agent',
    'init_selector_builder'
]