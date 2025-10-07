"""
LLM Agent-based Active Learning Selector Module

This module implements an intelligent active learning selector that uses Large Language Models (LLMs)
to analyze chemical space and select the most promising compounds for labeling. The selector employs
a multi-agent system with strategy generation and implementation phases.

The module provides:
- Strategy generation using LLM analysis of current AL state
- Multi-agent implementation of different selection strategies
- Chemical substructure search capabilities using SMARTS queries
- Molecular similarity analysis using Tanimoto coefficients
- Structured output validation for reliable selection results

Key Components:
- FinalSelectionOutput: Pydantic model for structured selection output
- Implementer: State management for strategy implementation
- StrategyState: TypedDict for managing AL strategy workflow
- StrategyOutput: Pydantic model for strategy generation output

Dependencies:
- langchain_anthropic: For LLM interactions
- langgraph: For multi-agent workflow orchestration
- rdkit: For molecular fingerprinting and substructure search
- pandas/numpy: For data manipulation and analysis
"""

import pandas as pd
import numpy as np
import sys, os, getpass, yaml, json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import TanimotoSimilarity


from langchain_core.tools import tool, InjectedToolArg, BaseTool
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from typing import TypedDict, List, Annotated, operator, Optional, Tuple, Any, Dict
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from utils.APIUtils import with_retry

from alselectors.implementer import init_implementer_builder

def _set_env(var: str):
    """Set environment variable if not already set.
    
    Args:
        var: Environment variable name to set
    """
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("LANGSMITH_API_KEY")
os.environ['LANGSMITH_TRACING'] = "true"
os.environ["LANGSMITH_PROJECT"] = "AL_LLM"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

from utils.SMILESUtils import compact_df
import io

class StrategyState(TypedDict):
    AL_task: str
    report: str
    final_selection : List[List[int]]
    selected_indices: Annotated[List[List[int]], operator.add]
    strategies : List[str]
    analysis : str
    reports : Annotated[List[str], operator.add]
    batch_size : int

class StrategyOutput(BaseModel):
    strategies: List[str] = Field(description="List of strategies to implement")
    analysis : str = Field(description="Analysis of the current state of the AL campaign and rationale for the strategies")

def get_prompts(oracle_model : str, batch_size : int, total_cycles : int, cycle : int, train_df : pd.DataFrame, pred_df : pd.DataFrame, protein : str, training_results_path : str = None, simple_agent : int = 0, disable_ucb=False, disable_smarts=False, disable_tanimoto=False):
    """Generate prompts for strategy and selection phases.
    
    Loads prompt templates from YAML configuration and formats them with current AL state
    information including cycle details, training data statistics, and prediction summaries.
    
    Args:
        oracle_model: Name of the oracle model being used
        batch_size: Number of compounds to select in this cycle
        total_cycles: Total number of AL cycles planned
        cycle: Current cycle number
        train_df: Training data DataFrame with SMILES and affinity columns
        pred_df: Prediction DataFrame with SMILES, predictions, and confidence scores
        protein: Name of the target protein
        
    Returns:
        tuple: (strategy_prompt, selection_prompt) - Formatted prompt strings
        
    Raises:
        FileNotFoundError: If prompts_LLMChainSelector.yaml is not found
        KeyError: If required prompt templates are missing from YAML
    """
    with open('alselectors/prompts_LLMChainSelector.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    strategy_prompt = config['strategy_prompt']['template']
    selection_prompt = config['selection_prompt']['template']
    cycle_info = config['cycle_info']['template']
    implementer_system_prompt = config['implementer_system_prompt']['template']
    
    # Read previous cycles data from JSON if available
    past_cycles_data = ""
    max_per_cycle = train_df.groupby('cycle_added')['affinity'].max().to_list()
    mean_per_cycle = train_df.groupby('cycle_added')['affinity'].mean().to_list()
    rmse_per_cycle = [np.nan]    

    if training_results_path and training_results_path != "" and Path(training_results_path).exists():
        with open(training_results_path, 'r') as f:
            results = json.load(f)
        summaries = []
        # Format past cycles data
        past_cycles_list = []
        for i, cycle_data in enumerate(results['al_cycles']):
            if i == 0:  # Initial cycle
                summaries.append(cycle_data.get('summary', 'Initial training data'))
            else:
                summaries.append(cycle_data.get('summary', 'No summary available'))
        cycle_data = results['al_cycles'][-1]
        if simple_agent == 1 or cycle - 7 > 0: # Context management
            past_cycles_list.append(f"Cycle 0:\n{summaries[0]}\n") # {compact_df(train_df[train_df['cycle_added'] == 0][['SMILES', 'affinity']], index=False)}
        else:
            past_cycles_list.append(f"Cycle 0:\n{summaries[0]}\n{compact_df(train_df[train_df['cycle_added'] == 0][['SMILES', 'affinity']], index=False)}")
        # Get selected candidates data for this cycle
        if 'selected_predictions' in cycle_data and 'selected_confidence_scores' in cycle_data:
            # Validate that all arrays have the same length
            samples_selected = cycle_data['samples_selected']
            selected_predictions = cycle_data['selected_predictions']
            selected_confidence_scores = cycle_data['selected_confidence_scores']
            
            # Ensure all are lists/arrays and have the same length
            if not all(isinstance(arr, (list, np.ndarray)) for arr in [samples_selected, selected_predictions, selected_confidence_scores]):
                print(f"Warning: Invalid data types in cycle {cycle_data.get('cycle', 'unknown')}")
                
            if not (len(samples_selected) == len(selected_predictions) == len(selected_confidence_scores)):
                print(f"Warning: Array length mismatch in cycle {cycle_data.get('cycle', 'unknown')}: "
                      f"samples={len(samples_selected)}, predictions={len(selected_predictions)}, confidence={len(selected_confidence_scores)}")
                        
            # Create DataFrame for selected candidates with only valid data
            selected_df = pd.DataFrame({
                'SMILES': [str(i) for i in samples_selected],
                'oracle_prediction': selected_predictions,
                'oracle_conf': selected_confidence_scores,
            })
            selected_df['oracle_std'] = 1/selected_df['oracle_conf'] - 1
            selected_df.drop(columns=['oracle_conf'], inplace=True)

            selected_df = train_df.merge(selected_df, on='SMILES', how='left')
            # Calculate RMSE for this cycle

            for cycle_added in range(1, max(selected_df['cycle_added']) + 1):
                cycle_data_filtered = selected_df[selected_df['cycle_added'] == cycle_added]
                if len(cycle_data_filtered) > 0:
                    # Only calculate RMSE if we have valid data
                    rmse = np.sqrt(np.mean((cycle_data_filtered['affinity'] - 
                                            cycle_data_filtered['oracle_prediction'])**2))
                    rmse_per_cycle.append(rmse)

                    if cycle_added == cycle-1:
                        if simple_agent == 1: # Context management
                            past_cycles_list.append(f"Cycle {cycle_added}:\n{summaries[cycle_added]}\n") # {compact_df(cycle_data_filtered.drop(columns=['cycle_added']), index=False)}
                        else:
                            past_cycles_list.append(f"Cycle {cycle_added}:\n{summaries[cycle_added]}\nSelected candidates from cycle {cycle_added} with oracle predictions and std:\n{compact_df(cycle_data_filtered.drop(columns=['cycle_added']), index=False)}")
                    else:
                        if simple_agent == 1 or cycle_added < cycle - 7: # Context management
                            past_cycles_list.append(f"Cycle {cycle_added}:\n{summaries[cycle_added]}\n") # {compact_df(cycle_data_filtered.drop(columns=['cycle_added','oracle_prediction','oracle_confidence']), index=False)}
                        else:
                            past_cycles_list.append(f"Cycle {cycle_added}:\n{summaries[cycle_added]}\nSelected candidates from cycle {cycle_added}:\n{compact_df(cycle_data_filtered.drop(columns=['cycle_added','oracle_prediction','oracle_std']), index=False)}")
        
        
        past_cycles_data = "\n\n".join(past_cycles_list)
    pred_df['std'] = 1/pred_df['confidence_scores'] - 1
    # Add RMSE to cycle info
    cycle_info = cycle_info.format(
        cycles_completed=cycle-1, total_cycles=total_cycles, 
        batch_size=batch_size, oracle_model=oracle_model, 
        train_df_describe=train_df[['SMILES', 'affinity']].describe().round(2), 
        pred_df_describe=pred_df[['SMILES', 'predictions', 'std']].describe().round(2),
        max_per_cycle=[round(i, 2) for i in max_per_cycle], mean_per_cycle=[round(i, 2) for i in mean_per_cycle],
        oracle_rmse_per_cycle=[round(i, 2) for i in rmse_per_cycle])
    
    # Build available tools list based on ablation settings
    available_tools = []
    if not disable_ucb:
        available_tools.append("Upper confidence bound (UCB): Predictions + beta * std, given beta. Example: [UCB beta  = C affinity > D]")
    available_tools.append("Computational approaches: Predictions")
    if not disable_smarts:
        available_tools.append("Chemical approaches: Substructures (SMARTS or substructure names). Example: [with at least 2 <substructure2> subtituents, or a <substructure4> core if no <substructure3> rings]")
    if not disable_tanimoto:
        available_tools.append("Diversity approaches: Tanimoto similarity metrics (pairwise between selected candidates or to training data). Example: [with Tanimoto similarity <Z to high affinity training compounds (affinity > B)]")
    available_tools.append("Hybrid approaches: Combining the above")
    
    available_tools_text = "\n    - ".join(available_tools)
    
    strategy_prompt = strategy_prompt.format(
        cycle_info=cycle_info, batch_size=batch_size,
        protein=protein, past_cycles_data=past_cycles_data, available_tools=available_tools_text)
    selection_prompt = selection_prompt.format(pred_df_describe=pred_df[['SMILES', 'predictions', 'std']].describe().round(2))
    
    # Add tool restrictions to implementer system prompt if any tools are disabled
    if disable_ucb or disable_smarts or disable_tanimoto:
        disabled_capabilities = []
        if disable_ucb:
            disabled_capabilities.append("UCB (beta parameter)")
        if disable_smarts:
            disabled_capabilities.append("SMARTS substructure search")
        if disable_tanimoto:
            disabled_capabilities.append("Tanimoto similarity calculations")
            
        # Add restriction to implementer system prompt  
        implementer_restriction = f"""

**TOOL RESTRICTIONS (ABLATION STUDY):**
The following tools are DISABLED and will return errors if used:
{chr(10).join(f"- {cap}" for cap in disabled_capabilities)}
Do not attempt to use these disabled functionalities in your tool calls.
"""
        implementer_system_prompt = implementer_system_prompt + implementer_restriction
    
    return strategy_prompt, selection_prompt, implementer_system_prompt

def init_selector_builder(implement_builder, selection_prompt, pred_df : pd.DataFrame, simple_agent : int = 0):
    """Initialize the main selector builder with strategy generation and implementation workflow.
    
    Creates a StateGraph that orchestrates the complete AL selection process:
    1. Strategy generation using LLM analysis
    2. Parallel implementation of multiple strategies
    3. Final summarization and selection consolidation
    
    Args:
        implement_builder: Compiled implementer workflow graph
        selection_prompt: Formatted prompt template for strategy implementation
        pred_df: Prediction DataFrame with SMILES, predictions, and confidence scores
    Returns:
        StateGraph: Compiled selector workflow graph
        
    Example:
        >>> builder = init_selector_builder(implement_builder, selection_prompt)
        >>> result = builder.compile().invoke({"AL_task": "...", "batch_size": 10})
    """
    llm_strategy = ChatOpenAI(
            openai_api_key=os.environ.get('LAMBDA_API_KEY'),
            openai_api_base="https://api.lambda.ai/v1",
            model_name='qwen3-32b-fp8',
            max_tokens=32000 if simple_agent else 8000,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}}
        )
    llm_struct = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)

    strategy_system_prompt = """EXAMPLE OUTPUT DONT COPY:
<think>
/think Restate the goal in your own words. Reasoning about the task step by step.
</think>
<strategies>
{
"analysis": "# Active Learning Campaign Analysis - Cycle 3/5\n## Analysis of Current Campaign State\nThe campaign has reached its midpoint with promising results through two cycles",
"strategies": [
    "Select 12 candidates with RBFE predictions >X, [Other filters]",
    "Select 8 candidates [Filter 2]",
    "Select 1 candidate [Filter3]"
]}
</strategies>
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
        response = with_retry(llm_strategy.invoke, [HumanMessage(content=state['AL_task'])] + [SystemMessage(content=strategy_system_prompt)])
        structured_llm = llm_struct.with_structured_output(StrategyOutput)
        print(f'response: {response.content}')
        response = with_retry(structured_llm.invoke, [HumanMessage(content=response.content)])
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

        Based on your colleages' strategies, select exactly{state['batch_size'] - len(prev_selected)} more compounds. Make sure there are no duplicates."""

            implementer = Send("do_implementation", {
                "messages": [HumanMessage(content=selection_prompt.format(strategy=str(strategy)))], 
                "strategy": "Other strategies didnt meet the batch size. Enforce batch size by selecting more compounds.",
                "prev_selected": prev_selected})
            return implementer
        return "summarise_implementers"

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
        prompt = f"""You are an expert chemist tasked with summarizing a compound acquisition step in an Active Learning (AL) cycle. Multiple expert chemists have independently selected compounds using different strategies based on previous experimental results. Some have succeded with implementing their strategies, others may have failed.
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
- BE VERY CLEAR IF ANY SUBSTRUCTURE WAS NOT FOUND IN THE DATABASE OR IF FILTERS WERE RELEASED, THESE ARE CONSIDERED FAILURES
- Output only the final report (no preamble or explanations)

<report_structure>
# AL Campaign Acquisition Summary
## Implementation
[Highlight most important compound choices and if the selection targets were achieved]

## Campaign Impact
[How selections advance AL objectives, what was exploit/explore split?]
</report_structure>
"""
        validated = with_retry(llm_struct.invoke, [SystemMessage(content=prompt)] + [HumanMessage(content="Write a report based upon these memos.")])
        return {"report" : validated.content, "final_selection" : state['selected_indices']}
    
    builder = StateGraph(StrategyState)
    builder.add_node("generate_strategy", generate_strategy)
    builder.add_node("do_implementation", implement_builder.compile())
    builder.add_node("summarise_implementers", summarise_implementers)
    builder.add_edge(START, "generate_strategy")
    builder.add_conditional_edges("generate_strategy", strategy_iterator, ['do_implementation'])
    builder.add_conditional_edges("do_implementation", strategy_iterator, ['do_implementation', 'summarise_implementers'])
    builder.add_edge("summarise_implementers", END)

    return builder

def run_selector_agent(oracle_model : str, batch_size : int, total_cycles : int, cycle : int, train_df : pd.DataFrame, pred_df : pd.DataFrame, protein : str, past_cycles : str = "", training_results_path : str = None, simple_agent : int = 0, disable_ucb=False, disable_smarts=False, disable_tanimoto=False):
    """Run the full LLM-based selector agent workflow for active learning.
    
    Orchestrates the entire AL selection process by generating prompts, initializing builders,
    and executing the multi-agent workflow. Returns the final selected indices and a summary report.
    
    Args:
        oracle_model: Name of the oracle model
        batch_size: Number of compounds to select in this cycle
        total_cycles: Total number of AL cycles
        cycle: Current cycle number
        train_df: Training data DataFrame
        pred_df: Prediction DataFrame
        protein: Name of the target protein
        
    Returns:
        tuple: (final_selection, summary) where final_selection is a list of selected indices and summary is a markdown report
    """
    strategy_prompt, selection_prompt, implementer_system_prompt = get_prompts(oracle_model, batch_size, total_cycles, cycle, train_df, pred_df, protein, training_results_path, simple_agent, disable_ucb, disable_smarts, disable_tanimoto)

    implement_builder = init_implementer_builder(train_df, pred_df, implementer_system_prompt, 
                                                disable_ucb=disable_ucb, disable_smarts=disable_smarts, disable_tanimoto=disable_tanimoto)
    selector_builder = init_selector_builder(implement_builder, selection_prompt, pred_df, simple_agent)

    memory = MemorySaver()
    graph = selector_builder.compile(checkpointer=memory)
    thread = {"configurable": {"thread_id": "1"}}
    input = {'AL_task': strategy_prompt, 'batch_size': batch_size}
    response = with_retry(graph.invoke, input, thread, stream_mode="values")
    return response['final_selection'], response['report'], response['strategies']
