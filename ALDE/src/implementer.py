"""
Implementer Module for LLM Agent-based Active Learning Selector

This module implements the implementer workflow that executes selection strategies
using LLM interactions with tools for protein sequence querying and selection.

The implementer provides two main tools:
- query_sequences: Advanced sequence selection with Hamming distance constraints
- explore_uncertainty: Discovery-focused exploration of high-uncertainty sequences

Key Features:
- Hamming distance filtering for sequence diversity control
- Pattern-based sequence filtering using regex
- Fitness and uncertainty-based selection strategies
- Automatic exclusion of previously selected sequences

Key Components:
- FinalSelectionOutput: Pydantic model for structured selection output
- Implementer: State management for strategy implementation
- InjectedToolNode: Custom ToolNode for state injection
- init_implementer_builder: Main function to create implementer workflow

Dependencies:
- langchain_anthropic: For LLM interactions
- langgraph: For multi-agent workflow orchestration
- pandas/numpy: For data manipulation and analysis
"""

import pandas as pd
import numpy as np
from typing import TypedDict, List, Annotated, Optional, Tuple, Any, Dict
import operator
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import anthropic

from langchain_core.tools import tool, InjectedToolArg, BaseTool
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

# Import tools from our new tools module
from src.tools import (
    apply_fitness_filters, 
    filter_by_pattern, 
    filter_hamming_distance
)

# Import prompt utilities
from src.prompts import compact_df


def retry_node(fn):
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff
        stop=stop_after_attempt(5),  # Retry up to 5 times
        retry=retry_if_exception_type(anthropic._exceptions.OverloadedError)
        )
    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapped


class FinalSelectionOutput(BaseModel):
    selected_indices: List[int] = Field(description="List of indices of selected candidates")
    report: str = Field(description="Brief report of rationale behind the selection of candidates and clear description of any substructures that were not found in the database")


class Implementer(MessagesState):
    strategy: str
    reports : List[str]
    prev_selected: List[int]
    selected_indices: List[int]


class InjectedToolNode(ToolNode):
    """Custom ToolNode that handles state injection for specific tools"""
    def __init__(self, tools: List[BaseTool], injection_config: Dict[str, List[Tuple[str, str, Any]]] = None, **kwargs):
        """
        Initialize the tool node with injection configuration.
        
        Args:
            tools: List of tools to be executed
            injection_config: Dict mapping tool names to injection specs.
                Format: {
                    "tool_name": [
                        ("kwarg_name", "state_key", default_value),
                        ...
                    ]
                }
            **kwargs: Additional arguments passed to parent ToolNode
        
        Example:
            injection_config = {
                "query_mols": [("prev_selected", "selected_indices", [])],
                "other_tool": [("user_id", "user_id", "default_user")]
            }
        """
        super().__init__(tools, **kwargs)
        self.injection_config = injection_config or {}
    
    def _inject_state_values(self, tool_call: dict, state: dict) -> dict:
        """Inject state values for tools that require them"""
        kwargs = tool_call["args"].copy()
        tool_name = tool_call["name"]
        
        if tool_name in self.injection_config:
            for param_name, state_key, default_value in self.injection_config[tool_name]:
                kwargs[param_name] = state.get(state_key, default_value)
                
        return kwargs

    def invoke(self, state: Dict[str, Any], config=None):
        """Override the invoke method to handle our custom injection logic"""
        messages = state["messages"]
        last_message = messages[-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return {"messages": []}
        
        tool_messages = []
        
        for tool_call in last_message.tool_calls:
            try:
                # Get modified kwargs with injected values
                kwargs = self._inject_state_values(tool_call, state)
                
                # Find and execute the tool using our stored mapping
                tool = self.tools_by_name[tool_call["name"]]
                output = tool.invoke(kwargs)
                
                # Convert output to string if needed
                content = str(output) if hasattr(output, '__str__') else output
                
                tool_message = ToolMessage(
                    content=content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
                tool_messages.append(tool_message)
                
            except Exception as e:
                # Handle tool execution errors based on handle_tool_errors setting
                if self.handle_tool_errors is True:
                    error_content = f"Error: {str(e)}\n Please fix your mistakes."
                elif isinstance(self.handle_tool_errors, str):
                    error_content = self.handle_tool_errors
                else:
                    # Re-raise the exception if error handling is disabled
                    raise e
                
                error_message = ToolMessage(
                    content=error_content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    status="error"
                )
                tool_messages.append(error_message)
        
        return {"messages": tool_messages}


def init_implementer_builder(train_df, pred_df, implementer_system_prompt, random_seed=42):
    """Initialize the implementer builder with protein sequence search tools.
    
    Creates a StateGraph for implementing selection strategies with tools for:
    - Sequence pattern search using regex queries
    - Prediction querying with fitness and std constraints
    - Amino acid pattern matching for motif discovery
    
    Args:
        train_df: pandas DataFrame with training data (sequence, fitness)
        pred_df: pandas DataFrame with predictions (sequence, predictions, std_predictions)
        implementer_system_prompt: System prompt for the implementer
        random_seed: Random seed for reproducibility
        
    Returns:
        StateGraph: Compiled implementer workflow graph
        
    Example:
        >>> builder = init_implementer_builder(train_df, pred_df, system_prompt)
        >>> result = builder.compile().invoke({"messages": [HumanMessage(content="...")]})
    """

    # Validate that the train_df has columns "sequence" and "fitness" and the pred_df has columns "sequence", "predictions", and "std_predictions"
    if not all(col in train_df.columns for col in ["sequence", "fitness"]):
        raise ValueError("train_df must have columns 'sequence' and 'fitness'")
    if not all(col in pred_df.columns for col in ["sequence", "predictions", "std_predictions"]):
        raise ValueError("pred_df must have columns 'sequence', 'predictions', and 'std_predictions'")
    
    np.random.seed(random_seed)
    def query_sequences_by_regex_and_std(
        prev_selected=None,
        n_select=10,
        pattern_query=None,
        beta=0,
        range_fitness=[None,None],
        min_hamming_distance=None,
        max_hamming_distance=None,
        n_best_training : Optional[int] = 5,
        max_position_frequency_batch=None,
        max_position_frequency_training=None
    ):
        """
        Query protein sequences by amino acid regex pattern and Hamming distance filters.
        
        Args:
            prev_selected: List of previously selected sequence indices to exclude
            n_select: Number of sequences to select (default: 10)
            pattern_query: Regex pattern to match amino acid sequences (e.g., "^A.*K.*$")
            beta: Add upper confidence bound to the predictions. Prediction = prediction + beta * std
            range_fitness: [min, max] fitness range. Use None for unbounded sides
            min_hamming_distance: Minimum Hamming distance to training sequences and between selections
            max_hamming_distance: Maximum Hamming distance to training sequences (not enforced between selections)
            n_best_training: Number of best training sequences to consider for distance filtering (default: 5)
            max_position_frequency_batch: Maximum frequency (0.0-1.0) any amino acid can appear at any position in selected batch
            max_position_frequency_training: Maximum frequency (0.0-1.0) any amino acid can appear at any position in training data
        
        Examples:
            - query_sequences_by_regex_and_std(n_select=5)  # Top 5 by predictions
            - query_sequences_by_regex_and_std(pattern_query="^A.*K.*$", n_select=10)  # A at position 1, K at position 3
            - query_sequences_by_regex_and_std(pattern_query="^.*G.*$", beta=1)  # G in middle, high uncertainty
            - query_sequences_by_regex_and_std(beta=2)  # High std
            - query_sequences_by_regex_and_std(min_hamming_distance=1, max_hamming_distance=3, n_best_training=5)  # Distance constraints
        
        Pattern format: Use . for wildcards, e.g., "^A.*K.*$", "^.*G.*$", "^A.*K.*L.*$"
        Pattern length must match sequence length (4 amino acids).
        
        Returns:
            Formatted string with sequence count and results table.
        """
        info = ""
        if n_select > 48:
            n_select = 48
            info += f"Warning: maximum output is 48 sequences, setting n_select to 48\n"
        # Handle None values
        if prev_selected is None:
            prev_selected = []
        if n_select is None:
            n_select = 10

        # Remove previously selected sequences
        diff_df = pred_df.loc[pred_df.index.difference(prev_selected)].copy()
        diff_df['predictions'] = diff_df['predictions'] + np.random.normal(0, 1e-8, len(diff_df)) + beta * diff_df['std_predictions']
        available_df = diff_df.copy()
        previous_selected_sequences = pred_df.loc[prev_selected]['sequence'].tolist()
        fitness_filter = 1
        if len(available_df) == 0:
            return "No sequences available for selection.", []
        sequences_before_filtering = len(available_df)
        # Apply fitness and std filters
        available_df = apply_fitness_filters(available_df, range_pred_fitness=range_fitness)
        if len(available_df) < sequences_before_filtering:
            sequences_after_filtering = len(available_df)
            info += f"{sequences_after_filtering} sequences after fitness and std filter, {sequences_before_filtering - sequences_after_filtering} sequences removed\n"
            sequences_before_filtering = sequences_after_filtering
        # Apply pattern filter
        if pattern_query:
            available_df = filter_by_pattern(available_df, pattern_query)
            if len(available_df) < n_select:
                info = info + f"Fewer than {n_select} sequences match pattern '{pattern_query}'.\n Releasing fitness filter.\n"
                available_df = diff_df.copy()
                available_df = filter_by_pattern(available_df, pattern_query)
                fitness_filter = 0
            info += f"{len(available_df)} sequences after pattern filter\n"
            sequences_before_filtering = len(available_df)

        if len(available_df) == 0:
            return info + "No sequences match the specified filters.", []
        
        # Sort by fitness (descending) and select top n
        available_df = available_df.sort_values('predictions', ascending=False)
        # Check if Hamming distance or position frequency filtering is needed
        distance_filter_needed = (
            min_hamming_distance is not None or 
            max_hamming_distance is not None or
            max_position_frequency_batch is not None or
            max_position_frequency_training is not None
        )
        if distance_filter_needed:
            selected = filter_hamming_distance(available_df, train_df, n_select, 
                                             min_hamming_distance=min_hamming_distance, 
                                             max_hamming_distance=max_hamming_distance, 
                                             n_best_training=n_best_training, 
                                             prev_selected=previous_selected_sequences,
                                             max_position_frequency_batch=max_position_frequency_batch,
                                             max_position_frequency_training=max_position_frequency_training)
            if len(selected) < n_select and fitness_filter == 1:
                info = info + f"Fewer than {n_select} sequences match the Hamming distance filters.\n Try different distance constraints or increasing beta.\n"
            info += f"{len(selected)} sequences left after Hamming distance filter:\n"
            if beta > 0:
                selected = selected.rename(columns={'predictions': 'predictions_ucb'})
            return info + compact_df(selected)
        if beta > 0:
            available_df = available_df.rename(columns={'predictions': 'predictions_ucb'})
        return info + compact_df(available_df[:n_select])
    
    
    @tool
    def query_sequences(
        prev_selected : Annotated[List[int], InjectedToolArg] = [],
        n_select : int = 32,
        pattern_query: Optional[str] = None,
        beta : Optional[float] = 0,
        range_fitness : Optional[List[Optional[float]]] = None,
        min_hamming_distance : Optional[int] = None,
        max_hamming_distance : Optional[int] = None,
        n_best_training : Optional[int] = 5,
        max_position_frequency_batch : Optional[float] = None,
        max_position_frequency_training : Optional[float] = None
    ):
        """
        Query and filter protein sequences using pattern matching, fitness ranges, Hamming distance constraints, and position frequency caps.
        
        This tool filters a database of 4-amino acid sequences through multiple criteria to ensure diverse,
        high-quality selections while avoiding over-representation of specific amino acids at any position.
        
        Args:
            prev_selected: List of previously selected sequence indices to exclude from results
            n_select: Number of sequences to select (default: 32)
            pattern_query: Regex pattern to match amino acid sequences (e.g., "^A.*K.*$")
            beta: Add upper confidence bound to the predictions. Prediction = prediction + beta * std
            range_fitness: [min, max] fitness range. Use None for unbounded sides
            min_hamming_distance: Minimum Hamming distance to training sequences and between selections
            max_hamming_distance: Maximum Hamming distance to training sequences (not enforced between selections)
            n_best_training: Number of best training sequences to consider for distance filtering (default: 5)
            max_position_frequency_batch: Maximum frequency (0.0-1.0) any amino acid can appear at any position in selected batch
            max_position_frequency_training: Maximum frequency (0.0-1.0) any amino acid can appear at any position in training data
        
        Position Frequency Cap Details:
            Position frequency caps prevent over-representation of specific amino acids at any position:
            - max_position_frequency_training: Rejects sequences containing amino acids that appear 
              too frequently at any position in the training data (checked during greedy selection)
            - max_position_frequency_batch: Ensures no amino acid appears more than the cap percentage 
              at any position in the final batch (checked during greedy selection)
            
            Example: With max_position_frequency_batch=0.6 and selecting 10 sequences:
            - No amino acid can appear more than 6 times (60%) at position 1
            - No amino acid can appear more than 6 times (60%) at position 2, etc.

        Hamming Distance Filter Details:
            When min_hamming_distance or max_hamming_distance is specified, the tool ensures:
            - Each selected sequence has Hamming distance within [min, max] to ALL top training sequences
            - Each selected sequence has Hamming distance >= min to ALL other sequences returned by the tool
            This ensures sequence diversity in batches while maintaining appropriate distance to known good sequences.

        Range Filters:
            - Specify [min, max] where either bound can be None
            - [0.5, None]: value >= 0.5
            - [None, 0.8]: value <= 0.8  
            - [0.3, 0.7]: 0.3 <= value <= 0.7
        
        Examples:
            # Basic queries sorted by predictions
            query_sequences(n_select=5)  # Top 5 sequences by predictions
            query_sequences(n_select=10, min_hamming_distance=1, max_hamming_distance=3)  # Top 10 with distance constraints
            
            # Position frequency caps for amino acid diversity
            query_sequences(n_select=10, max_position_frequency_batch=0.5)  # No amino acid >50% at any position in batch
            query_sequences(max_position_frequency_training=0.7, n_select=15)  # Avoid amino acids >70% frequent in training
            
            # Upper confidence bound (UCB) for exploration
            query_sequences(beta=1, range_fitness=[0.8, None], n_select=10)  # Small UCB weight
            query_sequences(beta=2, n_select=15, max_hamming_distance=2)  # High uncertainty, close to training
            
            # Pattern matching
            query_sequences(pattern_query="^A.*K.*$", n_select=10)  # A at pos 1, K at pos 3
            query_sequences(pattern_query="^.*G.*$")  # G at position 2
            query_sequences(pattern_query="^[RK].*[DE].*$")  # Basic/acidic residues at pos 1/3
            
            # Combined filtering with all constraints (advanced usage)
            query_sequences(
                pattern_query="^A.*$", 
                min_hamming_distance=2,
                max_hamming_distance=3,
                max_position_frequency_batch=0.6,
                max_position_frequency_training=0.8,
                n_select=10,
                n_best_training=5,
                beta=0.5
            )  # A at pos 1, diverse batch with distance and frequency constraints

        Returns:
            Formatted string containing:
            - Number of sequences found matching criteria
            - Results table with sequence, predictions, std, and distance columns
            - Summary statistics if distance filtering was applied
        """
        if range_fitness is None:
            range_fitness = [None, None]
        return query_sequences_by_regex_and_std(prev_selected, n_select, pattern_query, beta, range_fitness, min_hamming_distance, max_hamming_distance, n_best_training, max_position_frequency_batch, max_position_frequency_training)
    
    @tool
    def explore_uncertainty(
        prev_selected : Annotated[List[int], InjectedToolArg] = [],
        n_select : int = 20,
        pattern_query: Optional[str] = None
    ):
        """
        Explore sequences with highest uncertainty (std predictions) for discovery.
        
        This tool helps identify sequences where the model is most uncertain, which often
        represent interesting areas of sequence space worth exploring for potential discoveries.
        
        Args:
            prev_selected: List of previously selected sequence indices to exclude from results
            n_select: Number of sequences to show (default: 20)
            pattern_query: Optional regex pattern to filter sequences (e.g., "^A.*K.*$")
        
        Examples:
            # Basic uncertainty exploration
            explore_uncertainty(n_select=10)  # Top 10 most uncertain sequences
            explore_uncertainty(n_select=25)  # Top 25 most uncertain sequences
            
            # Pattern-constrained uncertainty exploration
            explore_uncertainty(pattern_query="^A.*$", n_select=15)  # Most uncertain sequences starting with A
            explore_uncertainty(pattern_query="^.*G.*$", n_select=10)  # Most uncertain sequences with G at position 2
            explore_uncertainty(pattern_query="^[RK].*[DE].*$")  # Most uncertain basic/acidic combinations
        
        Returns:
            Formatted string containing:
            - Number of sequences found
            - Results table sorted by std_predictions (highest to lowest)
            - Shows sequence, predictions, std_predictions, and other relevant columns
        """
        info = ""
        if n_select > 48:
            n_select = 48
            info += f"Warning: maximum output is 48 sequences, setting n_select to 48\n"
            
        # Handle None values
        if prev_selected is None:
            prev_selected = []
        if n_select is None:
            n_select = 20

        # Remove previously selected sequences
        available_df = pred_df.loc[pred_df.index.difference(prev_selected)].copy()
        
        if len(available_df) == 0:
            return "No sequences available for exploration."
        # Apply pattern filter if provided
        if pattern_query:
            available_df = filter_by_pattern(available_df, pattern_query)
            if len(available_df) == 0:
                return info + f"No sequences match pattern '{pattern_query}'."
            info += f"{len(available_df)} sequences after pattern filter\n"
        
        # Sort by uncertainty (std_predictions) in descending order
        available_df = available_df.sort_values('std_predictions', ascending=False)
        
        # Select top n_select most uncertain sequences
        selected_df = available_df.head(n_select)
        
        info += f"Top {len(selected_df)} most uncertain sequences:\n"
        return info + compact_df(selected_df)
    
    llm_selection = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)
    tools = [query_sequences, explore_uncertainty]
    llm_with_tools = llm_selection.bind_tools(tools, parallel_tool_calls=False)
    
    systemmsg = SystemMessage(content=implementer_system_prompt)

    @retry_node
    def implement_strategy(state : Implementer):
        """Implement a selection strategy using LLM with tools.
        
        Processes the current state messages and invokes the LLM with available tools
        for molecular search and prediction querying. Ensures system message is included
        if not already present. Node communnicates with the tools with the messages state
        
        Args:
            state: Implementer state containing messages and strategy
            
        Returns:
            dict: Updated state with LLM response and strategy information
        """
        messages = state['messages']
        
        response = llm_with_tools.invoke(messages[:-1] + [systemmsg] + [messages[-1]])
        return {"messages": [response], "strategy": state['strategy']}

    @retry_node
    def clean_implementation(state: Implementer):
        """Clean and validate the implementation results.
        
        Extracts structured output from the LLM response and validates it against
        the FinalSelectionOutput schema to ensure proper format and content.
        
        Args:
            state: Implementer state containing messages from strategy implementation
            
        Returns:
            dict: Validated selected indices and implementation report
        """
        structured_output = llm_selection.with_structured_output(FinalSelectionOutput)
        validated = structured_output.invoke(state['messages'][-1].content)
        return {
            "selected_indices": [validated.selected_indices],
            "reports": [validated.report]
        }

    def route_implement(state : MessagesState):
        """Route implementation based on message content.
        
        Determines whether to continue with tool calls or proceed to clean implementation
        based on the presence of tool calls in the last message.
        
        Args:
            state: MessagesState containing the conversation history
            
        Returns:
            str: Next node to execute ("tools" or "clean_implementation")
        """
        last_message = state["messages"][-1]
        
        # If there are tool calls, route to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        else:
            return "clean_implementation"
    
    tool_node = InjectedToolNode(tools, injection_config={
        "query_sequences": [("prev_selected", "prev_selected", [])],
        "explore_uncertainty": [("prev_selected", "prev_selected", [])]
    })
    implement_builder = StateGraph(Implementer)
    implement_builder.add_node("implement_strategy", implement_strategy)
    implement_builder.add_node("clean_implementation", clean_implementation)
    implement_builder.add_node("tools", tool_node)
    implement_builder.add_edge(START, "implement_strategy")
    implement_builder.add_conditional_edges("implement_strategy",route_implement, {
            "tools": "tools",
            "clean_implementation": "clean_implementation"})
    implement_builder.add_edge("tools", "implement_strategy")
    implement_builder.add_edge("clean_implementation", END)
    return implement_builder
