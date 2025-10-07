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


def query_mols_by_substructure(SMARTS_as_sql: str, SMILES_df: pd.DataFrame) -> pd.DataFrame:
    """
    Search molecules using pre-tokenized SQL-style query with AND, OR, NOT operators, parentheses,
    and numeric prefixes for repeated substructures. Results are sorted by affinity in descending order.
    Note: SMARTS_as_sql is a string, not a list.

    Examples:
    search_mols_by_substructure("c1ccncc1 AND C(=O)O")
    search_mols_by_substructure("c1ccncc1 AND ( C(=O)O OR CCO )")
    search_mols_by_substructure("( c1ccncc1 OR C1CCNCC1 ) AND NOT CCO")
    search_mols_by_substructure("2*F")  # Two fluorides
    search_mols_by_substructure("F AND F")  # Two fluorides (alternative syntax)
    search_mols_by_substructure("3*c1ccccc1 AND C(=O)O")  # Three benzene rings and carboxylic acid
    search_mols_by_substructure("2*C(=O)O OR 3*F")  # Two carboxylic acids OR three fluorides
    
    Args:
        SMARTS_as_sql: SMARTS patterns and logical operators with spaces between all operators. 
                      Supports numeric prefixes like "2*F" for repeated substructures.
                      SMARTS do not include H atoms.    
    Returns:
        df with search results
    """
    
    def parse_tokens_with_parentheses(query_string):
        """
        Parse tokens, distinguishing between SMARTS parentheses and SQL logical parentheses.
        Strategy: Use a stack to track parentheses pairs and determine context.
        """
        result = []
        stack = []  # Stack of (position, is_in_smarts, result_index) tuples
        current_token = ""
        
        def is_at_token_boundary(pos, char_type='before'):
            """Check if we're at a token boundary (whitespace, start/end of string)"""
            if char_type == 'before':
                if pos == 0:
                    return True
                prev_char = query_string[pos - 1]
                return prev_char.isspace()
            else:  # char_type == 'after'
                if pos == len(query_string) - 1:
                    return True
                # Look ahead to see if we hit whitespace or end
                next_pos = pos + 1
                while next_pos < len(query_string) and query_string[next_pos] == ')':
                    next_pos += 1
                if next_pos >= len(query_string):
                    return True
                return query_string[next_pos].isspace()
        
        def flush_current_token():
            """Add current token to result if not empty"""
            nonlocal current_token
            if current_token.strip():
                result.append(current_token.strip())
            current_token = ""
        
        i = 0
        while i < len(query_string):
            char = query_string[i]
            
            if char == '(':
                # Determine if this '(' is part of SMARTS based on the previous character
                if i == 0:
                    # At start of string - not part of SMARTS
                    is_in_smarts = False
                else:
                    prev_char = query_string[i - 1]
                    if prev_char.isspace():
                        # Previous char is whitespace - not part of SMARTS
                        is_in_smarts = False
                    elif prev_char == '(':
                        # Previous char is '(' - check if that one is part of SMARTS
                        _, prev_is_in_smarts, _ = stack[-1]
                        is_in_smarts = prev_is_in_smarts
                    else:
                        # Previous char is other - part of SMARTS
                        is_in_smarts = True
                
                if is_in_smarts:
                    # Part of SMARTS - add to current token
                    current_token += char
                    # Push to stack with no result index
                    stack.append((i, is_in_smarts, None))
                else:
                    # SQL parenthesis - flush current token and add the parenthesis
                    flush_current_token()
                    result.append('(')
                    # Push to stack with result index
                    stack.append((i, is_in_smarts, len(result) - 1))
                
            elif char == ')':
                if not stack:
                    # Unmatched ')' - treat as part of current token
                    current_token += char
                else:
                    # Pop from stack
                    open_pos, open_is_in_smarts, result_index = stack.pop()
                    
                    # Check if this ')' is at a token boundary
                    close_is_in_smarts = not is_at_token_boundary(i, 'after')
                    
                    # If EITHER parenthesis is in SMARTS, treat both as SMARTS
                    if open_is_in_smarts or close_is_in_smarts:
                        # Both are part of SMARTS
                        if not open_is_in_smarts:
                            # The opening paren was treated as SQL, but now we know it's SMARTS
                            # Remove it from the result list and add to current token
                            if result_index is not None:
                                del result[result_index]
                                # Update result indices in stack for any items after the deleted one
                                for j in range(len(stack)):
                                    if stack[j][2] is not None and stack[j][2] > result_index:
                                        old_pos, old_smarts, old_idx = stack[j]
                                        stack[j] = (old_pos, old_smarts, old_idx - 1)
                            current_token = '(' + current_token
                        current_token += char
                    else:
                        # Both are SQL parentheses
                        # The opening parenthesis is already in the result
                        # Flush any content between the parentheses
                        flush_current_token()
                        # Add the closing parenthesis
                        result.append(')')
                        
            elif char.isspace():
                # Whitespace - flush current token
                flush_current_token()
                # Skip whitespace
                while i < len(query_string) and query_string[i].isspace():
                    i += 1
                i -= 1  # Adjust because we'll increment at the end of the loop
            else:
                # Regular character - add to current token
                current_token += char
            i += 1
        
        # Flush any remaining token
        flush_current_token()
        
        return result
    
    # Parse the query string with improved parentheses handling
    SMARTS_tokens = parse_tokens_with_parentheses(SMARTS_as_sql)
    pos = 0
    print(SMARTS_tokens)
    old_stderr = sys.stderr
    sys.stderr = captured_output = io.StringIO()

    try:
        for token in SMARTS_tokens:
            if not token in ['AND', 'OR', 'NOT', '(', ')']:
                pattern = Chem.MolFromSmarts(token)
                
                if pattern is None:
                    # Force flush and check for captured errors
                    captured_output.flush()
                    error_output = captured_output.getvalue().strip()
                    
                    if error_output:
                        return pd.DataFrame(columns=SMILES_df.columns), error_output
                    else:
                        return pd.DataFrame(columns=SMILES_df.columns), f"\nUnable to parse {token}"
    finally:
        sys.stderr = old_stderr

    def count_substructure_matches(smiles, smarts_pattern):
        """Count the number of times a substructure appears in a molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            pattern = Chem.MolFromSmarts(smarts_pattern)
            if not pattern or not mol:
                return 0
            matches = mol.GetSubstructMatches(pattern)
            return len(matches)
        except:
            return 0

    def parse_term():
        nonlocal pos
        if pos >= len(SMARTS_tokens):
            return lambda x: True
        
        if SMARTS_tokens[pos] == 'NOT':
            pos += 1
            term_func = parse_term()
            return lambda x: not term_func(x)
        elif SMARTS_tokens[pos] == '(':
            pos += 1
            expr_func = parse_or()
            pos += 1  # skip ')'
            return expr_func
        else:
            token = SMARTS_tokens[pos]
            pos += 1
            mol = Chem.MolFromSmarts(token)
            return lambda x: mol and Chem.MolFromSmiles(x) and Chem.MolFromSmiles(x).HasSubstructMatch(mol)

    def parse_and():
        nonlocal pos
        left = parse_term()
        
        # Handle repeated AND with same pattern (e.g., "F AND F" = "2*F")
        pattern_counts = {}
        current_patterns = []
        
        # Collect the first pattern
        if pos > 0:
            prev_token = SMARTS_tokens[pos - 1]
            if prev_token not in [')', 'NOT']:
                current_patterns.append(prev_token)
        
        while pos < len(SMARTS_tokens) and SMARTS_tokens[pos] == 'AND':
            pos += 1
            
            # Look ahead to see if next term is a simple SMARTS (not NOT or parentheses)
            if pos < len(SMARTS_tokens) and SMARTS_tokens[pos] not in ['NOT', '(']:
                next_token = SMARTS_tokens[pos]
                current_patterns.append(next_token)
            
            right = parse_term()
            left = lambda x, l=left, r=right: l(x) and r(x)
        
        # Optimize repeated patterns
        if len(current_patterns) > 1:
            pattern_counts = {}
            for pattern in current_patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Create optimized function for patterns that repeat
            def optimized_and(x):
                try:
                    mol = Chem.MolFromSmiles(x)
                    if not mol:
                        return False
                    
                    for pattern, required_count in pattern_counts.items():
                        if required_count > 1:
                            actual_count = count_substructure_matches(x, pattern)
                            if actual_count < required_count:
                                return False
                        else:
                            pattern_mol = Chem.MolFromSmarts(pattern)
                            if not pattern_mol or not mol.HasSubstructMatch(pattern_mol):
                                return False
                    return True
                except:
                    return False
            
            # If we have repeated patterns, use optimized version
            if any(count > 1 for count in pattern_counts.values()):
                return optimized_and
        
        return left
    
    def parse_or():
        nonlocal pos
        left = parse_and()
        while pos < len(SMARTS_tokens) and SMARTS_tokens[pos] == 'OR':
            pos += 1
            right = parse_and()
            left = lambda x, l=left, r=right: l(x) or r(x)
        return left
    
    filter_func = parse_or()
    matches = SMILES_df[SMILES_df['SMILES'].apply(filter_func)][['SMILES', 'predictions', 'confidence_scores']]
    return matches, ""


class Implementer(MessagesState):
    strategy: str
    reports : List[str]
    prev_selected: List[int]
    selected_indices: List[int]

class FinalSelectionOutput(BaseModel):
    selected_indices: List[int] = Field(description="List of indices of selected candidates")
    report: str = Field(description="Brief report of rationale behind the selection of candidates and clear description of any substructures that were not found in the database")

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
                output = with_retry(tool.invoke, kwargs)
                
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

def init_implementer_builder(train_df, pred_df, implementer_system_prompt, random_seed=42,
                           disable_ucb=False, disable_smarts=False, disable_tanimoto=False):
    """Initialize the implementer builder with molecular fingerprinting and search tools.
    
    Creates a StateGraph for implementing selection strategies with tools for:
    - Substructure search using SMARTS queries with SQL-like syntax
    - Prediction querying with similarity and confidence constraints
    - Molecular fingerprinting using Morgan fingerprints
    
    Args:
        train_df: pandas DataFrame with training data (SMILES, affinity)
        pred_df: pandas DataFrame with predictions (SMILES, predictions, confidence_scores)
        implementer_system_prompt: System prompt for the implementer
        random_seed: Random seed for reproducibility
        disable_ucb: Disable Upper Confidence Bound functionality
        disable_smarts: Disable SMARTS substructure search functionality
        disable_tanimoto: Disable Tanimoto similarity functionality
        
    Returns:
        StateGraph: Compiled implementer workflow graph
        
    Example:
        >>> builder = init_implementer_builder(train_df, pred_df, prompt)
        >>> result = builder.compile().invoke({"messages": [HumanMessage(content="...")]})
    """

    # Validate that the train_df has columns "SMILES" and "affinity" and the pred_df has columns "SMILES", "predictions", and "confidence_scores"
    if not all(col in train_df.columns for col in ["SMILES", "affinity", "cycle_added"]):
        raise ValueError("train_df must have columns 'SMILES', 'affinity', and 'cycle_added'")
    if not all(col in pred_df.columns for col in ["SMILES", "predictions", "confidence_scores"]):
        raise ValueError("pred_df must have columns 'SMILES', 'predictions', and 'confidence_scores'")
    
    np.random.seed(random_seed)
    morgan_gen = GetMorganGenerator(radius=4, fpSize=4096, countSimulation=False)
    def get_fingerprint(smiles):
        return morgan_gen.GetFingerprint(Chem.MolFromSmiles(smiles))

    fp_train = [get_fingerprint(smiles) for smiles in train_df['SMILES']]
    fp_pred = [get_fingerprint(smiles) for smiles in pred_df['SMILES']]

    @tool
    def query_mols(prev_selected : Annotated[List[int], InjectedToolArg] = [],n_select : Optional[int] = 10, SMARTS_query: Optional[str] = None, max_pairwise_similarity : Optional[float] = None, range_similarity_to_training : Optional[List[Optional[float]]] = [None,None], min_training_affinity : Optional[float] = None, range_pred_affinity : Optional[List[Optional[float]]] = [None,None], beta : Optional[float] = 0):
        """
        Query molecular predictions database with tanimoto, affinity, std, and substructure filters. Results sorted by predicted affinity (descending).
        Examples:
        
        # Basic
        query_mols(n_select=5)  # Top 5 by affinity
        query_mols(range_pred_affinity=[8.0, None])  # Affinity ≥ 8.0

        # Upper confidence bound (UCB): pred_affinity = affinity + beta * uncertainty
        query_mols(beta=4.0)  # Sorts by pred_affinity = affinity + 4 * uncertainty 
        
        # Substructure
        query_mols(SMARTS_query="c1ccccc1")  # Benzene ring
        query_mols(SMARTS_query="c1ccncc1 AND c[OH]")  # Pyridine + alcohol on aromatic ring (like phenol)
        query_mols(SMARTS_query="F AND F AND c1ccccc1")  # Two fluorines + benzene
        query_mols(SMARTS_query="((c1ccnac1 OR A1*~*[N,O,P]=*[#6]1) AND C(=O)O) AND NOT (Cl AND Cl AND Cl)")  # Complex logic
        
        # Similarity & diversity
        query_mols(range_similarity_to_training=[0.6, 1.0], min_training_affinity=9.0)  # Similar to high-affinity training
        query_mols(max_pairwise_similarity=0.7, n_select=10)  # 10 diverse molecules
        
        # Combined filtering
        query_mols(n_select=25, SMARTS_query="c1ccccc1 AND C(=O)O", range_pred_affinity=[7.0, None], 
                beta=1.0, max_pairwise_similarity=0.8)
        
        Key parameters:
        - SMARTS_query: Substructure patterns with AND/OR/NOT logic, repeated AND for multiple matches
        - max_pairwise_similarity: Diversity constraint (0.6-0.8 for diverse sets)
        - range_similarity_to_training: Similarity to training compounds [0.3-0.7 novel-like, 0.7-1.0 similar]
        - min_training_affinity: Filter training compounds used for similarity
        - Ranges: Use [min, None] or [None, max] for one-sided constraints
        
        Returns formatted string with molecule count and results table.
        """

        # Validate parameters against disabled tools
        if disable_ucb and beta is not None and beta > 0:
            return "ERROR: UCB functionality (beta parameter) is disabled in this study. Please use beta=0 or omit the beta parameter."
            
        if disable_smarts and SMARTS_query is not None:
            return "ERROR: SMARTS substructure search is disabled in this study. Please omit the SMARTS_query parameter."
            
        if disable_tanimoto:
            if max_pairwise_similarity is not None and max_pairwise_similarity < 1.0:
                return "ERROR: Tanimoto similarity calculations are disabled in this study. Please omit max_pairwise_similarity or set it to 1.0."
            if range_similarity_to_training != [None, None] and range_similarity_to_training is not None:
                return "ERROR: Tanimoto similarity calculations are disabled in this study. Please omit range_similarity_to_training."

        # Force disabled parameters to safe defaults
        if disable_ucb:
            beta = 0
        if disable_smarts:
            SMARTS_query = None  
        if disable_tanimoto:
            max_pairwise_similarity = 1.0 if max_pairwise_similarity is None else 1.0
            range_similarity_to_training = [None, None]

        # Handle None values by setting to default values
        if n_select is None:
            n_select = 10
        
        if max_pairwise_similarity is None:
            max_pairwise_similarity = 1
        
        print(prev_selected)
        # Helper function to normalize ranges
        def normalize_range(range_val, default_range):
            if range_val is None:
                return default_range
            if len(range_val) != 2:
                raise ValueError(f"Range must have exactly 2 elements, got {len(range_val)}")
            
            min_val, max_val = range_val
            # Handle None in range bounds
            if min_val is None:
                min_val = default_range[0]
            if max_val is None:
                max_val = default_range[1]
                
            return [min_val, max_val]
        
        # Normalize all range parameters
        range_similarity_to_training = normalize_range(range_similarity_to_training, [0, 1])
        range_pred_affinity = normalize_range(range_pred_affinity, [0, np.inf])
        
        if min_training_affinity is None:
            min_training_affinity = 0

        train_reduced = train_df[train_df['affinity'] > min_training_affinity][['SMILES', 'affinity']].copy()
        if len(train_reduced) == 0:
            return "No training compounds match affinity criteria"

        info = ""
        if min_training_affinity > 0:
            info += f"{len(train_reduced)} molecules left in the training database after training affinity filters\n"
        

        pred_reduced = pred_df.loc[pred_df.index.difference(prev_selected), ['SMILES', 'predictions', 'confidence_scores']].copy()
        pred_reduced['predictions'] = pred_reduced['predictions'] + beta * (1/pred_reduced['confidence_scores']-1)
        pred_reduced = pred_reduced[
            (pred_reduced['predictions'] >= range_pred_affinity[0]) & 
            (pred_reduced['predictions'] <= range_pred_affinity[1])
        ]
        if len(pred_reduced) == 0:
            return info + "No molecules match prediction criteria"
        if range_pred_affinity[0] > 0 or range_pred_affinity[1] < np.inf:
            info += f"{len(pred_reduced)} molecules left after prediction filter\n"

        if SMARTS_query is not None:
            pred_reduced, errors = query_mols_by_substructure(SMARTS_query, pred_reduced)
            if len(pred_reduced) < n_select:
                tmp = pred_df.loc[pred_df.index.difference(prev_selected), ['SMILES', 'predictions', 'confidence_scores']].copy()
                tmp, errors = query_mols_by_substructure(SMARTS_query, tmp)
                if len(tmp) == 0:
                    return info + "No molecules in database match substructure criteria" + errors
                else:
                    pred_reduced = tmp
                    info += f"Less than {n_select} molecules match all of substructure, prediction and std criteria, releasing prediction filters\n Continuing\n"

            info += f"{len(pred_reduced)} molecules left after substructure filters\n"
        fp_train_reduced = [fp_train[i] for i in train_reduced.index]

        def tanimoto_pairwise_matrix(fps1, fps2) -> np.ndarray:
            matrix = np.zeros((len(fps1), len(fps2)))
            for i, fp1 in enumerate(fps1):
                for j, fp2 in enumerate(fps2):
                    matrix[i, j] = TanimotoSimilarity(fp1, fp2)
            return matrix

        if range_similarity_to_training[0] != 0 or range_similarity_to_training[1] != 1:
            max_tanimoto_to_training = []
            for index in pred_reduced.index:
                training_tanimoto = tanimoto_pairwise_matrix([fp_pred[index]], fp_train_reduced)
                max_tanimoto_to_training.append(np.max(training_tanimoto))

            pred_reduced.loc[:, 'max_tanimoto_to_training'] = max_tanimoto_to_training
            pred_reduced = pred_reduced[
                (pred_reduced['max_tanimoto_to_training'] >= range_similarity_to_training[0]) &
                (pred_reduced['max_tanimoto_to_training'] <= range_similarity_to_training[1])
            ]
        if len(pred_reduced) == 0:
            return info + "No molecules match similarity to training criteria"
        info += f"{len(pred_reduced)} molecules left after similarity to training filters\n"

        pred_reduced_sorted = pred_reduced.sort_values(by='predictions', ascending=False)
        if beta > 0:
            pred_reduced_sorted = pred_reduced_sorted.rename(columns={'predictions': 'predictions_ucb'})
        if max_pairwise_similarity == 1:
            return info + compact_df(pred_reduced_sorted.iloc[:n_select])
        else:
            selected_indices = []
            max_pairwise_similarities = []
            for idx in pred_reduced_sorted.index:
                if len(selected_indices) >= n_select:
                    break
                
                # Check pairwise similarity to all already selected molecules
                max_sim = 0
                if selected_indices: 
                    sim_matrix = tanimoto_pairwise_matrix([fp_pred[idx]], [fp_pred[j] for j in selected_indices])
                    max_sim = np.max(sim_matrix)
                    
                    # Skip if similarity is too high
                    if max_sim >= max_pairwise_similarity:
                        continue
                    for j, val in enumerate(max_pairwise_similarities):
                        if val < sim_matrix[0][j]:
                            max_pairwise_similarities[j] = sim_matrix[0][j]

                
                # Add to selected
                selected_indices.append(idx)
                max_pairwise_similarities.append(max_sim)

            if selected_indices:
                pred_reduced_selected = pred_reduced.loc[selected_indices].copy()
                pred_reduced_selected.loc[:, 'max_pairwise_similarity'] = max_pairwise_similarities
            else:
                # Return empty dataframe with correct columns if no molecules selected
                pred_reduced_selected = pd.DataFrame(columns=pred_reduced.columns)
                pred_reduced_selected.loc[:, 'max_pairwise_similarity'] = []
        
            return info + compact_df(pred_reduced_selected.drop(columns=['SMILES']))
    
    llm_selection =  ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)

    llm_struct = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)
    tools = [query_mols]
    llm_with_tools = llm_selection.bind_tools(tools, parallel_tool_calls=False)
    
    systemmsg = SystemMessage(content=implementer_system_prompt)

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
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = messages + [systemmsg]
        
        response = with_retry(llm_with_tools.invoke, messages)
        return {"messages": [response], "strategy": state['strategy']}

    def clean_implementation(state: Implementer):
        """Clean and validate the implementation results.
        
        Extracts structured output from the LLM response and validates it against
        the FinalSelectionOutput schema to ensure proper format and content.
        
        Args:
            state: Implementer state containing messages from strategy implementation
            
        Returns:
            dict: Validated selected indices and implementation report
        """
        structured_output = llm_struct.with_structured_output(FinalSelectionOutput)
        validated = with_retry(structured_output.invoke, state['messages'][-1].content)
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
    tool_node = InjectedToolNode(tools, injection_config={"query_mols": [("prev_selected", "prev_selected", [])]})
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