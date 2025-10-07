#!/usr/bin/env python3
"""
Standalone LLM Agent for Protein Active Learning Campaign

This script runs a complete active learning campaign using a simplified LLM agent
that directly controls sequence selection without complex multi-agent workflows.
"""

import argparse
import os
import getpass
import json
import random
import time
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import torch
from pydantic import BaseModel, Field

# LangChain imports
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage, ToolMessage
from typing import TypedDict, Annotated, List, Dict, Any, Literal
from typing_extensions import NotRequired
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from langgraph.graph.message import add_messages

# Import existing utilities for data loading
import sys
sys.path.append('src')
import src.objectives as objectives


def _set_env(var: str):
    """Set environment variable if not already set."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


# Set up environment variables like agent.py
_set_env("LANGSMITH_API_KEY")
os.environ['LANGSMITH_TRACING'] = "true"
os.environ["LANGSMITH_PROJECT"] = "ALDE_Standalone"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"


class SimpleSelection(BaseModel):
    """Simple structured output for sequence selection."""
    selection: List[str] = Field(description="List of protein sequences to validate next.")

class DirectedEvolutionState(TypedDict):
    """State schema for directed evolution workflow"""
    # Core message history with reducer to append new messages
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Current cycle information
    current_cycle: int
    max_cycles: int
    n_iter : int
    
    # Results and selections
    current_selection: List[str]
    
    # Context management
    context_length: NotRequired[int]
    max_context_length: int
    summary: NotRequired[str]


class StandaloneAgent:
    """Simplified standalone LLM agent for protein active learning."""
    
    def __init__(self, protein: str, batch_size: int, total_budget: int, random_seed: int, model: str, output_dir: str = "results/"):
        self.protein = protein
        self.batch_size = batch_size
        self.total_budget = total_budget
        self.random_seed = random_seed
        self.max_context_length = 35000
        self.model = model
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Load protein data
        self.all_sequences = pd.read_csv(f"data/{protein}/fitness.csv")
        self.all_sequences.columns = ["sequence", "fitness"]
        self.all_sequences["fitness"] = (self.all_sequences["fitness"] - self.all_sequences["fitness"].min()) / self.all_sequences["fitness"].max()
        self.n_total = len(self.all_sequences)
        
        print(f"Loaded {protein} dataset: {self.n_total} sequences")
        print(f"Fitness range: [{self.all_sequences['fitness'].min():.4f}, {self.all_sequences['fitness'].max():.4f}]")
        
        # Initialize LLMs
        self.blind = False
        if "-blind" in self.model:
            self.blind = True
        if "qwen" in self.model:
            self.llm_think = ChatOpenAI(
                openai_api_key=os.environ.get('LAMBDA_API_KEY'),
                openai_api_base="https://api.lambda.ai/v1",
                model_name='qwen3-32b-fp8',
                max_tokens=8192,
                temperature=0.6,
                extra_body={
                    "repetition_penalty": 1.15,      # HF/vLLM-style penalty (>1 reduces repeats)
                    "top_p": 0.9,
                    "top_k": 50,
                    "chat_template_kwargs": {"enable_thinking": True},  # or set False to test
                }
            )
        elif "deepseek" in self.model:
            self.llm_think = ChatOpenAI(
                openai_api_key=os.environ.get('LAMBDA_API_KEY'),
                openai_api_base="https://api.lambda.ai/v1",
                model_name='deepseek-r1-0528',
            )
        else:
            self.llm_think = ChatOpenAI(
                openai_api_key=os.environ.get('OPENAI_API_KEY'),
                model_name='gpt-5',
                max_tokens=25000,
                reasoning={"effort": "medium"}
        )
        
        self.llm_struct = ChatAnthropic(
            model="claude-3-5-sonnet-latest", 
            temperature=0
        )
        
        # Initialize conversation state and tracking
        self.conversation_messages = [
            SystemMessage(content="You are an expert protein engineer leading a directed evolution campaign.")
        ]
        self.cycle = 0
        if protein == 'GB1':
            wt = {"sequence": "VDGV", "fitness": 0.11413}
        elif protein == 'TrpB':
            wt = {"sequence": "VFVS", "fitness": 0.408074}

        if self.blind:
            self.validated_results = pd.DataFrame(columns=["sequence", "fitness"])
            self.last_validated_results = pd.DataFrame(columns=["sequence", "fitness"])
        else:
            self.validated_results = pd.DataFrame([wt])
            self.last_validated_results = pd.DataFrame([wt])
        self.results_dir = Path(output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Add campaign history tracking for JSON output
        self.campaign_history = []  # Store cycle-by-cycle results in expected format
        self.max_cycles = (total_budget + batch_size - 1) // batch_size  # ceiling division
        self.all_indices = []
        self.graph = self._build_graph()
        # Copy this script to results directory for reproducibility
        import shutil
        shutil.copy(__file__, self.results_dir / "standalone_agent.py")
    
    def run_campaign(self):
        """Run the campaign"""
        prompt = self.get_prompt()
        thread = {"configurable": {"thread_id": "0"}, "recursion_limit": 50}
        self.graph.invoke({"messages": [HumanMessage(content=prompt)], "current_cycle": self.cycle, "n_iter": 0, "max_cycles": self.max_cycles, "max_context_length": self.max_context_length, "summary": ""}, config=thread, stream_mode="values")
        
    def get_prompt(self) -> str:
        """Generate comprehensive background prompt for the reasoning LLM."""
        
        # Protein-specific backgrounds 
        backgrounds = {
            "GB1": """The target is a four-site epistatic region (wildtype: V39, D40, G41, V54, fitness ~0.1) of the 56-residue protein G domain B1 (GB1), an immunoglobulin-binding domain from Streptococcal bacteria. These sites account for a majority of the most strongly epistatic interactions in GB1 and span a fitness landscape of 160,000 variants. Variants were assessed for IgG-Fc binding using mRNA display and high-throughput sequencing.""",
            "TrpB": "The target is the β-subunit of tryptophan synthase, an enzyme that catalyzes l-tryptophan synthesis from indole and l-serine. The protein is highly conserved across all life forms except animals and exhibits rich natural sequence diversity. Fitness is linked to enzyme activity via a growth-based selection in an E. coli Trp auxotroph. The campaign is designed to optimise a 4 amino acid motif (wildtype motif V183,F184,V227,S228, fitness ~0.4)"
        }
        if self.protein == "TrpB":
            protein = 'a protein'
        else:
            protein = self.protein
        #BLIND
        if self.blind:
            protein = "unknown"
            background = "No background available, we enter the campaign blind." 
        else:
            background = backgrounds.get(self.protein, f"Protein optimization campaign for {self.protein}")
        
        return f"""You are an expert protein engineer with deep chemical intuition leading a directed evolution campaign. Apply rigorous chemical principles to discover high-fitness variants within your experimental budget.

**CAMPAIGN OVERVIEW:**
- Target: Four-site region of {protein}
- Background: {background}
- Total Budget: {self.total_budget} experimental validations
- Batch Size: {self.batch_size} sequences per round
- Number of cycles: {self.max_cycles}
- Sequence Length: 4 amino acids

This is the START of your campaign. You have no prior data.

**STRATEGIC APPROACH:**
Test well-reasoned chemical hypotheses. Balance thorough exploration with chemical principles to maximize discovery potential through systematic, chemistry-guided experimentation.

**OUTPUT REQUIREMENTS:**
1. **Chemical Reasoning**: Evaluate the past cycles and explain your mechanistic hypotheses and chemical logic for the next cycle.
3. **Priority Ranking**: Sort by priority with chemical justification
4. **Final List**: End with {3*self.batch_size//2} ranked sequences for next cycle validation for buffer, {self.batch_size} of which will be validated in the next cycle. 
**Output format**: 

List 15 sequences at the end of your response within <selection> tags. Example: <selection>ACDE,FGHI,KLMN,PQRS,TVWY</selection>
"""

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create state graph
        workflow = StateGraph(DirectedEvolutionState)
        
        # Add nodes
        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("evolution_cycle", self._evolution_cycle_node)
        workflow.add_node("update_state", self._update_state_node)
        workflow.add_node("count", self._count_node)
        # Set entry point
        workflow.add_edge(START, "evolution_cycle")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "update_state",
            self._route,
            {
                "summarize": "summarize",
                "continue": "evolution_cycle",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "evolution_cycle",
            self._route_count,
            {
                "count": "count",
                "continue": "update_state"
            }
        )
        workflow.add_edge("summarize", "evolution_cycle")
        workflow.add_edge("count", "evolution_cycle")
        graph = workflow.compile(checkpointer=MemorySaver())
        # Save the graph image to file
        graph.get_graph().draw_mermaid_png(output_file_path = self.results_dir / "graph.png")
        print(f"N test sequences: {len(self.all_sequences)}")
        return graph
    

    def _route(self, state: DirectedEvolutionState) -> Literal["summarize", "continue", "end"]:
        """Route based on context length"""
        if state["current_cycle"] >= state["max_cycles"]:
            return "end"
        

        return "summarize" if state["max_context_length"] < state["context_length"] else "continue"

    def _route_count(self, state: DirectedEvolutionState):
        """Route based on number of sequences validated"""
        if len(state["current_selection"]) == 0:
            print(state["messages"][-1])
            raise ValueError("No selection made, GPT-5 locked up") # Sometimes GPT-5 hits a bio-safetly lockout and decides not to respond
        new_unique = list(set([sq for sq in state["current_selection"] if sq not in self.validated_results["sequence"].tolist() and sq in self.all_sequences["sequence"].tolist()]))
        print(f"New unique: {len(new_unique)}")
        print(f"Current selection: {len(state['current_selection'])}")
        print(f"Batch size: {self.batch_size}")
        print(f"N iter: {state['n_iter']}")
        if len(new_unique) >= self.batch_size or state["n_iter"] > 5: #  Allow 5 attempts
            return "continue"
        return "count"
    
    def _count_node(self, state: DirectedEvolutionState):
        """Count node: re-prompt the LLM to select more sequences"""

        duplicates = [sq for sq in state["current_selection"] if sq in self.validated_results["sequence"].tolist() or len(sq) != 4 or sq not in self.all_sequences["sequence"].tolist()]
        selected_unique = list(set(state["current_selection"])-set(duplicates))
        left = int(3*self.batch_size//2 - len(selected_unique))
        
        msg = f"""You selected:
{state['current_selection']}, but the following sequences are invalid for validation:
{duplicates}
Select another {left} unique, 4 amino acid long, sequences different from the the ones you already selected. Make sure there are no duplicates in selection.
**Output format**: List exactly {left} sequences (STRICTLY 4-letter codes like 'AMHG', 'QPEI') within <selection> tags. Write out every sequence in the list. Example: <selection>ABCD,EFGH,IJKL</selection>
        """

        return {"messages": [SystemMessage(content=msg)], "n_iter": state["n_iter"] + 1, "current_selection": selected_unique}

    def _summarize_node(self, state: DirectedEvolutionState) -> Dict[str, Any]:
        """Summarize conversation history when context becomes too large"""
        messages = state["messages"]
        summary = state.get("summary", "")
        # Keep the initial system message and recent messages
        if summary:
            summary_prompt = (
                f"This is the current summary of the conversation: {summary}\n\n"
                "Update this summary by taking into account the new messages above."
            )
        else:
            summary_prompt = "Create a concise summary of the conversation above. Include background about the task. The summary should be cohesive and not too long. The sumamry should cover all important details necessary to understand the conversation and continue the campaign."
        
        # Get the new summary
        response = self.llm_struct.invoke(messages + [HumanMessage(content=summary_prompt)])
        summary_message = SystemMessage(
            content=f"Summary: {response.content}")
        last_ai = None
        last_human = None
        for msg in reversed(messages):
            if msg.type == "ai" and last_ai is None:
                last_ai = msg
            elif msg.type == "human" and last_human is None:
                last_human = msg
            
            # Break if we found both
            if last_ai is not None and last_human is not None:
                break
        remove_all = [RemoveMessage(id=msg.id) for msg in messages 
              if msg is not last_human and msg is not last_ai]

        return {
            "messages": remove_all, 
            "summary": response.content
            }
    
    def _evolution_cycle_node(self, state: DirectedEvolutionState) -> Dict[str, Any]:
        """Core AL cycle node"""
        messages = state["messages"]
        summaryinfo = state['summary'] if state['summary'] is not None else ""
        system_msg = SystemMessage(content=f"""You are an expert protein engineer with designing and controlling a directed evolution campaign. You must strategically select protein sequences to maximize fitness discovery within your experimental budget at the end of the campaign.
{summaryinfo}
Current cycle {state["current_cycle"]} of {state["max_cycles"]}.
Select {3*self.batch_size//2} sequences, optimal for the campaign goal, to validate in next cycle.
Of these, {self.batch_size} will be validated in the next cycle, depending on experimental feasibility.
Sort you selection by priority.

**Highest performing sequences so far**:
{self.validated_results.sort_values(by='fitness', ascending=False).head(10).round(3).to_string(index=False)}

**Output format**: End your analysis with a list of {3*self.batch_size//2} sequences (STRICTLY 4-letter codes like 'AMHG', 'QPEI') within <selection> tags. Write out every sequence in the list. Dont duplicate sequences.
""") 

        for i in range(3): # Allow 3 attempts to generate a selection
            try:
                print(f"Messages length: {len(messages)}")
                if not isinstance(messages[-1], SystemMessage):
                    think_response = self.llm_think.invoke(messages[:-1] + [system_msg] + [messages[-1]])
                else:
                    think_response = self.llm_think.invoke(messages)
                if "qwen" in self.model:
                    content = think_response.content
                elif "deepseek" in self.model:
                    content = think_response.content
                else:
                    content = think_response.content[0]['text'] 
                
                # Extract selection from response
                selection = list(set([sq.strip() for sq in content.split("<selection>")[1].split("</selection>")[0].split(",") if len(sq.strip()) == 4]))
                # If the model isnt faithfully following the output format, adding a extraction step here helps greatly at the cost of compute:
                # selection = self.llm_struct.with_structured_output(SimpleSelection).invoke(content).selection
                break
            except Exception as e:
                print(f"Error in iteration {i}: {e}")
                continue
        if i == 2:
            raise Exception("Failed to generate selection after 3 attempts")
        
        # Remove failed attempts from the conversation history
        context_length = sum(len(str(msg.content)) for msg in state["messages"]) / 1.1
        remove_iterations =[]
        for msg in messages[::-1]:
            if msg.type == "human":
                break
            else:
                remove_iterations.append(RemoveMessage(id=msg.id))
        selection = list(set(state.get("current_selection", []) + selection))
        return {
            "messages": [{"role": "assistant", "content": content}] + remove_iterations,
            "current_selection": selection,
            "context_length": context_length
        }
    
    def _update_state_node(self, state: DirectedEvolutionState) -> Dict[str, Any]:
        """Update state with experiment results and prepare for next cycle"""        
        selection = state.get("current_selection", [])
        print(f"Selection: {len(selection)}")
        selection = list(set([sq for sq in selection if sq not in self.validated_results["sequence"].tolist() and len(sq) == 4 and sq in self.all_sequences["sequence"].tolist()]))[:self.batch_size]
        validated_result = self.validate_sequences(selection)
        
        # Create experiment record
        self.save_cycle_results()
        results = HumanMessage(content=f"""/think 
The validation experiment in cycle {state['current_cycle']} is finished. These are the results:
{self.last_validated_results.sort_values(by="fitness", ascending=False)[["sequence", "fitness"]].round(3).to_string(index=False)}
""")
        self.cycle += 1
        return {
            "messages": [results],
            "current_cycle": state["current_cycle"] + 1,
            "n_iter": 0,
            "current_selection": []
        }

    def validate_sequences(self, sequences: List[str]) -> pd.DataFrame:
        """Look up fitness values for selected sequences and update tracking."""
        mask = self.all_sequences["sequence"].isin(sequences)
        subdf = self.all_sequences[mask].copy()  
        indicies = self.all_sequences[mask].index.tolist()
        self.validated_results = pd.concat([self.validated_results, subdf])
        self.last_validated_results = subdf
        self.all_indices.extend(indicies)
    
        # Print progress
        mean_fitness = np.mean(subdf["fitness"])
        max_fitness = np.max(subdf["fitness"])
        overall_max = np.max(self.validated_results["fitness"])
        
        print(f"Cycle {self.cycle + 1} results from {len(subdf)} sequences - Mean: {mean_fitness:.4f}, Max: {max_fitness:.4f}, Overall Max: {overall_max:.4f}")
        
        return subdf
    
    def save_cycle_results(self):
        """Save results in the same format as cursorrules example."""
        
        # Save indices as .pt file (matching optimize.py format)
        indices_tensor = torch.tensor(self.all_indices, dtype=torch.long)
        
        # Also save in execute_simulation.py compatible format
        agent_tag = self.model.upper()
        output_file = self.results_dir / f"AGENT_{agent_tag}-DO-0-RBF-AGENT-[1, 1]_{self.random_seed}indices.pt"
        torch.save(indices_tensor, output_file)
        
        # Get current cycle data from last_validated_results
        overall_mean_fitness = self.validated_results["fitness"].mean()
        overall_max_fitness = self.validated_results["fitness"].max()
        cycle_mean_fitness = self.last_validated_results["fitness"].mean()
        cycle_max_fitness = self.last_validated_results["fitness"].max()
        # Create strategy description
        if self.cycle == 0:
            strategy_desc = "Initial training data"
        else:
            strategy_desc = f"LLM agent selection cycle {self.cycle}: strategic sequence selection based on fitness landscape analysis"
        
        # Build cycle entry in expected format
        cycle_entry = {
            "cycle": self.cycle,
            "summary": f"Cycle {self.cycle} completed with {len(self.last_validated_results)} sequences validated. Mean fitness: {cycle_mean_fitness:.4f}, Max fitness: {cycle_max_fitness:.4f}",
            "mean_fitness": overall_mean_fitness,
            "max_fitness": overall_max_fitness,
            "oracle_rmse": float('nan'),  # Set to NaN as requested
            "strategies": [
                {
                    "strategy": strategy_desc,
                    "selected": self.last_validated_results["sequence"].tolist(),
                    "validated_fitness": self.last_validated_results["fitness"].tolist(),
                    "predicted_fitness": [0.0] * len(self.last_validated_results),  # No predictions in standalone mode
                    "predicted_std": [0.0] * len(self.last_validated_results),  # No predictions in standalone mode
                    "mean_fitness": cycle_mean_fitness,
                    "max_fitness": cycle_max_fitness,
                    "oracle_rmse": float('nan')
                }
            ]
        }
        
        # Add to campaign history
        self.campaign_history.append(cycle_entry)
        
        # Save complete campaign history in expected format
        with open(self.results_dir / "campaign_history.json", 'w') as f:
            json.dump(self.campaign_history, f, indent=2)
        
        # Also save individual cycle for compatibility
        with open(self.results_dir / f"cycle_{self.cycle}_summary.json", 'w') as f:
            json.dump(cycle_entry, f, indent=2)
        
        print(f"Saved results to {self.results_dir}")


def main():
    """Main function to run the standalone agent."""
    parser = argparse.ArgumentParser(description="Standalone LLM Agent for Protein Active Learning")
    parser.add_argument("--batch_size", type=int, required=True, help="Number of sequences per cycle")
    parser.add_argument("--total_budget", type=int, required=True, help="Total experimental budget")
    parser.add_argument("--random_seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--protein", type=str, required=True, choices=["GB1", "TrpB"], help="Target protein")
    parser.add_argument("--model", type=str, default="gpt-5", help="Model to use")
    parser.add_argument("--output_dir", type=str, default="results/", help="Output directory")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.total_budget <= args.batch_size:
        raise ValueError("total_budget must be greater than batch_size")
    
    print(f"Initializing standalone agent...")
    print(f"Protein: {args.protein}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total budget: {args.total_budget}")
    print(f"Random seed: {args.random_seed}")
    
    # Initialize and run agent
    agent = StandaloneAgent(
        protein=args.protein,
        batch_size=args.batch_size,
        total_budget=args.total_budget,
        random_seed=args.random_seed,
        model=args.model,
        output_dir=args.output_dir
    )
    
    start_time = time.time()
    agent.run_campaign()
    end_time = time.time()
    
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main() 