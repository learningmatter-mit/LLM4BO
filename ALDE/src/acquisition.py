from __future__ import annotations

import copy
import json
import os
import re
import sys
import warnings
from pathlib import Path
import traceback
from typing import TypedDict, List, Annotated, operator, Optional, Tuple, Any, Dict

import gpytorch
import numpy as np
import pandas as pd
import torch
import botorch
import yaml
import json
from botorch.acquisition.analytic import PosteriorMean
from botorch.utils.gp_sampling import get_gp_samples

# LLM and agent dependencies
try:
    from langchain_core.tools import tool, InjectedToolArg, BaseTool
    from langgraph.graph import MessagesState, StateGraph, START, END
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import ToolNode
    from langgraph.types import Send
    from pydantic import BaseModel, Field
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    
    # Flag indicating LLM dependencies are available
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    warnings.warn(f"LLM agent dependencies not available: {e}. AcquisitionAgent will not work.")

warnings.filterwarnings('ignore')

if AGENT_AVAILABLE:
    # Pydantic models for LLM structured output
    class FinalSelectionOutput(BaseModel):
        selected_indices: List[int] = Field(description="List of indices of selected candidates")
        report: str = Field(description="Brief report of rationale behind the selection of candidates")

    class StrategyOutput(BaseModel):
        strategies: List[str] = Field(description="List of strategies to implement")
        analysis: str = Field(description="Analysis of the current state of the AL campaign")

    # State management classes
    class Implementer(MessagesState):
        strategy: str
        reports: List[str]
        prev_selected: List[int]
        selected_indices: List[int]

    class StrategyState(TypedDict):
        AL_task: str
        report: str
        final_selection: List[List[int]]
        selected_indices: Annotated[List[List[int]], operator.add]
        strategies: List[str]
        analysis: str
        reports: Annotated[List[str], operator.add]
        batch_size: int

class Acquisition:
    """Generic class for acquisition functions that includes the function and
    its optimizer."""

    def __init__(self, acq_fn_name, domain, queries_x, norm_y, normalizer, disc_X, verbose, xi, seed_index, save_dir, device='cpu', batch_size: int = 1):
        """Initializes Acquisition object.
        Args:
            acq_fn_name: name of acquisition function ('GREEDY', 'UCB', or 'TS')
            domain: domain for acquisition function (only used if continuous design space)
            queries_x: already queried x values
            norm_y: already queried normalized y values
            normalizer: normalizer for y values
            disc_X: discrete domain for acquisition function
            verbose: verbosity level
            xi: parameter for UCB
            seed_index: index of seed
            device: device to use ('cuda' or 'cpu')
        """
        # Respect user's device preference and check availability
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
            self.gpu = True
        else:
            self.device = 'cpu'
            self.gpu = False

        self.acq = acq_fn_name
        self.queries_x = queries_x.double().to(self.device)
        self.nqueries = queries_x.shape[0]
        self.norm_y = norm_y.double()
        
        self.disc_X = disc_X.double()
        self.normalizer = normalizer
        self.verbose = verbose
        self.domain = domain # not used for a discrete domain
        self.xi = xi
        if self.acq.upper() == 'EI' and self.xi > 1: # <-- Use to default xi from UCB default value
            self.xi = 0.0
        self.seed_index = seed_index
        self.save_dir = save_dir
        self.batch_size = int(batch_size) if batch_size is not None else 1

        self.embeddings = None #embeddings from the neural network of the deep kernel
        self.preds = None #acquisition function values at each point in the discrete domain

    def get_next_query(self, samp_x, samp_y, samp_indices):
        """Returns the next sample to query."""

        self.preds[np.array(samp_indices, dtype=int)] = min(self.preds) #set the already queried values to the minumum of acquisition

        ind = int(np.argmax(self.preds))

        best_x = torch.reshape(self.disc_X[ind].detach(), (1, -1)).double()
        acq_val = self.preds[ind]
        best_idx = torch.tensor(ind)

        return best_x, acq_val, best_idx
    

class AcquisitionEnsemble(Acquisition):
    def __init__(self, acq_fn_name, domain, queries_x, norm_y, y_preds_full_all, normalizer, disc_X, verbose, xi, seed_index, save_dir, device='cpu', batch_size: int = 1):
        """
        Initializes Acquisition object for models that are ensembles (BOOSTING_ENSEMBLE or DNN_ENSEMBLE).
        Additionally takes in y_preds_full_all, the predictions of each model in the ensemble at each point in disc_X.
        """
        super().__init__(acq_fn_name, domain, queries_x, norm_y, normalizer, disc_X, verbose, xi, seed_index, save_dir, device, batch_size=batch_size)

        self.y_preds_full_all = y_preds_full_all

    def get_preds(self, X_pending):
        """
        Updates self.preds to be the acquisition function values at each point.
        X_pending are the previously queried points from the same batch, but is not used.
        """
        if self.acq.upper() == 'UCB':
            #alternatively could implement this as a direct value from one of the functions in the ensemble
            mu = torch.mean(self.y_preds_full_all, axis = 1)
            sigma = torch.std(self.y_preds_full_all, axis = 1)
            delta = (self.xi * torch.ones_like(mu)).sqrt() * sigma
            torch.save(sigma*self.normalizer, self.save_dir + '_' + str(self.nqueries) + 'sigma.pt')
            torch.save(mu*self.normalizer, self.save_dir + '_' + str(self.nqueries) + 'mu.pt')
            self.preds = mu + delta
        elif self.acq.upper() in ('GREEDY', 'EPSILON_GREEDY'):
            self.preds = torch.mean(self.y_preds_full_all, axis = 1)
            if self.acq.upper() == 'EPSILON_GREEDY':
                # Batch-aware epsilon-greedy: boost k random indices where k~Binomial(batch_size, epsilon)
                epsilon = 0.05
                k = int(np.random.binomial(self.batch_size, epsilon))
                if k > 0:
                    n = self.disc_X.shape[0]
                    k = min(k, int(n))
                    boost_idx = np.random.choice(np.arange(n), size=k, replace=False)
                    boosted_value = (self.preds.max().item() if torch.is_tensor(self.preds) else float(np.max(self.preds))) + 1.0
                    # ensure tensor indexing
                    if torch.is_tensor(self.preds):
                        self.preds[torch.tensor(boost_idx, dtype=torch.long)] = boosted_value
                    else:
                        tmp = torch.tensor(self.preds)
                        tmp[torch.tensor(boost_idx, dtype=torch.long)] = boosted_value
                        self.preds = tmp
        elif self.acq.upper() == 'TS':
            column = np.random.randint(self.y_preds_full_all.shape[1])
            self.preds = (self.y_preds_full_all[:, column])
        elif self.acq.upper() == 'EI':
            mu = torch.mean(self.y_preds_full_all, axis = 1)
            sigma = torch.std(self.y_preds_full_all, axis = 1)
            # Compute EI in normalized space
            best_norm_y = torch.max(self.norm_y)
            improvement = mu - best_norm_y - float(self.xi)
            min_std = 1e-12
            safe_sigma = torch.clamp(sigma, min=min_std)
            z = improvement / safe_sigma
            # Normal PDF and CDF approximations
            sqrt_2pi = np.sqrt(2.0 * np.pi)
            phi = torch.exp(-0.5 * z**2) / sqrt_2pi
            Phi = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
            ei = improvement * Phi + safe_sigma * phi
            # handle near-zero variance explicitly
            zero_mask = sigma <= min_std
            if torch.any(zero_mask):
                ei[zero_mask] = torch.clamp(improvement[zero_mask], min=0.0)
            # guard NaN/Inf
            ei = torch.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
            self.preds = ei
        
        self.preds = self.preds.detach().numpy()

class AcquisitionGP(Acquisition):
    def __init__(self, acq_fn_name, domain, queries_x, norm_y, model, normalizer, disc_X, verbose, xi, seed_index, save_dir, device='cpu', batch_size: int = 1):
        """
        Initializes Acquisition object for models that are based on Gaussian processes (GP_BOTORCH or DKL_BOTORCH).
        Additionally takes in the trained model object.
        """
        super().__init__(acq_fn_name, domain, queries_x, norm_y, normalizer, disc_X, verbose, xi, seed_index, save_dir, device, batch_size=batch_size)

        self.model = model.double().to(self.device)

    def get_embedding(self):
        """
        For DKL_BOTORCH, passes  all encodings in disc_X through the trained neural network layers of the model. Only necessary for thompson sampling acquisition (TS)
        Updates self.embeddings to be the embeddings of each point in disc_X.
        """

        if self.model.dkl and self.acq.upper() == 'TS':
            self.embeddings = self.model.embed_batched_gpu(self.disc_X).double()
        else:
            self.embeddings = self.disc_X
       
    def get_preds(self, X_pending):
        """
        Passes the encoded values in disc_X through the acquisition function.
        Updates self.preds to be the acquisition function values at each point.
        X_pending are the previously queried points from the same batch, but is not used.
        """
        model = copy.copy(self.model).to(self.device)

        #Thompson Sampling
        if self.acq.upper() in ('TS'):
            #Deep Kernel
            if self.model.dkl:
                inputs = model.train_inputs[0].to(self.device)
                nn_x = model.embedding(inputs)
                model.train_inputs = (nn_x,)
            #GP
            else:
                model.train_inputs = (self.model.train_inputs[0],)
            
            #Sample a random function from the posterior
            gp_sample = get_gp_samples(
                    model=model,
                    num_outputs=1,
                    n_samples=1,
                    num_rff_features=1000,
            )
            self.acquisition_function = PosteriorMean(model=gp_sample)
            
            self.preds = self.model.eval_acquisition_batched_gpu(self.embeddings, f=self.max_obj).cpu().detach().double()
        #For UCB and Greedy
        else: 
            with gpytorch.settings.fast_pred_var(), torch.no_grad():
                mu, sigma = model.predict_batched_gpu(self.embeddings)

            if self.acq.upper() == 'UCB':
                delta = (self.xi * torch.ones_like(mu)).sqrt() * sigma

                #save for uncertainty quantification
                torch.save(sigma*self.normalizer, self.save_dir + '_' + str(self.nqueries) + 'sigma.pt')
                torch.save(mu*self.normalizer, self.save_dir + '_' + str(self.nqueries) + 'mu.pt')

                self.preds = mu + delta
            elif self.acq.upper() in ('GREEDY', 'EPSILON_GREEDY'):
                self.preds = mu.cpu()
            elif self.acq.upper() == 'EI':
                # Compute EI in normalized space using mu, sigma
                best_norm_y = torch.max(self.norm_y)
                improvement = mu - best_norm_y - float(self.xi)
                min_std = 1e-12
                safe_sigma = torch.clamp(sigma, min=min_std)
                z = improvement / safe_sigma
                sqrt_2pi = np.sqrt(2.0 * np.pi)
                phi = torch.exp(-0.5 * z**2) / sqrt_2pi
                Phi = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
                ei = improvement * Phi + safe_sigma * phi
                zero_mask = sigma <= min_std
                if torch.any(zero_mask):
                    ei[zero_mask] = torch.clamp(improvement[zero_mask], min=0.0)
                ei = torch.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
                self.preds = ei
        self.preds = self.preds.detach().numpy()

    def max_obj(self, x):
        """
        Acquisition function to maximize.
        """
        return self.acquisition_function.forward(x.reshape((x.shape[0], 1, x.shape[1])))


# Agent utility functions
if AGENT_AVAILABLE:
    def _set_env(var: str):
        """Set environment variable if not already set."""
        if not os.environ.get(var):
            import getpass
            os.environ[var] = getpass.getpass(f"{var}: ")

    def tensor_to_sequence_df(disc_X: torch.Tensor, disc_y: torch.Tensor, 
                              model_predictions: torch.Tensor, model_std: torch.Tensor,
                              protein_sequences: List[str]) -> pd.DataFrame:
        """Convert tensor data to DataFrame format expected by the agent."""
        return pd.DataFrame({
            'sequence': protein_sequences,
            'predictions': model_predictions.cpu().numpy(),
            'std_predictions': model_std.cpu().numpy()
        })

    def sequence_df_to_indices(selected_sequences: List[str], 
                               all_sequences: List[str]) -> List[int]:
        """Convert selected sequences back to indices in the original tensor."""
        indices = []
        for seq in selected_sequences:
            try:
                idx = all_sequences.index(seq)
                indices.append(idx)
            except ValueError:
                print(f"Warning: Selected sequence {seq} not found in sequence list")
        return indices


class AcquisitionAgent(Acquisition):
    """
    LLM-based acquisition function that uses multi-agent reasoning for protein sequence selection.
    
    This acquisition function integrates an intelligent LLM agent that analyzes the current 
    active learning state and generates sophisticated selection strategies. It's designed to 
    work alongside traditional acquisition functions while providing interpretable, strategy-driven
    sequence selection for protein optimization campaigns.
    """
    
    def __init__(self, acq_fn_name, domain, queries_x, norm_y, normalizer, disc_X, verbose, xi, seed_index, save_dir,
                 protein_sequences: List[str], model_predictions: torch.Tensor = None, model_std: torch.Tensor = None,
                 protein: str = "GB1", cycle: int = 1, total_cycles: int = 5,
                 oracle_model: str = "ENSEMBLE", batch_size: int = 96, summaries_file: str = None, device: str = 'cpu'):
        """
        Initialize the agent-based acquisition function.
        
        Args:
            acq_fn_name: Must be 'AGENT' to use this acquisition function
            domain: Domain bounds (inherited from base class)
            queries_x: Already queried x values (encoded sequences)
            norm_y: Already queried normalized y values  
            normalizer: Normalizer for y values
            disc_X: All possible encoded sequences in the discrete domain
            verbose: Verbosity level
            xi: UCB parameter (not used by agent, but kept for compatibility)
            seed_index: Random seed index
            save_dir: Directory to save results
            protein_sequences: List of all protein sequences corresponding to disc_X
            model_predictions: Model predictions for all sequences in disc_X
            model_std: Model uncertainties for all sequences in disc_X
            protein: Protein name ('GB1' or 'TrpB') for prompt configuration
            cycle: Current AL cycle number
            total_cycles: Total number of AL cycles planned
            oracle_model: Name of the oracle model being used
            batch_size: Number of sequences to select
            summaries_file: Path to file for storing cycle summaries
            device: Device to use ('cuda' or 'cpu')
        """
        if not AGENT_AVAILABLE:
            raise ImportError("LLM agent dependencies not available. Install langchain, langgraph, and pydantic.")
        
        if acq_fn_name.upper() != 'AGENT':
            raise ValueError("AcquisitionAgent requires acq_fn_name='AGENT'")
            
        super().__init__(acq_fn_name, domain, queries_x, norm_y, normalizer, disc_X, verbose, xi, seed_index, save_dir, device)
        
        # Agent-specific parameters
        self.protein_sequences = protein_sequences
        self.model_predictions = model_predictions
        self.model_std = model_std
        self.protein = protein
        self.cycle = cycle
        self.total_cycles = total_cycles
        self.oracle_model = oracle_model
        self.batch_size = batch_size
        self.summaries_file = summaries_file or f"{save_dir}_summaries.json"
        
        # Validate inputs
        if len(protein_sequences) != disc_X.shape[0]:
            raise ValueError(f"protein_sequences length ({len(protein_sequences)}) must match disc_X rows ({disc_X.shape[0]})")
        
        if model_predictions is not None and len(model_predictions) != len(protein_sequences):
            raise ValueError(f"model_predictions length must match protein_sequences length")
            
        if model_std is not None and len(model_std) != len(protein_sequences):
            raise ValueError(f"model_std length must match protein_sequences length")
        
        # Set up environment variables for LLM access
        _set_env("LANGSMITH_API_KEY")
        os.environ.setdefault('LANGSMITH_TRACING', "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "ALDE")
        os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    
    def get_preds(self, X_pending):
        """
        Run the LLM agent workflow to select sequences and generate predictions.
        
        This method converts the tensor data to DataFrame format, runs the multi-agent
        LLM workflow, and returns acquisition scores for all sequences.
        """
        try:
            # Convert tensors to DataFrame format for the agent
            pred_df = self._prepare_prediction_dataframe()
            train_df = self._prepare_training_dataframe()
            train_seqs = set(train_df['sequence'].values)
            pred_df = pred_df[~pred_df['sequence'].isin(train_seqs)]
            
            if self.verbose >= 2:
                print(f"Running LLM agent for cycle {self.cycle}")
                print(f"Prediction space: {len(pred_df)} sequences")
                print(f"Training data: {len(train_df)} sequences")
            
            # Run the agent workflow
            final_selection, summary, strategies = self._run_agent_workflow(train_df, pred_df)
            
            # Convert agent output to acquisition scores
            self.preds = self._convert_agent_output_to_scores(final_selection, pred_df)
            
            if self.verbose >= 1:
                print(f"Agent selected {len([item for sublist in final_selection for item in sublist])} sequences")
                print(f"Generated {len(strategies)} strategies")
                
        except Exception as e:
            if self.verbose >= 1:
                print(f"Agent failed with error: {e}")
                print(traceback.format_exc())
            raise e
    
    def _prepare_prediction_dataframe(self) -> pd.DataFrame:
        """Convert disc_X tensor data to DataFrame format expected by agent."""
        n_sequences = len(self.protein_sequences)
        
        # Use provided model predictions or generate fallback values
        predictions = self.model_predictions.cpu().numpy()
        std_predictions = self.model_std.cpu().numpy()
        
        return pd.DataFrame({
            'sequence': self.protein_sequences,
            'predictions': predictions,
            'std_predictions': std_predictions
        })
    
    def _prepare_training_dataframe(self) -> pd.DataFrame:
        """Provide basic training info - agent will build complete historical data from summaries."""
        # Minimal training data structure for interface compatibility
        # Agent will build complete training data from summaries file
        queried_sequences = []
        queried_fitness = []
        
        # Simple distance-based mapping for current queries
        for i, x_query in enumerate(self.queries_x):
            distances = torch.sum((self.disc_X - x_query.cpu())**2, dim=1)
            closest_idx = torch.argmin(distances).item()
            queried_sequences.append(self.protein_sequences[closest_idx])
            queried_fitness.append((self.norm_y[i] * self.normalizer).item())
        
        # Return minimal interface data - agent builds complete data from summaries
        return pd.DataFrame({
            'sequence': queried_sequences,
            'fitness': queried_fitness
        })
    
    def _run_agent_workflow(self, train_df: pd.DataFrame, pred_df: pd.DataFrame):
        """Execute the full LLM agent workflow."""
        from src.agent import run_selector_agent  # Import from src folder
        
        return run_selector_agent(
            oracle_model=self.oracle_model,
            batch_size=self.batch_size,
            total_cycles=self.total_cycles,
            cycle=self.cycle,
            train_df=train_df,
            pred_df=pred_df,
            protein=self.protein,
            summaries_file=self.summaries_file,
            include_sequences=True
        )
    
    def _convert_agent_output_to_scores(self, final_selection: List[List[int]], pred_df: pd.DataFrame) -> np.ndarray:
        """Convert agent's selected indices to binary acquisition scores."""
        scores = np.zeros(len(self.protein_sequences))
        
        # Flatten the nested selection structure
        selected_indices = []
        for strategy_indices in final_selection:
            if isinstance(strategy_indices, list):
                selected_indices.extend(strategy_indices)
            else:
                selected_indices.append(strategy_indices)
        # Convert to set to remove duplicates, then back to list
        selected_indices = list(set(selected_indices))

        # Assign binary scores: 1.0 for selected sequences, 0.0 for others
        for idx in selected_indices:
            if  0 <= idx < len(scores):
                scores[int(idx)] = 1.0  # High acquisition score for selected sequences
            else:
                print(f"Warning: Selected index {idx} is out of range")
        
        # Add small random noise to break ties in acquisition selection
        scores += np.random.random(len(scores)) * 0.001
        return scores

    def _load_prompts_config(self):
        """Load prompts configuration from YAML file."""
        config_path = Path(__file__).parent / 'prompts_LLMChainSelector.yaml'  # Use relative path from src folder
        
        if not config_path.exists():
            raise FileNotFoundError(f"Prompts config file not found at {config_path}")
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


class AcquisitionSimpleAgent(AcquisitionAgent):
    """
    Simple LLM-based acquisition function that excludes sequence data from prompts.
    
    This is identical to AcquisitionAgent except that it doesn't include the actual
    sequence data in the strategy prompts, providing a cleaner comparison for testing
    the effect of sequence visibility on agent decision-making.
    """
    
    def __init__(self, acq_fn_name, domain, queries_x, norm_y, normalizer, disc_X, verbose, xi, seed_index, save_dir,
                 protein_sequences: List[str], model_predictions: torch.Tensor = None, model_std: torch.Tensor = None,
                 protein: str = "GB1", cycle: int = 1, total_cycles: int = 5,
                 oracle_model: str = "ENSEMBLE", batch_size: int = 96, summaries_file: str = None, device: str = 'cpu'):
        """Initialize the simple agent-based acquisition function."""
        
        if not AGENT_AVAILABLE:
            raise ImportError("LLM agent dependencies not available. Install langchain, langgraph, and pydantic.")
        
        if acq_fn_name.upper() != 'SIMPLEAGENT':
            raise ValueError("AcquisitionSimpleAgent requires acq_fn_name='SIMPLEAGENT'")
        
        # Call Acquisition constructor directly
        Acquisition.__init__(self, acq_fn_name, domain, queries_x, norm_y, normalizer, disc_X, verbose, xi, seed_index, save_dir, device)
        
        # Agent-specific parameters (copied from AcquisitionAgent)
        self.protein_sequences = protein_sequences
        self.model_predictions = model_predictions
        self.model_std = model_std
        self.protein = protein
        self.cycle = cycle
        self.total_cycles = total_cycles
        self.oracle_model = oracle_model
        self.batch_size = batch_size
        self.summaries_file = summaries_file or f"{save_dir}_summaries.json"
        
        # Validate inputs (copied from AcquisitionAgent)
        if len(protein_sequences) != disc_X.shape[0]:
            raise ValueError(f"protein_sequences length ({len(protein_sequences)}) must match disc_X rows ({disc_X.shape[0]})")
        
        if model_predictions is not None and len(model_predictions) != len(protein_sequences):
            raise ValueError(f"model_predictions length must match protein_sequences length")
            
        if model_std is not None and len(model_std) != len(protein_sequences):
            raise ValueError(f"model_std length must match protein_sequences length")
        
        # Set up environment variables for LLM access
        _set_env("LANGSMITH_API_KEY")
        os.environ.setdefault('LANGSMITH_TRACING', "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "ALDE")
        os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    
    def _run_agent_workflow(self, train_df: pd.DataFrame, pred_df: pd.DataFrame):
        """Execute the full LLM agent workflow without sequence data in prompts."""
        from src.agent import run_selector_agent  # Import from src folder
        
        return run_selector_agent(
            oracle_model=self.oracle_model,
            batch_size=self.batch_size,
            total_cycles=self.total_cycles,
            cycle=self.cycle,
            train_df=train_df,
            pred_df=pred_df,
            protein=self.protein,
            summaries_file=self.summaries_file,
            include_sequences=False  # This is the only difference from AcquisitionAgent
        )


# Convenience function for integration with existing codebase
def create_acquisition(acq_fn_name: str, *args, **kwargs):
    """
    Factory function to create the appropriate acquisition function.
    
    This function provides a unified interface for creating acquisition functions,
    automatically selecting the right class based on the acquisition function name.
    """
    acq_fn_name = acq_fn_name.upper()
    
    if acq_fn_name == 'AGENT':
        return AcquisitionAgent(acq_fn_name, *args, **kwargs)
    elif acq_fn_name == 'SIMPLEAGENT':
        return AcquisitionSimpleAgent(acq_fn_name, *args, **kwargs)
    else:
        # Return base Acquisition class for compatibility
        # In practice, this would route to AcquisitionEnsemble or AcquisitionGP
        return Acquisition(acq_fn_name, *args, **kwargs)