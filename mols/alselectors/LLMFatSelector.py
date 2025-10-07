"""
Selects candidates for verification using claude. First, it featurises the validated ligands using a LLM wit the gen_training_data_description_prompt.
Then, it selects the best candidates for verification using the following method:
- Compute number of tokens in the training data description
- Compute number of tokens left for the candidates
- Uses the approximation that 1 candidate, including prediction, confidence, and SMILES, and reasoning, is 50 tokens
- Computes largest possible number of candidates that can be analysed in a single chunk, N_max, given the number of tokens left (10 % margin)
- Then iterates over the chunks, asking the LLM to select the best candidates for validation, and appending the selected candidates to the selected_candidates dataframe, 
   making sure that len(chunk) + len(selected_candidates) <= N_max, and that N_select is proportional to len(chunk) in iteration of calling the LLM. 
- Returns the selected candidates as indicies of the inital dataset X_unlabeled
"""
import pandas as pd
from pathlib import Path
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Any, List, Tuple, Dict
import logging
import random
import re

from utils.Selector import Selector
from utils.APIUtils import LambdaAPI

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent  # Adjust based on your structure

class LLMFatSelector(Selector):
    """Selector that uses Claude to analyze chemical space and select candidates in chunks.
    
    This selector first generates a concise description of the validated chemical space using Claude,
    then processes candidates in chunks to stay within token limits. For each chunk, it asks Claude
    to select the best candidates based on the chemical space description and previous selections.
    """
    
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        """Initialize LLMFatSelector.
        
        Args:
            batch_size: Number of samples to select per AL cycle
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters
                - model: LLM model to use (default: llama-4-maverick-17b-128e-instruct-fp8)
                - max_input_tokens: Maximum input tokens (default: 800000)
                - tokens_per_candidate: Estimated tokens per candidate (default: 70)
                - temperature: Sampling temperature (default: 0.5)
        """
        super().__init__(batch_size, random_seed, **kwargs)
        
        # Set hyperparameters
        self.hyperparameters.update({
            'model': kwargs.get('model', 'deepseek-r1-0528'),
            'max_input_tokens': kwargs.get('max_input_tokens', 80000),
            'tokens_per_candidate': kwargs.get('tokens_per_candidate', 60)
        })
        
        # Initialize API client
        self.api_client = self._initialize_api_client()
        
        # Log initialization
        self.logger.info(f"Initialized LLMFatSelector with batch_size={batch_size}")
        self.logger.info(f"Token limits: max={self.hyperparameters['max_input_tokens']}, per_candidate={self.hyperparameters['tokens_per_candidate']}")
    
    def _initialize_api_client(self):
        """Initialize API client for Llama model."""
        return LambdaAPI(
            model=self.hyperparameters['model'],
            base_url="https://api.lambda.ai/v1",
            max_tokens=120000
        )
    
    def _compact_df(self, df, index=True, sig_digits=3):
        """
        Ultra-compact DataFrame display for token efficiency.
        Maintains structure while minimizing tokens.
        
        Args:
            df: DataFrame to display
            index: Whether to include row indices (default True)
            sig_digits: Significant digits for numeric columns (default 3)
        
        Returns:
            str: Compact string representation
        """
        def format_num(x):
            if pd.isna(x):
                return 'nan'
            if isinstance(x, (int, np.integer)):
                return str(x)
            if isinstance(x, (float, np.floating)):
                # Format to sig_digits significant figures
                if x == 0:
                    return '0'
                # Use scientific notation if very large/small, otherwise decimal
                if abs(x) >= 10**(sig_digits) or (abs(x) < 10**(-sig_digits+1) and x != 0):
                    return f"{x:.{sig_digits-1}e}"
                else:
                    # Round to sig_digits significant figures
                    return f"{x:.{max(0, sig_digits - 1 - int(np.floor(np.log10(abs(x)))))}f}".rstrip('0').rstrip('.')
            return str(x)
        
        # Get column names - use shortest reasonable abbreviations
        cols = list(df.columns)
        
        # Format header
        if index:
            header = "i|" + "|".join(cols)
        else:
            header = "|".join(cols)
        
        # Format rows
        rows = []
        for idx, row in df.iterrows():
            formatted_row = []
            if index:
                formatted_row.append(str(idx))
            
            for col in cols:
                val = row[col]
                if pd.api.types.is_numeric_dtype(type(val)):
                    formatted_row.append(format_num(val))
                else:
                    # For strings, keep as-is (mainly SMILES)
                    formatted_row.append(str(val))
            
            rows.append("|".join(formatted_row))
        
        return header + "\n" + "\n".join(rows)

    def _process_chunk(
        self,
        chunk: pd.DataFrame,
        labeled_data: pd.DataFrame,
        protein: str,
        cycle: int,
        total_cycles: int,
        oracle_name: str,
        current_chunk: int,
        total_chunks: int,
        budget: int
    ) -> List[int]:
        """Process a chunk of candidates using LLM.
        
        Args:
            chunk: DataFrame of candidates in current chunk
            selected_candidates: DataFrame of already selected candidates
            labeled_data: DataFrame of validated training data
            N_select: Number of candidates to select from this chunk
            N_remaining: Number of candidates in remaining chunks
            protein: Name of the protein
            cycle: Current AL cycle
            total_cycles: Total number of AL cycles
            oracle_name: Name of the oracle model
            current_chunk: Index of current chunk
            total_chunks: Total number of chunks
            budget: Total selection budget
            
        Returns:
            List of selected indices from this chunk
        """
        self.logger.info(f"Processing chunk {current_chunk}/{total_chunks}")
        self.logger.info(f"Chunk size: {len(chunk)}, Remaining budget: {budget}")

        labeled_data = labeled_data.sort_values(by='validated_RBFE', ascending=False)
        
        # Generate prompt for LLM
        prompt = f"""You are a chemoinformatic expert selecting ligands for experimental validation in an active learning campaign targeting {protein}. 
**CRITICAL OBJECTIVE:** Maximize the total number of truly high-affinity ligands discovered across all cycles. You are only interested in finding the very best compounds - those in the top fraction of all possible ligands. Higher RBFE values are better, but only the exceptional ones matter.
**Key Strategy Insight:** Since only the top fraction of compounds are valuable, it's crucial to test diverse scaffolds early rather than exploit too quickly. Most compounds will be mediocre regardless of predictions.

**Campaign Status:**
- Cycle: {cycle}/{total_cycles}
- Training data: {len(labeled_data)} validated ligands
- Oracle making predictions: Gaussian Process Regression

**Oracle Reliability Assessment:**
With only {len(labeled_data)} training examples, assess: How much should you trust GP predictions vs. your chemical intuition? Are RBFE predictions reliable across diverse scaffolds, or should chemical knowledge dominate early selections?

**Historical Data:**
Used to train the Gaussian Process Regression oracle, sorted by RBFE:
<validated_ligands>
{self._compact_df(labeled_data, index=False)}
</validated_ligands>

**Candidate Pool:**
Not ordered by importance.
<candidates>
{self._compact_df(chunk)}
</candidates>

**Task:** Select exactly {budget} candidates using integrated decision-making that combines predicted RBFE, standard deviation, and chemical intuition based on oracle maturity.
**Strategy:** Select candidates that optimize the combination of predicted value, uncertainty, and chemical rationale given the current campaign stage.

<reasoning>
1. **Oracle maturity**: Given {len(labeled_data)} training examples, how reliable are the predictions?
2. **Chemical patterns**: What SAR insights emerge from validated data? How is the chemical space explored?
3. **Integrated selection**: For 10 representative candidates, explain whether you're selecting based on: (a) confident high-activity prediction, (b) promising chemical features despite uncertainty, or (c) strategic exploration of chemical space.
4. **Campaign stage adaptation**: How is the selection strategy different at cycle {cycle}/{total_cycles} compared to previous and future cycles?
</reasoning>

<selected_indices>
[index1, index2, ...]
</selected_indices>

Select exactly {budget} candidates. Include one <reasoning> and one <selected_indices> section only."""
        
        # Get response from LLM
        self.logger.info("Sending chunk selection prompt to LLM")
        start_time = time.time()
        # Combine system prompt and user prompt
        for i in range(3):  
            try:
                message, input_tokens = self.api_client.pass_to_llm(prompt)
                response_time = time.time() - start_time
                self.logger.info(f"Received LLM response in {response_time:.2f} seconds")
                
                self.logger.info(f"Raw response:\n{message}")
                self.logger.info(f"Total usage: {input_tokens} input tokens")
                
                # Parse indices using the API utility
                selected_indices = self.api_client.parse_indicies(message, max(chunk.index), min(chunk.index))
                break
            except Exception as e:
                self.logger.error(f"Error in LLM call: {e}")
                time.sleep(1)
                if i == 2:
                    selected_indices = np.random.choice(chunk.index, size=budget, replace=False)
        return selected_indices

    
    def select(
        self,
        predictions: np.ndarray,
        confidence_scores: np.ndarray,
        training_data: Tuple[Dict[str, np.ndarray], np.ndarray],
        unlabeled_data: Dict[str, np.ndarray],
        random_seed: int,
        **kwargs: Any
    ) -> List[int]:
        """Select samples using LLM to analyze chemical space and process candidates in chunks.
        
        Args:
            predictions: Model predictions
            confidence_scores: Model confidence scores
            training_data: Current training set (features_dict, labels)
            unlabeled_data: Dictionary of unlabeled features, where keys are feature names and values are feature arrays.
                           Must contain 'SMILES' key with SMILES strings.
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters
                - cycle: Current AL cycle
                - total_cycles: Total number of AL cycles
                - oracle_name: Name of the oracle model
                - protein: Name of the protein
                
        Returns:
            List of selected sample indices
        """
        # Validate inputs
        if 'SMILES' not in unlabeled_data:
            raise ValueError("unlabeled_data must contain 'SMILES' key")
        
        n_samples = len(unlabeled_data['SMILES'])
        if n_samples < self.batch_size:
            self.logger.error(f"Not enough unlabeled samples ({n_samples}) for batch size {self.batch_size}")
            raise ValueError(
                f"Not enough unlabeled samples ({n_samples}) for batch size {self.batch_size}"
            )
        labeled_data = pd.DataFrame(training_data[0]['SMILES'], columns=['SMILES'])
        labeled_data['validated_RBFE'] = training_data[1]

        # Get required parameters
        cycle = kwargs.get('cycle', 1)
        total_cycles = kwargs.get('total_cycles', 6)
        oracle_name = kwargs.get('oracle_name', 'Unknown')
        protein = kwargs.get('protein', 'Unknown')
        
        self.logger.info(f"Starting selection for cycle {cycle}/{total_cycles}")
        self.logger.info(f"Protein: {protein}, Oracle: {oracle_name}")
        self.logger.info(f"Unlabeled pool size: {n_samples}")
        
        # Create DataFrame for candidates
        candidates_df = pd.DataFrame({
            'SMILES': unlabeled_data['SMILES'],
            'predicted_RBFE': predictions,
            'std_RBFE': 1/confidence_scores - 1
        })
        
        # Add additional features if available
        for feature_name, feature_values in unlabeled_data.items():
            if feature_name != 'SMILES' and feature_name != 'fingerprint':
                candidates_df[feature_name] = feature_values
        
        # Process chunks
        processed_unlabeled, n_selected = 0, 0
        n_data = len(candidates_df)
        selected_indices = []
        selected_candidates = pd.DataFrame(columns=['SMILES', 'predicted_RBFE', 'confidence'])
        current_chunk = 1
        max_candidates_in_chunk = self.hyperparameters['max_input_tokens'] / self.hyperparameters['tokens_per_candidate'] - len(labeled_data['SMILES'])
        min_chunks = int(np.ceil(n_data / max_candidates_in_chunk))

        while processed_unlabeled < n_data and current_chunk < 10000:
            # Calculate remaining budget and candidates
            new_points = int(min(max_candidates_in_chunk, n_data - processed_unlabeled))
                        
            chunk = candidates_df.iloc[processed_unlabeled:processed_unlabeled+new_points]

            # Process chunk
            chunk_indices = self._process_chunk(
                chunk=chunk,
                labeled_data=labeled_data,
                cycle=cycle,
                total_cycles=total_cycles,
                oracle_name=oracle_name,
                current_chunk=current_chunk,
                total_chunks=min_chunks,
                budget=int(min(max_candidates_in_chunk / min_chunks, self.batch_size)),
                protein=protein
            )

            # Update selected candidates
            selected_indices.extend(chunk_indices)
            selected_candidates = candidates_df.iloc[selected_indices]
            
            processed_unlabeled += new_points
            n_selected += len(chunk_indices)
            current_chunk += 1
            
            self.logger.info(f"Progress: {n_selected} selected, {processed_unlabeled}/{n_data} processed")
        
        final_indicies = self._process_chunk(
                chunk=selected_candidates,
                labeled_data=labeled_data,
                cycle=cycle,
                total_cycles=total_cycles,
                oracle_name=oracle_name,
                current_chunk=1,
                total_chunks=1, 
                budget=int(self.batch_size*1.1),
                protein=protein
            )
        if len(final_indicies) > self.batch_size:
            self.logger.warning(f"Dropped {len(final_indicies) - self.batch_size} candidates to keep batch size")
            final_indicies = final_indicies[:self.batch_size]

        self.logger.info(f"Selection complete: {len(final_indicies)} candidates selected")
        return final_indicies

class LLMFatHotSelector(LLMFatSelector):
    """Selector that uses Claude to analyze chemical space and select candidates in chunks.
    
    This selector first generates a concise description of the validated chemical space using Claude,
    then processes candidates in chunks to stay within token limits. For each chunk, it asks Claude
    to select the best candidates based on the chemical space description and previous selections.
    """
    
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        kwargs.update({
            'temperature': 0.8
        })
        super().__init__(batch_size, random_seed, **kwargs)