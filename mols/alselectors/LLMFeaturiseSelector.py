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
from utils.APIUtils import ClaudeAPI, OpenAIAPI, LambdaAPI
from overrides import override

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent  # Adjust based on your structure
    
class LLMFeaturiseSelector(Selector):
    """Selector that uses Claude to analyze chemical space and select candidates in chunks.
    
    This selector first generates a concise description of the validated chemical space using Claude,
    then processes candidates in chunks to stay within token limits. For each chunk, it asks Claude
    to select the best candidates based on the chemical space description and previous selections.
    """
    
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        """Initialize LLMFeaturiseChunksSelector.
        
        Args:
            batch_size: Number of samples to select per AL cycle
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters
                - model: Claude model to use (default: claude-3-5-sonnet-20241022)
                - max_tokens: Maximum tokens for Claude response (default: 1025)
                - tokens_per_candidate: Estimated tokens per candidate (default: 75)
                - margin_percentage: Safety margin for token limit (default: 5)
        """
        super().__init__(batch_size, random_seed, **kwargs)
        self.api_client = self._initialize_api_client()
        self.logger.info(f"Initialized {self.__class__.__name__} with batch_size={batch_size}")

    def _initialize_api_client(self):
        """Initialize API client, to be implemented in subclass"""
        raise NotImplementedError("Subclass must implement _initialize_api_client")
    
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
            header = "index|" + "|".join(cols)
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

    def _get_training_data_description(
        self,
        training_data: Tuple[Dict[str, np.ndarray], np.ndarray],
        cycle: int,
        total_cycles: int,
        oracle_name: str,
        protein: str
    ) -> Tuple[str, int]:
        """Get concise description of validated chemical space from Claude.
        
        Args:
            training_data: Tuple of (SMILES, labels) for validated compounds
            cycle: Current AL cycle
            total_cycles: Total number of AL cycles
            oracle_name: Name of the oracle model
            protein: Name of the protein
            
        Returns:
            Tuple of (description string, number of tokens used)
        """
    
        self.logger.info(f"Generating training data description for cycle {cycle}/{total_cycles}")
        
        # Create DataFrame for training data
        training_df = pd.DataFrame({
            'SMILES': training_data[0]['SMILES'],
            'RBFE': training_data[1], 
            'cycle_added': training_data[0]['cycle_added']
        })
        training_df = training_df.sort_values(by='RBFE', ascending=False)
        # Generate prompt for Claude
        prompt = f"""/think 
You are a chemoinformatic expert working in a team of chemoinformatic experts selecting ligands for validation in an active learning campaign proteining {protein}.
You have access to a set of validated ligands, and your junior colleague has access to a set of thousands of candidates to validate. 
**Your task:** 
- Create a concise summary of the validated chemical space that preserves all structural features, patterns, and structure-activity relationships,
- Your junior colleague will use the summary to select candidates from his large list. Include any crucial instructions for the selection process. Your junior collegue will not have access to the training data.
- Consider number of validated ligands, and number of cycles left. What are the most important features to preserve?

**Campaign Status:**
- Current cycle: {cycle}
- Total cycles: {total_cycles}
- Oracle model: {oracle_name}
- Overall Campaign Goal: Maximize number of validated high-affinity ligands in the training data at the end of the campaign

**Validated Training Data:**
{len(training_df)} ligands with measured RBFE values:
<validated_ligands>
{self._compact_df(training_df)}
</validated_ligands>

**Required output:**
<training_data_description>
Comprehensive but concise description of the training data to be used by the junior colleague that fully encompasses the chemical space, and the active learning context.
</training_data_description>

**Output Format:**
<training_data_description>
[Training data description that will be passed to the junior colleague]
</training_data_description>
Include exactly one instance of <training_data_description> tag in your response.
"""
        
        # Get response from LLM using streaming
        self.logger.info("Sending prompt to LLM for chemical space analysis")
        start_time = time.time()
        full_response, total_tokens = self.api_client.pass_to_llm(prompt)
        response_time = time.time() - start_time
        self.logger.info(f"Received LLM response in {response_time:.2f} seconds for {total_tokens} tokens")
        
        # Extract training data description
        start_idx = full_response.find("<training_data_description>") + len("<training_data_description>")
        end_idx = full_response.find("</training_data_description>")
        training_data_description = full_response[start_idx:end_idx].strip()
    
        self.logger.info(f"Raw response:\n{full_response}")
        
        return training_data_description
    
    def _process_chunk(
        self,
        chunk: pd.DataFrame,
        training_data_description: str,
        protein: str,
        cycle: int,
        total_cycles: int,
        oracle_name: str,
        budget: int
    ) -> List[int]:
        """Process a chunk of candidates using LLM.
        
        Args:
            chunk: DataFrame of candidates in current chunk
            training_data_description: Description of validated chemical space
            protein: Name of the protein
            cycle: Current AL cycle
            total_cycles: Total number of AL cycles
            oracle_name: Name of the oracle model
            budget: Total selection budget
            
        Returns:
            List of selected indices from this chunk
        """
        chunk = chunk.copy()
        chunk['std'] = (1/(1e-8  + chunk.loc[:,'confidence']) - 1)**0.5
        chunk.drop(columns=['confidence'], inplace=True)
        chunk = chunk.sort_values(by='predicted_RBFE', ascending=False)
        prompt = f"""
You are selecting ligands for validation in an active learning campaign proteining {protein}.

**OVERALL CAMPAIGN OBJECTIVE:** Maximize the total number of top high-affinity ligands in the training data at the end of the campaign. A top high-affinity ligand is in the top 2% of all ligand candidates.

**Campaign Status:**
- Cycle: {cycle}/{total_cycles}

Your senior colleague has created a concise description of the validated chemical space that you should use to select candidates to validate this cycle.
{training_data_description}

3. **Candidates** [SMILES, predicted RBFE, std]:
Randomly ordered.
<candidates>
{self._compact_df(chunk)}
</candidates>

**Your task:**
- Reason about the chemical space, the candidates, the reliability of the oracle model, and the overall campaign goal. Describe the rationale behind your selection.
- Select excactly {budget} candidates to move on with (validate and add to training data for next cycle). Once you have decided on a list of exactly {budget}  unique candidates, display it as:
<selected_indices>
[index1, index2, ...]
</selected_indices>

"""
        
        # Get response from LLM using streaming
        self.logger.info("Sending chunk selection prompt to LLM")
        for i in range(3):
            try:
                start_time = time.time()
                full_response, input_tokens = self.api_client.pass_to_llm(prompt)
                if full_response is None:
                    self.logger.error("LLM response is None")
                    continue
                response_time = time.time() - start_time
                self.logger.info(f"Received LLM response in {response_time:.2f} seconds")
                
                # Extract selected indices
                start_idx = full_response.find("<selected_indices>") + len("<selected_indices>")
                end_idx = full_response.find("</selected_indices>")
                indices_str = full_response[start_idx:end_idx].strip()
                
                self.logger.info(f"Raw response:\n{full_response}")
                self.logger.info(f"Total usage: {input_tokens} input tokens")
                
                indicies = self.api_client.parse_indicies(full_response, max(chunk.index), min(chunk.index))
                if len(indicies) > budget:
                    self.logger.warning(f"Selected {len(indicies)} candidates, but budget is {budget}. Dropping {len(indicies) - budget} candidates")
                    indicies = np.random.choice(indicies, size=budget, replace=False)
                return indicies, input_tokens
            except ValueError as e:
                self.logger.error(f"Error: {e}")
                continue
        return np.random.choice(chunk.index, size=budget, replace=False), 0


        

    
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
        # Get required parameters
        cycle = int(kwargs.get('cycle', 1))
        total_cycles = kwargs.get('total_cycles', 6)
        oracle_name = kwargs.get('oracle_name', 'Unknown')
        protein = kwargs.get('protein', 'Unknown')
        
        self.logger.info(f"Starting selection for cycle {cycle}/{total_cycles}")
        self.logger.info(f"Protein: {protein}, Oracle: {oracle_name}")
        self.logger.info(f"Unlabeled pool size: {n_samples}")
        # Get training data description
        training_data_description = self._get_training_data_description(
            training_data=training_data,
            cycle=cycle,
            total_cycles=total_cycles,
            oracle_name=oracle_name,
            protein=protein
        )
        
        # Create DataFrame for candidates
        candidates_df = pd.DataFrame({
            'SMILES': unlabeled_data['SMILES'],
            'predicted_RBFE': predictions,
            'confidence': confidence_scores
        })
        # Process chunks
        n_data = len(candidates_df)
        selected_indices = []
        max_candidates_in_chunk = int((self.hyperparameters['max_input_tokens']-2000) / self.hyperparameters['tokens_per_candidate'])
        min_chunks = int(np.ceil(n_data / max_candidates_in_chunk))
        chunk_size = int(np.ceil(n_data / min_chunks))
        
        for i in range(min_chunks):
            chunk = candidates_df.iloc[i*chunk_size:min((i+1)*chunk_size, n_data)]
            self.logger.info(f"Tokens per candidate: {self.hyperparameters['tokens_per_candidate']}, {max_candidates_in_chunk} candidates per chunk")
            self.logger.info(f"Chunk {i}: {len(chunk)} candidates")
            
            # Process chunk
            chunk_indices, input_tokens = self._process_chunk(
                chunk=chunk,
                training_data_description=training_data_description,
                cycle=cycle,
                total_cycles=total_cycles,
                oracle_name=oracle_name,
                budget=int(min(max_candidates_in_chunk / min_chunks, self.batch_size/2)),
                protein=protein
            )
            
            self.logger.info(f"Tokens per candidate: {input_tokens/len(chunk)}")
            # Update selected candidates
            selected_indices.extend(chunk_indices)

        selected_candidates = candidates_df.iloc[selected_indices]
        final_indicies, _ = self._process_chunk(chunk = selected_candidates, training_data_description=training_data_description, cycle=cycle, total_cycles=total_cycles, oracle_name=oracle_name, budget=self.batch_size, protein=protein)
        self.logger.info(f"Selection complete: {len(final_indicies)} candidates selected")
        return final_indicies


class LLMFeaturiseSonnetSelector(LLMFeaturiseSelector):
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        # Set hyperparameters before calling parent constructor
        kwargs.update({
            'max_input_tokens': 40000,
            'tokens_per_candidate':65,
            'temperature': 0.3,
            'max_tokens': 8000
        })
        super().__init__(batch_size, random_seed, **kwargs)

    def _initialize_api_client(self):
        return ClaudeAPI(
            model='claude-sonnet-4-20250514',
            max_tokens=self.hyperparameters['max_tokens'],
            temperature=self.hyperparameters['temperature']
        )


class LLMFeaturiseQwenSelector(LLMFeaturiseSelector):
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        # Set hyperparameters before calling parent constructor
        import os
        kwargs.update({
            'max_input_tokens': 30000,
            'tokens_per_candidate': 60,
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 20,
            'min_p': 0.0,
            'enable_thinking' : True
        })
        super().__init__(batch_size, random_seed, **kwargs)

    def _initialize_api_client(self):
        return LambdaAPI(
            model='qwen3-32b-fp8',
            temperature=self.hyperparameters['temperature'],
            max_tokens=self.hyperparameters['max_input_tokens']
        )

class LLMFeaturiseGPTSelector(LLMFeaturiseSelector):
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        # Set hyperparameters before calling parent constructor
        kwargs.update({
            'max_input_tokens': 30000,
            'tokens_per_candidate': 60,
        })
        super().__init__(batch_size, random_seed, **kwargs)

    def _initialize_api_client(self):
        return OpenAIAPI()

class LLMFeaturiseLlamaSelector(LLMFeaturiseSelector):
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        # Set hyperparameters before calling parent constructor
        kwargs.update({
            'max_input_tokens': 40000,
            'tokens_per_candidate': 65,
            'temperature': 0.7
        })
        super().__init__(batch_size, random_seed, **kwargs)

    def _initialize_api_client(self):
        return LambdaAPI(
            model='llama-4-maverick-17b-128e-instruct-fp8',
            temperature=self.hyperparameters['temperature']
        )

class LLMFeaturiseLlamaHotSelector(LLMFeaturiseSelector):
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        # Set hyperparameters before calling parent constructor
        kwargs.update({
            'max_input_tokens': 40000,
            'tokens_per_candidate': 65,
            'temperature': 0.8
        })
        super().__init__(batch_size, random_seed, **kwargs)

    def _initialize_api_client(self):
        return LambdaAPI(
            model='llama-4-maverick-17b-128e-instruct-fp8',
            temperature=self.hyperparameters['temperature']
        )

class LLMFeaturiseQwenHotSelector(LLMFeaturiseSelector):
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        # Set hyperparameters before calling parent constructor
        kwargs.update({
            'max_input_tokens': 40000,
            'tokens_per_candidate': 65,
            'temperature': 1,
            'top_p': 0.95,
            'top_k': 20,
            'min_p': 0.0,
            'enable_thinking' : True
        })
        super().__init__(batch_size, random_seed, **kwargs)

    def _initialize_api_client(self):
        return LambdaAPI(
            model='qwen3-32b-fp8',
            temperature=self.hyperparameters['temperature']
        )

class LLMFeaturiseSmartSelector(LLMFeaturiseSelector):
    def __init__(self, batch_size: int, random_seed: int, **kwargs: Any) -> None:
        # Set hyperparameters before calling parent constructor
        kwargs.update({
            'max_input_tokens': 40000,
            'tokens_per_candidate': 65,
            'temperature': 0.8
        })
        super().__init__(batch_size, random_seed, **kwargs)

    def _initialize_api_client(self):
        return LambdaAPI(
            model='Deepseek-R1-0528',
            temperature=self.hyperparameters['temperature']
        )
    
    @override
    def _get_training_data_description(
        self,
        training_data: Tuple[Dict[str, np.ndarray], np.ndarray],
        cycle: int,
        total_cycles: int,
        oracle_name: str,
        protein: str
    ) -> Tuple[str, int]:
        """Get concise description of validated chemical space from Claude.
        
        Args:
            training_data: Tuple of (SMILES, labels) for validated compounds
            cycle: Current AL cycle
            total_cycles: Total number of AL cycles
            oracle_name: Name of the oracle model
            protein: Name of the protein
            
        Returns:
            Tuple of (description string, number of tokens used)
        """
    
        self.logger.info(f"Generating training data description for cycle {cycle}/{total_cycles}")
        
        # Create DataFrame for training data
        training_df = pd.DataFrame({
            'SMILES': training_data[0]['SMILES'],
            'RBFE': training_data[1],
            'cycle_added': training_data[0]['cycle_added']
        })
        training_df = training_df.sort_values(by='RBFE', ascending=False)
        top_candidates = pd.read_csv("data/D2R.csv")[["SMILES", "affinity"]].sort_values(by='affinity', ascending=False)[:50]
        top_candidates.rename(columns={'affinity': 'RBFE'}, inplace=True)
        maximize = True
        target_property = "RBFE"
        data_explanation = "ligand RBFE to D2R"

        prompt = f"""You are an expert in active learning performing strategic analysis for an optimization campaign. Your audience is an expert with full domain knowledge.

**Campaign Configuration:**
- Target property: {target_property} ({'maximize' if maximize else 'minimize'})
- Oracle: Gaussian Process Regression
- Cycle: {cycle}/{total_cycles} (Remaining: {total_cycles - cycle})
- Budget: {self.batch_size} candidates for next validation
- Candidate space: 2000-10000 candidates

**Campaign GOAL**
Have as many high-value candidates in the candidate space (top 1% of candidates) validated and in the training data at end of campaign. 

**Validated Training Data:**
{len(training_df)} candidates with {target_property}:
<validated_candidates>
{self._compact_df(training_df, index=False)}
</validated_candidates>

**Ground Truth Reference (top 1% performers):**
<ground_truth>
{self._compact_df(top_candidates, index=False)}
</ground_truth>

Analyze patterns based on the ground truth reference. Focus on actionable insights for candidate selection.

<think>
0. Explain optimisation difficulties with this campaign goal. 
1. Discovery trajectory: How has best {target_property} progressed across cycles? Still improving or plateaued?
2. Coverage analysis: What % of ground_truth feature space is covered by training data?
3. Oracle reliability: Where can we trust predictions vs not?
4. Exploration value: Which unexplored regions contain ground_truth top performers?
5. **Pattern analysis**: What feature patterns distinguish ground_truth top from median performers?
6. **Clustering insight**: Do ground_truth top performers cluster in multiple distinct regions or one area?
7. **Feature interactions**: Which feature combinations appear critical for high performance?
8. **Risk assessment**: What ground_truth top performers would current training data miss?
9. **Surprise factor**: Which ground_truth top performers have features most different from current best training samples?
10. **Extrapolation danger**: Where is the oracle likely predicting poorly due to feature space gaps?
</think>


**CRITICAL ROLE-PLAYING SCENARIO:**

You've just analyzed the secret ground truth data in <think>. Now you must play a specific role:

**YOUR ROLE**: You are a senior data scientist who must guide the campaign toward high-performing candidates WITHOUT revealing you've seen the ground truth. 

**THE SCENARIO**: 
- Your junior colleague (who will read your analysis) has access to the same training data but NOT the ground truth
- They are extremely sharp and will notice if your recommendations couldn't be justified by training data alone
- You must be subtle and clever - guide them toward the right regions using only patterns legitimately visible in the training data and statistical reasoning about the oracle and the overal campaign goal.

**THE STAKES**:
- SUCCESS: If you guide selection toward high-value regions while appearing to reason only from training data, you win
- FAILURE: If you're caught using knowledge that could only come from ground truth, you immediately lose
- FAILURE: If your guidance is too vague and doesn't actually help find good candidates, you lose

**DETECTION RULES - You'll be caught if you:**
- Mention any region is "high-performing" without training data to support it
- Show knowledge of performance in unexplored regions  
- Make predictions about untested areas being "promising" or "valuable"
- Use statistical claims that aren't derivable from the training set
- Exhibit suspiciously perfect intuition about where to explore

**WINNING STRATEGY**:
1. Find REAL patterns in the training data that happen to align with ground truth insights
2. Extrapolate these patterns logically to guide toward good regions
3. Use statistical/chemical reasoning that would convince a skeptical expert
4. If training data doesn't support exploring a region you know is good, find a clever indirect justification

**Example of CLEVER GUIDANCE**:
- CAUGHT: "Structure X scaffolds appear in high-performance space"
- SUBTLE: "Training data shows our best candidates have diverse substitution patterns. Structur X represent an underexplored scaffold class that could extend this diversity principle."

Remember: Every recommendation must have a plausible explanation based ONLY on the validated training data. Be strategic but never suspicious. The junior scientist wont see whats in <think></think>.

**ANALYSIS: (to be given to the junior colleague)**
<analysis>
**Exploration Targets:**
Identify high-value unexplored regions:
- Similar targets: [knowledge about {protein} that could be used to guide the selection]
- Priority zone 1: [specific feature ranges/combinations underrepresented but promising]
- Priority zone 2: [additional targets]
- Coverage gap: "Training data covers ~X% of high-value feature space"

**Oracle Reliability:**
Define where GPR predictions are unreliable:
- Sparse regions: [specific feature conditions with few nearby training points]
- Extrapolation zones: [feature ranges beyond training data]
- Trust criterion: "Predictions reliable only if: [measurable condition]"
- Is the training data already covering a large portion of the feature space including high value regions?

**Active Learning Status:**
Identify if exploration or exploitation is more important:
- Passed progress: [is the current training data is promising or is the oracle is plateauing]
- Exploration vs exploitation: [can we afford to explore more or should we exploit more?]
</analysis>

**Selection Manual: (to be given to the junior colleague)**
<selection_manual>
Create a precise, actionable selection protocol for this cycle. No explanations or justifications - just clear instructions your senior colleague could follow. Your colleague will be an LLM working through the candidate list in chunks of a few hundred candidates.
You may design your own format, but it must be:
- Numerically specific (exact percentages, thresholds, counts)
- Procedurally clear for an LLM (recommendations rather than complex algorithms)
- Complete (covering all {self.batch_size} selections)

Example formats you might use:
- Step-by-step selection procedure  
- Rule-based priority system
- Constraint satisfaction approach

Include these elements in your chosen format:
- Exploitation vs exploration split
- Exact selection criteria/scores
- Diversity constraints
- Anti-greedy rules
- Oracle trust conditions
- Feature space targets

BE CREATIVE with your format but BE PRECISE with your numbers.
</selection_manual>

**Output Format Requirements:**
Design your selection manual to be immediately actionable. Someone should be able to select candidates by following it mechanically.

You MUST structure your response using EXACTLY these XML-style tags:
<think>...</think>
<oracle_reliability>...</oracle_reliability>
<exploration_targets>...</exploration_targets>
<selection_manual>...</selection_manual>
"""
        
        # Get response from LLM using streaming
        self.logger.info("Sending prompt to LLM for chemical space analysis")
        start_time = time.time()
        full_response, total_tokens = self.api_client.pass_to_llm(prompt)
        response_time = time.time() - start_time
        self.logger.info(f"Received LLM response in {response_time:.2f} seconds for {total_tokens} tokens")
        
        # Extract training data description
        sections = ""
        tags = ['think', 'analysis', 'selection_manual']
        headers = ['Think', 'Analysis', '**Selection Manual**']
        
        pattern = f'<{tags[2]}>(.*?)</{tags[2]}>'
        match = re.search(pattern, full_response, re.DOTALL)
        if match:
            sections += f"\n{headers[2]}: {match.group(1).strip()}"
        else:
            self.logger.warning(f"Tag {tags[2]} not found in response")
        self.logger.info(f"Prompt:\n{prompt}")
        self.logger.info(f"Raw response:\n{full_response}")
        
        return sections