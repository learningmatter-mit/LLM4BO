"""
Simplified ToolFactory for molecular analysis tools.

This module provides tools for molecular analysis in active learning contexts,
specifically for LLMChainSelector to analyze chemical space and select candidates.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)


def create_tools(train_df: pd.DataFrame, pred_df: pd.DataFrame, random_seed: int = 42) -> List[Callable]:
    """Create molecular analysis tools for active learning.
    
    This function creates tools as standalone functions to avoid serialization issues with LangChain.
    
    Args:
        train_df: Training dataframe with columns ['SMILES', 'affinity']
        pred_df: Prediction dataframe with columns ['SMILES', 'predicted_RBFE', 'confidence']
        random_seed: Random seed for reproducibility
        
    Returns:
        List of tool functions compatible with LangChain
    """
    # Make copies to avoid modifying original data
    train_data = train_df.copy()
    pred_data = pred_df.copy()
    
    # Compute Morgan fingerprints once
    logger.info("Computing Morgan fingerprints for all compounds...")
    morgan_gen = GetMorganGenerator(radius=4, fpSize=4096, countSimulation=False)
    
    # Store fingerprints as lists (more serializable than RDKit objects)
    fp_training = []
    fp_pred = []
    
    # Compute fingerprints for training data
    for smiles in train_data['SMILES']:
        mol = Chem.MolFromSmiles(smiles)  # type: ignore
        if mol:
            fp = morgan_gen.GetFingerprint(mol)
            fp_training.append(fp)
        else:
            logger.warning(f"Invalid SMILES in training data: {smiles}")
            fp_training.append(None)
    
    # Compute fingerprints for prediction data
    for smiles in pred_data['SMILES']:
        mol = Chem.MolFromSmiles(smiles)  # type: ignore
        if mol:
            fp = morgan_gen.GetFingerprint(mol)
            fp_pred.append(fp)
        else:
            logger.warning(f"Invalid SMILES in prediction data: {smiles}")
            fp_pred.append(None)
        
    logger.info(f"Computed fingerprints for {len(fp_training)} training and {len(fp_pred)} prediction compounds")
    
    def _compact_df(df: pd.DataFrame, index: bool = True) -> str:
        """Convert DataFrame to compact string representation."""
        if len(df) == 0:
            return "No data"
        
        # Format the DataFrame
        formatted = df.to_string(index=index, max_rows=20, max_cols=None)
        
        if len(df) > 20:
            formatted += f"\n... and {len(df) - 20} more rows"
        
        return formatted
    
    def _tanimoto_pairwise_matrix(fps1, fps2) -> np.ndarray:
        """Compute Tanimoto similarity matrix between two sets of fingerprints."""
        matrix = np.zeros((len(fps1), len(fps2)))
        for i, fp1 in enumerate(fps1):
            for j, fp2 in enumerate(fps2):
                if fp1 is not None and fp2 is not None:
                    matrix[i, j] = DataStructs.TanimotoSimilarity(fp1, fp2)  # type: ignore
                else:
                    matrix[i, j] = 0.0
        return matrix
    
    # Define tools as standalone functions with closures
    def search_mols_by_substructure(SMARTS_as_sql: str, n_results: int = 10) -> str:
        """Search molecules using pre-tokenized SQL-style query with AND, OR, NOT operators.
        
        Examples:
            search_mols_by_substructure("c1ccncc1 AND C(=O)O")
            search_mols_by_substructure("2*F")  # Two fluorides
            search_mols_by_substructure("3*c1ccccc1 AND C(=O)O")  # Three benzene rings and carboxylic acid
        
        Args:
            SMARTS_as_sql: SMARTS patterns and logical operators with spaces between all operators
            n_results: Maximum number of results to return (capped at 15)
        
        Returns:
            String with search results (JSON serializable)
        """
        try:
            n_results = min(n_results, 15)
            pos = 0
            smarts_tokens = SMARTS_as_sql.split()
            
            def count_substructure_matches(smiles: str, smarts_pattern: str) -> int:
                """Count the number of times a substructure appears in a molecule."""
                try:
                    mol = Chem.MolFromSmiles(smiles)  # type: ignore
                    pattern = Chem.MolFromSmarts(smarts_pattern)  # type: ignore
                    if not mol or not pattern:
                        return 0
                    matches = mol.GetSubstructMatches(pattern)
                    return len(matches)
                except:
                    return 0
            
            def parse_term():
                nonlocal pos
                if pos >= len(smarts_tokens):
                    return lambda x: True
                
                if smarts_tokens[pos] == 'NOT':
                    pos += 1
                    term_func = parse_term()
                    return lambda x: not term_func(x)
                elif smarts_tokens[pos] == '(':
                    pos += 1
                    expr_func = parse_or()
                    pos += 1  # skip ')'
                    return expr_func
                else:
                    token = smarts_tokens[pos]
                    pos += 1
                    
                    # Check if token has numeric prefix (e.g., "2*F")
                    if '*' in token:
                        try:
                            count_str, smarts = token.split('*', 1)
                            required_count = int(count_str)
                            return lambda x, s=smarts, c=required_count: count_substructure_matches(x, s) >= c
                        except (ValueError, IndexError):
                            mol = Chem.MolFromSmarts(token)  # type: ignore
                            return lambda x: mol and Chem.MolFromSmiles(x) and Chem.MolFromSmiles(x).HasSubstructMatch(mol)  # type: ignore
                    else:
                        mol = Chem.MolFromSmarts(token)  # type: ignore
                        return lambda x: mol and Chem.MolFromSmiles(x) and Chem.MolFromSmiles(x).HasSubstructMatch(mol)  # type: ignore
            
            def parse_and():
                nonlocal pos
                left = parse_term()
                
                while pos < len(smarts_tokens) and smarts_tokens[pos] == 'AND':
                    pos += 1
                    right = parse_term()
                    left = lambda x, l=left, r=right: l(x) and r(x)
                
                return left
            
            def parse_or():
                nonlocal pos
                left = parse_and()
                while pos < len(smarts_tokens) and smarts_tokens[pos] == 'OR':
                    pos += 1
                    right = parse_and()
                    left = lambda x, l=left, r=right: l(x) or r(x)
                return left
            
            filter_func = parse_or()
            
            matches = pred_data[pred_data['SMILES'].apply(filter_func)][['SMILES', 'predicted_RBFE', 'confidence']].sort_values(by='predicted_RBFE', ascending=False)
            result_text = f'{len(matches)} molecules found:\n' + _compact_df(matches.iloc[:min(n_results, len(matches)), :])
            
            return result_text
            
        except Exception as e:
            logger.error(f"Error in search_mols_by_substructure: {e}")
            return f"Error in substructure search: {str(e)}"
    
    def query_predictions(n_select: int = 10, max_pairwise_similarity: float = 1, 
                         min_similarity_to_training: float = 0, max_similarity_to_training: float = 1, 
                         min_training_affinity: float = 0, min_pred_affinity: float = 0, 
                         min_pred_uncertainty: float = 0, max_pred_uncertainty: float = float('inf')) -> str:
        """Query the predictions dataframe for molecules that match the given criteria.
        
        Example:
            To get 10 molecules Tanimoto similarity ≥ 0.5 to at least one training sample with RBFE ≥ 10.0 
            and Maximum pairwise Tanimoto similarity ≤ 0.8 among selected:
            query_predictions(n_select=10, max_pairwise_similarity=0.8, min_similarity_to_training=0.5, min_training_affinity=10)
        
        Args:
            n_select: Number of molecules to select
            max_pairwise_similarity: Maximum pairwise Tanimoto similarity among selected molecules
            min_similarity_to_training: Minimum Tanimoto similarity to at least one training sample
            max_similarity_to_training: Maximum Tanimoto similarity to at least one training sample
            min_training_affinity: Minimum affinity of training samples used to calculate Tanimoto similarity
            min_pred_affinity: Minimum predicted affinity
            min_pred_uncertainty: Minimum predicted uncertainty
            max_pred_uncertainty: Maximum predicted uncertainty
            
        Returns:
            String with query results (JSON serializable)
        """
        try:
            # Filter by predicted affinity and uncertainty
            pred_reduced = pred_data[
                (pred_data['predicted_RBFE'] >= min_pred_affinity) &
                (pred_data['confidence'] >= min_pred_uncertainty) &
                (pred_data['confidence'] <= max_pred_uncertainty)
            ].copy()
            
            if len(pred_reduced) == 0:
                return "No molecules found matching the affinity and uncertainty criteria"
            
            # Calculate similarity to training data if needed
            if min_similarity_to_training > 0 or max_similarity_to_training < 1:
                # Filter training data by affinity
                high_affinity_train = train_data[train_data['affinity'] >= min_training_affinity]
                
                if len(high_affinity_train) == 0:
                    return "No training samples found matching the affinity criteria"
                
                # Calculate similarity matrix
                high_affinity_fps = [fp_training[i] for i in high_affinity_train.index if fp_training[i] is not None]
                pred_fps = [fp_pred[i] for i in range(len(pred_reduced)) if fp_pred[i] is not None]
                
                if len(high_affinity_fps) == 0 or len(pred_fps) == 0:
                    return "No valid fingerprints found for similarity calculation"
                
                similarity_matrix = _tanimoto_pairwise_matrix(pred_fps, high_affinity_fps)
                
                # Find molecules that meet similarity criteria
                max_similarities = np.max(similarity_matrix, axis=1)
                valid_indices = np.where(
                    (max_similarities >= min_similarity_to_training) & 
                    (max_similarities <= max_similarity_to_training)
                )[0]
                
                if len(valid_indices) == 0:
                    return "No molecules found matching the similarity criteria"
                
                pred_reduced = pred_reduced.iloc[valid_indices].copy()
            
            # Sort by predicted affinity (descending)
            pred_reduced_sorted = pred_reduced.sort_values(by='predicted_RBFE', ascending=False)
            
            # Select molecules with pairwise similarity constraint
            selected = []
            max_pairwise_selected = {'max_pairwise_similarity': []}
            
            # Create mapping from DataFrame index to position in fp_pred
            pred_indices = list(pred_reduced_sorted.index)
            
            for pos, (idx, row) in enumerate(pred_reduced_sorted.iterrows()):
                if len(selected) >= n_select:
                    break
                
                # Check pairwise similarity to all already selected molecules
                max_sim = 0.0
                if selected and fp_pred[pos] is not None: 
                    selected_fps = [fp_pred[j] for j in selected if fp_pred[j] is not None]
                    if selected_fps:
                        sim = _tanimoto_pairwise_matrix([fp_pred[pos]], selected_fps)
                        max_sim_val = float(np.max(sim))
                        if max_sim_val >= max_pairwise_similarity:
                            continue
                        max_sim = max_sim_val
                    
                selected.append(pos)
                max_pairwise_selected['max_pairwise_similarity'].append(max_sim)
            
            if selected:
                # Map back to original DataFrame indices
                selected_indices = [pred_indices[pos] for pos in selected]
                pred_reduced_selected = pred_reduced.loc[selected_indices]
                pred_reduced_selected = pd.concat([pred_reduced_selected, pd.DataFrame(max_pairwise_selected, index=pred_reduced_selected.index)], axis=1)
                return _compact_df(pred_reduced_selected)
            else:
                return "No molecules found matching the criteria"
                
        except Exception as e:
            logger.error(f"Error in query_predictions: {e}")
            return f"Error in query: {str(e)}"
    
    def get_max_pairwise_similarity(indices: List[int]) -> str:
        """Get the maximum pairwise Tanimoto similarity among the provided indices.
        
        Args:
            indices: List of indices to check pairwise similarity
            
        Returns:
            String with maximum pairwise similarity value (JSON serializable)
        """
        try:
            if len(indices) < 2:
                return "At least 2 indices required for pairwise similarity calculation"
            
            # Get valid fingerprints for the indices
            valid_fps = [fp_pred[i] for i in indices if i < len(fp_pred) and fp_pred[i] is not None]
            
            if len(valid_fps) < 2:
                return "At least 2 valid fingerprints required for pairwise similarity calculation"
            
            # Compute pairwise similarity matrix
            similarity_matrix = _tanimoto_pairwise_matrix(valid_fps, valid_fps)
            
            # Get maximum similarity (excluding diagonal)
            max_similarity = float(np.max(similarity_matrix - np.eye(len(valid_fps))))
            
            return f"Maximum pairwise Tanimoto similarity among {len(indices)} compounds: {max_similarity:.4f}"
            
        except Exception as e:
            logger.error(f"Error in get_max_pairwise_similarity: {e}")
            return f"Error calculating pairwise similarity: {str(e)}"
    
    return [search_mols_by_substructure, query_predictions, get_max_pairwise_similarity] 