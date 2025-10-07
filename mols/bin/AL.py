#!/usr/bin/env python3
"""Active Learning implementation module."""

import json
import logging
import importlib
import os
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
import torch
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.Model import Model
from utils.Selector import Selector
from train import train_model
from sample_selector import select_samples, update_pools
from eval_AL import evaluate_performance
from utils.PlotUtils import plot_training_results
from utils.SMILESUtils import smiles_to_fingerprint
from utils.DataUtils import (
    get_output_path, ExperimentConfig, CheckpointManager, get_all_training_repeats
)

class ActiveLearningExperiment:
    """Active Learning experiment manager class."""
    
    def __init__(
        self,
        config: ExperimentConfig = None,
        output_dir: Optional[Path] = None,
        restore_from: Optional[Path] = None,
        restore_cycle: int = -1,
    ):
        """Initialize Active Learning experiment.
        
        Args:
            config: Experiment configuration
            output_dir: Results directory path
            restore_from: Path to checkpoint to restore from
            restore_cycle: Cycle to restore from (-1 for latest)
            bad_start: Whether to use bad start initialization
        """
        self.config = config
        if self.config is None and restore_from is None:
            raise ValueError("Either config or restore_from must be provided")
        elif self.config is None and restore_from is not None:
            self.config = CheckpointManager(restore_from).get_config()
        elif self.config is not None and restore_from is not None:
            raise ValueError("Strictly one of config or restore_from must be provided")
        
        self.device = torch.device("cpu")
        
        # Setup output directory
        if output_dir is None:
            target = Path(self.config.data_path).stem
            self.output_dir = get_output_path(
                self.config.model_name,
                self.config.selector_name,
                self.config.initial_size,
                self.config.batch_size,
                target, 
                self.config.bad_start,
                self.config.selector_kwargs
            )
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        self.total_budget = 600
        
        # Initialize or restore experiment
        if restore_from:
            self._restore_from_checkpoint(restore_from, restore_cycle)
        else:
            self._initialize_experiment()
            
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        log_file = self.output_dir / f"al_loop_training_{datetime.now().strftime('%Y%m%d_%H')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
        
    def _initialize_experiment(self) -> None:
        """Initialize new experiment components."""
        # Load data
        self.X_labeled, self.y_labeled, self.X_unlabeled, self.y_unlabeled = self._load_data()
        
        # Initialize model and selector
        self.model = self._initialize_model()
        self.selector = self._initialize_selector()
        
        # Initialize results file
        self.results_file = self.output_dir / "Training_results.json"
        with open(self.results_file, 'w') as f:
            json.dump({
                "experiment_config": self.config.to_dict(),
                "al_cycles": []
            }, f, indent=4)
            
        self.cycle = 1
        
    def _restore_from_checkpoint(self, checkpoint_path: Path, cycle: int) -> None:
        """Restore experiment from checkpoint."""
        checkpoint = CheckpointManager(checkpoint_path)
        
        self.logger.info(f"Restoring checkpoint from {checkpoint_path}")
        
        # Restore data
        labeled_data, unlabeled_data = checkpoint.get_data(cycle)
        
        # Convert to feature dictionaries
        self.X_labeled = self._dataframe_to_features(labeled_data)
        self.y_labeled = labeled_data['affinity'].values
        self.X_unlabeled = self._dataframe_to_features(unlabeled_data)
        self.y_unlabeled = unlabeled_data['affinity'].values
        
        # Restore experiment components
        self.model = self._initialize_model()
        self.selector = self._initialize_selector()
        
        self.results_file = checkpoint_path
        self.cycle = checkpoint.get_cycle(cycle)
        self.logger.info(f"Starting cycle {self.cycle}")
        
    def _dataframe_to_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert DataFrame to feature dictionary."""
        features = {}
        for column in df.columns:
            if column == 'affinity':
                continue  # Skip target column
            if column == 'SMILES':
                features['fingerprint'] = np.array([smiles_to_fingerprint(smiles, radius=4, nBits=4096) for smiles in df[column].values])
            features[column] = df[column].values
        return features
        
    def _load_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """Load and split data into initial training and unlabeled pools."""
        df = pd.read_csv(self.config.data_path).iloc[:,:2]
        self.logger.info(f"Loaded features {df.columns}")
        # Validate required columns
        required_columns = ['SMILES', 'affinity']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")
        
        if self.config.bad_start:
            if self.config.selector_name == "RandomSelector":
                return self._generate_bad_start(df)
            return self._load_data_from_random_selector(df)
        else:
            return self._load_data_random(df)
    
    def _load_data_random(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """Load data using random sampling."""
        # Shuffle data
        df = df.sample(frac=1, random_state=self.config.random_seed).reset_index(drop=True)
        
        if self.config.initial_size >= len(df):
            raise ValueError(
                f"Initial size ({self.config.initial_size}) must be less than total samples ({len(df)})"
            )
        
        # Split data
        train_df = df.iloc[:self.config.initial_size]
        test_df = df.iloc[self.config.initial_size:]
        
        # Add cycle_added feature (0 for initial data)
        train_df = train_df.copy()
        train_df['cycle_added'] = 0
        
        # Convert to feature dictionaries
        X_labeled = self._dataframe_to_features(train_df)
        y_labeled = train_df['affinity'].values
        X_unlabeled = self._dataframe_to_features(test_df)
        y_unlabeled = test_df['affinity'].values
        
        return X_labeled, y_labeled, X_unlabeled, y_unlabeled
    
    def _load_data_from_random_selector(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """Load initial samples from RandomSelector's Training_results.json file for consistency."""
        target = Path(self.config.data_path).stem
        
        # Find RandomSelector results with matching random seed
        random_selector_repeats = get_all_training_repeats(
            oracle_name=self.config.model_name,
            selector_name="RandomSelector",
            initial_size=self.config.initial_size,
            batch_size=self.config.batch_size,
            target=target,
            bad_start=self.config.bad_start,
            selector_kwargs=None  # RandomSelector doesn't need specific kwargs
        )
        
        # Find the repeat with matching random seed
        matching_results_file = None
        for repeat_dir in random_selector_repeats:
            results_file = repeat_dir / "Training_results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    if results['experiment_config']['random_seed'] == self.config.random_seed:
                        matching_results_file = results_file
                        break
                except (json.JSONDecodeError, KeyError):
                    continue
        
        if matching_results_file is None:
            self.logger.warning(f"No RandomSelector results found for random_seed={self.config.random_seed}. Falling back to random initialization.")
            return self._load_data_random(df)
        
        self.logger.info(f"Loading initial samples from RandomSelector results: {matching_results_file}")
        
        # Load the initial samples from RandomSelector's cycle 0
        with open(matching_results_file, 'r') as f:
            results = json.load(f)
        
        # Get the initial samples (cycle 0)
        if len(results['al_cycles']) == 0:
            self.logger.warning("No cycles found in RandomSelector results. Falling back to random initialization.")
            return self._load_data_random(df)
        
        initial_samples = results['al_cycles'][0]['samples_selected']
        self.logger.info(f"Loaded {len(initial_samples)} initial samples from RandomSelector")
        
        # Split data based on the initial samples
        labeled_mask = df['SMILES'].isin(initial_samples)
        labeled_df = df[labeled_mask].copy()
        unlabeled_df = df[~labeled_mask].copy()
        
        # Add cycle_added feature (0 for initial data)
        labeled_df['cycle_added'] = 0
        
        # Convert to feature dictionaries
        X_labeled = self._dataframe_to_features(labeled_df)
        y_labeled = labeled_df['affinity'].values
        X_unlabeled = self._dataframe_to_features(unlabeled_df)
        y_unlabeled = unlabeled_df['affinity'].values
        
        self.logger.info(f"Bad start initialization from RandomSelector: {len(X_labeled['SMILES'])} labeled samples, {len(X_unlabeled['SMILES'])} unlabeled samples")
        
        return X_labeled, y_labeled, X_unlabeled, y_unlabeled

    def _generate_bad_start(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data using bad start initialization with UMAP clustering."""
        from umap import UMAP
        from sklearn.cluster import KMeans
        from utils.SMILESUtils import smiles_to_fingerprint
        
        self.logger.info("Using bad start initialization with UMAP clustering")
        
        # Convert SMILES to fingerprints
        smiles_list = df['SMILES'].values
        fingerprints = np.array([smiles_to_fingerprint(smiles, radius=4, nBits=4096) for smiles in smiles_list])
        
        # Apply UMAP dimensionality reduction
        umap_reducer = UMAP(
            n_components=10,
            random_state=self.config.random_seed,
            transform_seed=self.config.random_seed,
            n_jobs=1
        )
        embeddings = umap_reducer.fit_transform(fingerprints)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=10, random_state=self.config.random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Add cluster labels to dataframe
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # Get cluster sizes
        cluster_sizes = df_with_clusters['cluster'].value_counts().sort_index()
        self.logger.info(f"Cluster sizes: {cluster_sizes.to_dict()}")
        
        # Randomly select a cluster using the random seed
        np.random.seed(self.config.random_seed)
        available_clusters = list(range(10))
        selected_cluster = np.random.choice(available_clusters)
        
        # Get samples from selected cluster
        cluster_samples = df_with_clusters[df_with_clusters['cluster'] == selected_cluster]
        
        # If selected cluster doesn't have enough samples, select another cluster
        while len(cluster_samples) < self.config.initial_size and available_clusters:
            available_clusters.remove(selected_cluster)
            if available_clusters:
                selected_cluster = np.random.choice(available_clusters)
                cluster_samples = df_with_clusters[df_with_clusters['cluster'] == selected_cluster]
            else:
                # If no cluster has enough samples, use the largest cluster
                largest_cluster = cluster_sizes.idxmax()
                cluster_samples = df_with_clusters[df_with_clusters['cluster'] == largest_cluster]
                self.logger.warning(f"No cluster has {self.config.initial_size} samples. Using largest cluster ({len(cluster_samples)} samples)")
                break
        
        self.logger.info(f"Selected cluster {selected_cluster} with {len(cluster_samples)} samples")
        
        # Sample initial_size samples from the selected cluster
        if len(cluster_samples) > self.config.initial_size:
            cluster_samples = cluster_samples.sample(n=self.config.initial_size, random_state=self.config.random_seed)
        
        # Get remaining samples as unlabeled
        remaining_samples = df_with_clusters[~df_with_clusters.index.isin(cluster_samples.index)]
        cluster_samples['cycle_added'] = 0
        # Extract SMILES and affinity values
        X_labeled = self._dataframe_to_features(cluster_samples)
        y_labeled = cluster_samples['affinity'].values
        X_unlabeled = self._dataframe_to_features(remaining_samples)
        y_unlabeled = remaining_samples['affinity'].values

        
        self.logger.info(f"Bad start initialization: {len(X_labeled)} labeled samples from cluster {selected_cluster}, {len(X_unlabeled)} unlabeled samples")
        
        return X_labeled, y_labeled, X_unlabeled, y_unlabeled

    def _initialize_model(self) -> Model:
        """Initialize model instance."""
        try:
            model_module = importlib.import_module(f"aloracles.{self.config.model_name}")
            model_class = getattr(model_module, self.config.model_name)
            return model_class(self.config.random_seed, **self.config.model_kwargs)
        except (ModuleNotFoundError, AttributeError) as e:
            self.logger.error(f"Failed to load model {self.config.model_name}: {e}")
            available = [f[:-3] for f in os.listdir("aloracles") if f.endswith('.py') and f != '__init__.py']
            self.logger.error(f"Available models: {available}")
            raise
            
    def _initialize_selector(self) -> Selector:
        """Initialize selector instance."""
        try:
            if "LLMFeaturise" in self.config.selector_name:
                selector_module = importlib.import_module("alselectors.LLMFeaturiseSelector")
            elif "LLMFat" in self.config.selector_name:
                selector_module = importlib.import_module("alselectors.LLMFatSelector")
            elif "LLMWorkflow" in self.config.selector_name:
                selector_module = importlib.import_module("alselectors.LLMWorkflowSelector")
            else:
                selector_module = importlib.import_module(f"alselectors.{self.config.selector_name}")
            selector_class = getattr(selector_module, self.config.selector_name)
            return selector_class(
                batch_size=self.config.batch_size,
                random_seed=self.config.random_seed,
                **self.config.selector_kwargs
            )
        except (ModuleNotFoundError, AttributeError) as e:
            self.logger.error(f"Failed to load selector {self.config.selector_name}: {e}")
            available = [f[:-3] for f in os.listdir("alselectors") if f.endswith('.py') and f != '__init__.py']
            self.logger.error(f"Available selectors: {available}")
            raise

    def _store_cycle_results(self, cycle_results: Dict[str, Any]) -> None:
        # Drop "fingerprint" from cycle_results
        cycle_results.pop('fingerprint', None)
        """Store cycle results in results file."""
        with open(self.results_file, 'r') as f:
            all_results = json.load(f)
        if cycle_results['cycle'] in [cycle['cycle'] for cycle in all_results['al_cycles']]:
            self.logger.warning(f"Cycle {cycle_results['cycle']} already exists in results file. May be ok if restarting from cycle {cycle_results['cycle']+1}")
            return all_results
        
        # Check for LLM selector summary file
        if self.config.selector_name in ["LLMWorkflowSelector", "LLMWorkflowSimpleSelector", "LLMChainSelector", "LLMAgentSelector"]:
            summary_file = self.output_dir / f"cycle_{cycle_results['cycle']}_summary.txt"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = f.read().strip()
                cycle_results['summary'] = summary
                # Clean up the temporary file
                summary_file.unlink()
                self.logger.info(f"Added LLM summary to cycle {cycle_results['cycle']}")
        
        all_results["al_cycles"].append(cycle_results)
        with open(self.results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        self.logger.info(f"Cycle {cycle_results['cycle']} added to results file")   
        return all_results
    
    def run(self) -> Tuple[Path, Path]:
        """Run active learning experiment.
        
        Returns:
            Tuple containing:
                - Path to results file
                - Path to output directory
        """
        self.logger.info("Starting Active Learning experiment")
        self.logger.info(f"Model: {self.config.model_name}")
        self.logger.info(f"Selector: {self.config.selector_name}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Data path: {self.config.data_path}")
        self.logger.info(f"Bad start: {self.config.bad_start}")
        self.logger.info(f"Loaded data: {len(self.X_labeled['SMILES'])} training, {len(self.X_unlabeled['SMILES'])} testing")

        total_labeled = len(self.X_labeled['SMILES'])

        def train_and_evaluate_model(cycle_start_time):
            self.model, predictions, confidence_scores, train_predictions = train_model(
                model=self.model,
                data=(self.X_labeled, self.y_labeled, self.X_unlabeled, self.y_unlabeled),
                output_dir=self.output_dir,
                cycle=self.cycle,
                random_seed=self.config.random_seed,
            )
            
            # Evaluate model
            metrics = evaluate_performance(
                predictions=(predictions, confidence_scores, train_predictions),
                data=(self.X_labeled, self.y_labeled, self.X_unlabeled, self.y_unlabeled),
                batch_size=self.config.batch_size
            )
            
            # Log metrics
            self.logger.info("\nPerformance Metrics:")
            for metric_name, value in metrics.items():
                self.logger.info(f"{metric_name}: {value:.4f}")
            
            # Store cycle results with feature arrays
            cycle_data = {
                "cycle": self.cycle-1,
                "samples_selected": [str(smile) for smile in self.X_labeled['SMILES']],
                "total_labeled": total_labeled,
                "performance_metrics": metrics,
                "computational_time": time.time() - cycle_start_time
                }
            
            # Add feature arrays for each feature
            for feature_name, feature_values in self.X_labeled.items():
                if feature_name == 'SMILES':
                    continue  # Skip SMILES as it's already in samples_selected
                if feature_name == 'fingerprint':
                    continue  # Skip fingerprint as it contains numpy arrays that can't be serialized
                cycle_data[feature_name] = feature_values.tolist()
            
            self._store_cycle_results(cycle_results=cycle_data)
            return predictions, confidence_scores

        cycle_start_time = time.time()
        with tqdm(total=self.total_budget, initial=total_labeled, desc="Labeled samples") as pbar:
            while total_labeled < self.total_budget:
                self.logger.info(f"\nStarting AL cycle {self.cycle}")
                
                predictions, confidence_scores = train_and_evaluate_model(cycle_start_time)
                cycle_start_time = time.time()
                # Select new samples
                self.config.selector_kwargs['cycle'] = self.cycle
                self.config.selector_kwargs['total_cycles'] = np.ceil((self.total_budget - self.config.initial_size) / self.config.batch_size)
                self.config.selector_kwargs['oracle_name'] = self.config.model_name
                self.config.selector_kwargs['protein'] = Path(self.config.data_path).stem
                self.config.selector_kwargs['training_results_path'] = str(self.results_file)
                self.config.selector_kwargs['y_unlabeled'] = self.y_unlabeled
                
                selected_indices = select_samples(
                    selector=self.selector,
                    unlabeled_data=self.X_unlabeled,
                    predictions=predictions,
                    confidence_scores=confidence_scores,
                    training_data=(self.X_labeled, self.y_labeled),
                    output_dir=self.output_dir,
                    random_seed=self.config.random_seed,
                    **self.config.selector_kwargs
                )
                # Update pools
                self.X_unlabeled, self.y_unlabeled, X_new, y_new = update_pools(
                    unlabeled_data=self.X_unlabeled,
                    selected_indices=selected_indices,
                    labels=self.y_unlabeled
                )

                # Add cycle_added feature to new samples
                X_new['cycle_added'] = np.full(len(selected_indices), self.cycle)
                
                # Add selected data to training features (predictions, confidence_scores, selected_indices)
                X_new['selected_predictions'] = predictions[selected_indices]
                X_new['selected_confidence_scores'] = confidence_scores[selected_indices]
                current_len = len(self.X_labeled['SMILES'])
                # Concatenate all features
                for feature_name in X_new:
                    if feature_name in self.X_labeled:
                        self.X_labeled[feature_name] = np.concatenate([self.X_labeled[feature_name], X_new[feature_name]])
                    else:
                        # For new features (like selected_predictions, selected_confidence_scores), 
                        # we need to pad with NaN for existing samples that don't have these values
                        padded_new_feature = np.concatenate([np.full(current_len, np.nan), X_new[feature_name]])
                        self.X_labeled[feature_name] = padded_new_feature
                
                self.y_labeled = np.concatenate([self.y_labeled, y_new])


                # Update progress
                total_labeled = len(self.X_labeled['SMILES'])
                pbar.update(len(selected_indices))
                self.cycle += 1
        train_and_evaluate_model(cycle_start_time)

        self.logger.info("\nActive Learning completed successfully!")
        return self.results_file, self.output_dir

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Active Learning Framework")
    
    # Experiment configuration
    experiment_group = parser.add_argument_group("Experiment Configuration")
    experiment_group.add_argument("--model", type=str, help="Oracle model class name")
    experiment_group.add_argument("--selector", type=str, help="Selector class name")
    experiment_group.add_argument("--data", type=str, help="Path to dataset CSV file")
    experiment_group.add_argument("--batch_size", type=int, default=60,
                                help="Samples per AL cycle (default: 60)")
    experiment_group.add_argument("--initial_size", type=int, default=60,
                                help="Initial training set size (default: 60)")
    experiment_group.add_argument("--random_seed", type=int, default=42,
                                help="Random seed for reproducibility (default: 42)")
    experiment_group.add_argument("--bad_start", action="store_true",
                                help="Use bad start initialization with UMAP clustering")
    
    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("--output_dir", type=str, help="Results directory path")
    
    # Checkpoint configuration
    checkpoint_group = parser.add_argument_group("Checkpoint Configuration")
    checkpoint_group.add_argument("--restore_checkpoint", type=str,
                                help="Path to training results JSON file")
    checkpoint_group.add_argument("--restore_cycle", type=int, default=-1,
                                help="Cycle to restore from (default: -1 for latest)")
    
    # Additional configuration
    config_group = parser.add_argument_group("Additional Configuration")
    config_group.add_argument("--model_kwargs", type=str, default="{}",
                            help="JSON string of model-specific kwargs")
    config_group.add_argument("--selector_kwargs", type=str, default="{}",
                            help="JSON string of selector-specific kwargs")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.restore_checkpoint is None:
        if not all([args.model, args.selector, args.data]):
            parser.error("--model, --selector, and --data are required when not restoring from checkpoint")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    if args.restore_checkpoint:
        experiment = ActiveLearningExperiment(
            output_dir=args.output_dir,
            restore_from=Path(args.restore_checkpoint),
            restore_cycle=args.restore_cycle
        )
    else:
        config = ExperimentConfig(
            model_name=args.model,
            selector_name=args.selector,
            data_path=args.data,
            batch_size=args.batch_size,
            initial_size=args.initial_size,
            random_seed=args.random_seed,
            bad_start=args.bad_start,
            model_kwargs=json.loads(args.model_kwargs),
            selector_kwargs=json.loads(args.selector_kwargs)
        )
        experiment = ActiveLearningExperiment(
            config=config,
            output_dir=args.output_dir,
        )
    
    results_file, output_dir = experiment.run()
    plot_training_results(results_file, output_dir)