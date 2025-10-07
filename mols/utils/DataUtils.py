from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import re
from typing import Dict, Any, Optional, Tuple, List

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent

class ExperimentConfig:
    """Configuration manager for Active Learning experiments."""
    
    def __init__(
        self,
        model_name: str,
        selector_name: str,
        data_path: str,
        batch_size: int,
        initial_size: int,
        random_seed: int,
        bad_start: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        selector_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Initialize experiment configuration.
        
        Args:
            model_name: Name of the model class
            selector_name: Name of the selector class
            data_path: Path to dataset CSV file
            batch_size: Number of samples per AL cycle
            initial_size: Size of initial training set
            random_seed: Random seed for reproducibility
            model_kwargs: Model-specific parameters
            selector_kwargs: Selector-specific parameters
        """
        self.model_name = model_name
        self.selector_name = selector_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.initial_size = initial_size
        self.random_seed = random_seed
        self.bad_start = bad_start
        self.model_kwargs = model_kwargs or {}
        self.model_kwargs['selector_name'] = selector_name
        self.selector_kwargs = selector_kwargs or {}
        self.selector_kwargs['model_name'] = model_name
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "model": self.model_name,
            "selector": self.selector_name,
            "data_path": self.data_path,
            "batch_size": self.batch_size,
            "total_budget": 600,
            "initial_size": self.initial_size,
            "random_seed": self.random_seed,
            "bad_start": self.bad_start,
            "data_config": {
                "target": "affinity"
            },
            "hyperparameters": {
                "model": self.model_kwargs,
                "selector": self.selector_kwargs
            }
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        return cls(
            model_name=config_dict["model"],
            selector_name=config_dict["selector"],
            data_path=config_dict["data_path"],
            batch_size=config_dict["batch_size"],
            initial_size=config_dict["initial_size"],
            random_seed=config_dict["random_seed"],
            model_kwargs=config_dict["hyperparameters"]["model"],
            selector_kwargs=config_dict["hyperparameters"]["selector"]
        )

class CheckpointManager:
    """Manager for experiment checkpoints."""
    
    def __init__(self, results_path: Path):
        """Initialize checkpoint manager.
        
        Args:
            results_path: Path to results JSON file
        """
        self.results_path = results_path
        with open(results_path, 'r') as f:
            self.results = json.load(f)
            
    def get_config(self) -> ExperimentConfig:
        """Get experiment configuration from checkpoint."""
        return ExperimentConfig.from_dict(self.results["experiment_config"])
        
    def get_data(self, cycle: int = -1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get labeled and unlabeled data up to specified cycle.
        
        Args:
            cycle: Cycle to restore from (-1 for latest)
            
        Returns:
            Tuple of (labeled_data, unlabeled_data) DataFrames
        """
        cycle = self.get_cycle(cycle)
            
        config = self.get_config()
        all_data = pd.read_csv(config.data_path)
        
        # Check if we have feature arrays in the results
        if cycle > 0 and any(key not in ['cycle', 'samples_selected', 'summary', 'total_labeled', 'performance_metrics', 'computational_time'] for key in self.results['al_cycles'][cycle-1].keys()):
            # Use feature arrays from results
            return self._get_data_from_feature_arrays(cycle, all_data)
        else:
            # Fallback to old method using just SMILES
            return self._get_data_from_smiles(cycle, all_data)
    
    def _get_data_from_feature_arrays(self, cycle: int, all_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get data using feature arrays from results."""
        # Get selected samples and feature arrays
        selected_samples = self.results['al_cycles'][cycle-1]['samples_selected']
        
        # Create labeled data with features
        labeled_data = []
        for i, smile in enumerate(selected_samples):
            row = all_data[all_data['SMILES'] == smile].iloc[0].to_dict()
            # Add features from arrays
            for key, value in self.results['al_cycles'][cycle-1].items():
                if key not in ['cycle', 'samples_selected', 'summary', 'total_labeled', 'performance_metrics', 'computational_time']:
                    # This is a feature array
                    if i < len(value):  # Make sure we have enough values
                        row[key] = value[i]
            labeled_data.append(row)
        
        labeled_df = pd.DataFrame(labeled_data)
        
        # Create unlabeled data (everything not in labeled)
        unlabeled_mask = ~all_data['SMILES'].isin(selected_samples)
        unlabeled_df = all_data[unlabeled_mask].copy()
        
        # Add cycle_added feature to unlabeled data (set to -1 to indicate not yet added)
        if 'cycle_added' not in unlabeled_df.columns:
            unlabeled_df['cycle_added'] = -1
        
        return labeled_df, unlabeled_df
    
    def _get_data_from_features(self, cycle: int, all_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get data using feature information from results (old format)."""
        # Get selected samples and their features
        selected_samples = self.results['al_cycles'][cycle-1]['samples_selected']
        sample_features = self.results['al_cycles'][cycle-1]['sample_features']
        
        # Create labeled data with features
        labeled_data = []
        for smile in selected_samples:
            row = all_data[all_data['SMILES'] == smile].iloc[0].to_dict()
            # Add features from results
            if smile in sample_features:
                for feature_name, feature_value in sample_features[smile].items():
                    if feature_name != 'SMILES':  # Skip SMILES as it's already in the row
                        row[feature_name] = feature_value
            labeled_data.append(row)
        
        labeled_df = pd.DataFrame(labeled_data)
        
        # Create unlabeled data (everything not in labeled)
        unlabeled_mask = ~all_data['SMILES'].isin(selected_samples)
        unlabeled_df = all_data[unlabeled_mask].copy()
        
        # Add cycle_added feature to unlabeled data (set to -1 to indicate not yet added)
        if 'cycle_added' not in unlabeled_df.columns:
            unlabeled_df['cycle_added'] = -1
        
        return labeled_df, unlabeled_df
    
    def _get_data_from_smiles(self, cycle: int, all_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get data using only SMILES information (fallback method)."""
        # Get selected samples up to specified cycle
        selected_samples = self.results['al_cycles'][cycle-1]['samples_selected']
        
        # Split data
        labeled_data = all_data[all_data['SMILES'].isin(selected_samples)]
        unlabeled_data = all_data[~all_data['SMILES'].isin(selected_samples)]
        
        # Add cycle_added feature if not present
        if 'cycle_added' not in labeled_data.columns:
            labeled_data = labeled_data.copy()
            labeled_data['cycle_added'] = 0  # Assume all were added in initial cycle
        
        if 'cycle_added' not in unlabeled_data.columns:
            unlabeled_data = unlabeled_data.copy()
            unlabeled_data['cycle_added'] = -1  # Not yet added
        
        return labeled_data, unlabeled_data
        
    def get_cycle(self, cycle: int = -1) -> int:
        """Get cycle number to restore from."""
        if cycle == -1:
            return len(self.results["al_cycles"])
        return cycle

def get_output_path(
    oracle_name: str, 
    selector_name: str, 
    initial_size: int, 
    batch_size: int, 
    target: str, 
    bad_start: bool = False,
    selector_kwargs: Optional[Dict[str, Any]] = None
) -> Path:
    """Get the output path for a given experiment configuration.
    
    Args:
        oracle_name: Name of the oracle
        selector_name: Name of the selector
        initial_size: Initial size of the dataset
        batch_size: Batch size for the AL campaign
        target: Name of the target
        bad_start: Whether to use bad start initialization
        selector_kwargs: Selector-specific parameters for unique naming
        
    Returns:
        Path to output directory
    """
    subdir = target + "_bad" if bad_start else target
    
    # Build experiment name with selector-specific parameters
    experiment_parts = [oracle_name, selector_name, str(initial_size), str(batch_size)]
    
    # Add selector-specific parameters to experiment name for uniqueness
    if selector_kwargs:
        # For UCBSelector, include beta in the experiment name
        if selector_name == "UCBSelector" and "beta" in selector_kwargs:
            beta_str = f"beta{selector_kwargs['beta']:.1f}".replace('.', 'p')
            experiment_parts.append(beta_str)
        
        # For LLMWorkflowSelector ablation study, include disabled tools
        if "LLMWorkflow" in selector_name:
            ablation_parts = []
            if selector_kwargs.get('disable_ucb', False):
                ablation_parts.append("noUCB")
            if selector_kwargs.get('disable_smarts', False):
                ablation_parts.append("noSMARTS")
            if selector_kwargs.get('disable_tanimoto', False):
                ablation_parts.append("noTanimoto")
            
            if ablation_parts:
                ablation_str = "_".join(ablation_parts)
                experiment_parts.append(f"ablation_{ablation_str}")
        
        # Add other selector-specific parameters as needed
        # Example for other selectors:
        # elif selector_name == "SomeOtherSelector" and "param" in selector_kwargs:
        #     param_str = f"param{selector_kwargs['param']}"
        #     experiment_parts.append(param_str)
    
    experiment = "_".join(experiment_parts)
    directory = project_root / "output" / "training" / subdir
    directory.mkdir(parents=True, exist_ok=True)

    index = 0
    while (directory / f"{experiment}_{index}_{datetime.now().strftime('%Y%m%d')}").exists():
        index += 1
    return directory / f"{experiment}_{index}_{datetime.now().strftime('%Y%m%d')}"

def get_all_training_repeats(
    oracle_name: str, 
    selector_name: str, 
    initial_size: int, 
    batch_size: int, 
    target: str, 
    bad_start: bool,
    selector_kwargs: Optional[Dict[str, Any]] = None
) -> List[Path]:
    """Get all training repeats for a given experiment configuration.
    
    Args:
        oracle_name: Name of the oracle
        selector_name: Name of the selector
        initial_size: Initial size of the dataset
        batch_size: Batch size for the AL campaign
        target: Name of the target
        bad_start: Whether to use bad start initialization
        selector_kwargs: Selector-specific parameters for filtering
        
    Returns:
        List of paths to training repeat directories
    """
    subdir = target + "_bad" if bad_start else target
    directory = project_root / "output" / "training" / subdir
    
    # Build experiment pattern with selector-specific parameters
    experiment_parts = [oracle_name, selector_name, str(initial_size), str(batch_size)]
    
    # Add selector-specific parameters to filter pattern
    if selector_kwargs:
        # For UCBSelector, include beta in the pattern
        if selector_name == "UCBSelector" and "beta" in selector_kwargs:
            beta_str = f"beta{selector_kwargs['beta']:.1f}".replace('.', 'p')
            experiment_parts.append(beta_str)
            
        # For LLMWorkflowSelector ablation study, include disabled tools pattern
        if "LLMWorkflow" in selector_name:
            ablation_parts = []
            if selector_kwargs.get('disable_ucb', False):
                ablation_parts.append("noUCB")
            if selector_kwargs.get('disable_smarts', False):
                ablation_parts.append("noSMARTS")
            if selector_kwargs.get('disable_tanimoto', False):
                ablation_parts.append("noTanimoto")
            
            if ablation_parts:
                ablation_str = "_".join(ablation_parts)
                experiment_parts.append(f"ablation_{ablation_str}")
    
    experiment_pattern = "_".join(experiment_parts)
    pattern = re.compile(rf"{experiment_pattern}_(\d+)_(\d+)$")

    files = [
        f for f in directory.glob(f"{experiment_pattern}_*_*")
        if pattern.fullmatch(f.name)
    ]
    return files

def get_training_data(results_file: Path) -> pd.DataFrame:
    """Get the training data from a results file.
    
    Args:
        results_file: Path to the results file
        
    Returns:
        DataFrame containing training metrics
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract data
    cycles = [cycle["cycle"] for cycle in results["al_cycles"]]
    total_labeled = [cycle["total_labeled"] for cycle in results["al_cycles"]]
    metrics = [cycle["performance_metrics"] for cycle in results["al_cycles"]]
    selected_smiles = results["al_cycles"][-1]["samples_selected"]
    data_path = results["experiment_config"]["data_path"]
    data_dict = pd.read_csv(data_path).set_index("SMILES")['affinity'].to_dict()
    batch_size = results["experiment_config"]["batch_size"]
    fitness = [data_dict[smiles] for smiles in selected_smiles]
    cycle_added = results["al_cycles"][-1]['cycle_added'] if 'cycle_added' in results["al_cycles"][-1] else [i//batch_size for i in range(len(selected_smiles))]

    overall_max = [0]
    cycle = 0
    for i, fitness in enumerate(fitness):
        if cycle_added[i] > cycle:
            overall_max.append(overall_max[-1])
            cycle += 1
        if fitness > overall_max[-1]:
            overall_max[-1] = fitness

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics)
    metrics_df["total_labeled"] = total_labeled
    metrics_df["overall_max"] = overall_max
    return metrics_df