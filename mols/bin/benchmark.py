#!/usr/bin/env python3
"""
Benchmarks the performance of a selector, oracle, initial_size, batch_size combination by running the AL.py script 3 times.
The script then calls plot_benchmark on the given selector, oracle, initial_size, batch_size combination.
"""

import argparse
import json
from pathlib import Path
import sys
import random
import numpy as np
import shutil
import traceback
# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.PlotUtils import plot_benchmark, plot_training_results
from bin.AL import ActiveLearningExperiment
from utils.DataUtils import get_all_training_repeats, ExperimentConfig

def run_benchmark(
    model_name: str,
    selector_name: str,
    data_path: str,
    batch_size: int = 60,
    initial_size: int = 60,
    num_repeats: int = 3,
    replace_existing: bool = False,
    random_seed: int = 42,
    model_kwargs: dict = None,
    selector_kwargs: dict = None,
    bad_start: bool = False
) -> None:
    """Run multiple AL experiments and plot benchmark results.
    
    Args:
        model_name: Name of the model class to use
        selector_name: Name of the selector class to use
        data_path: Path to dataset CSV file
        batch_size: Number of samples to select per cycle
        initial_size: Size of initial training set
        num_repeats: Number of times to repeat the experiment
        replace_existing: Replace existing benchmark results
        random_seed: Random seed for the experiment
        model_kwargs: Model-specific parameters
        selector_kwargs: Selector-specific parameters
        bad_start: Whether to use bad start initializationss
    """
    # Set default kwargs
    model_kwargs = model_kwargs or {}
    selector_kwargs = selector_kwargs or {}
    np.random.seed(random_seed)
    target = Path(data_path).stem
    
    # Delete existing benchmarks if replace_existing is True
    if replace_existing:
        for repeat in get_all_training_repeats(model_name, selector_name, initial_size, batch_size, target, bad_start, selector_kwargs):
            print(f"Deleting existing benchmark {repeat}")
            shutil.rmtree(repeat)
    
    repeats = get_all_training_repeats(model_name, selector_name, initial_size, batch_size, target, bad_start, selector_kwargs)
    # Remove repeats from list if "Training_results.json" doesnt exist in directory
    repeats = [repeat for repeat in repeats if (repeat / "Training_results.json").exists()]
    print(f"Repeats: {repeats}")
    # Run AL experiments
    for i in range(num_repeats):
        print(f"\nRunning experiment {i+1}/{num_repeats}")
        # Ensure random seed is unique (for the same selector, model, initial_size, batch_size) ignore 
        while random_seed in [config['experiment_config']['random_seed'] for config in [json.load(open(repeat / "Training_results.json")) for repeat in repeats]]:
            random_seed = random_seed + 1
        print(f"Random seed: {random_seed}")
        try:
            # Create experiment configuration
            config = ExperimentConfig(
                model_name=model_name,
                selector_name=selector_name,
                data_path=data_path,
                batch_size=batch_size,
                initial_size=initial_size,
                random_seed=random_seed,    
                model_kwargs=model_kwargs.copy(),  
                selector_kwargs=selector_kwargs.copy(),
                bad_start=bad_start
            )
            print(config)
            # Run experiment
            experiment = ActiveLearningExperiment(config=config)
            results_file, output_dir = experiment.run()
            plot_training_results(results_file, output_dir)
           
            random_seed = random_seed + 1
        except Exception as e:
            print(f"Error running experiment {i+1}: {e}")
            traceback.print_exc()
            continue
    return 

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark active learning performance")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True,
                      help="Oracle model class name (must be in aloracles/)")
    parser.add_argument("--selector", type=str, required=True,
                      help="Selector class name (must be in alselectors/)")
    parser.add_argument("--data", type=str, required=True,
                      help="Path to dataset CSV file")
    
    # Optional arguments
    parser.add_argument("--batch_size", type=int, default=60,
                      help="Samples per AL cycle (default: 60)")
    parser.add_argument("--initial_size", type=int, default=60,
                      help="Initial training set size (default: 60)")
    parser.add_argument("--num_repeats", type=int, default=3,
                      help="Number of times to repeat the experiment (default: 3)")
    
    # Additional kwargs
    parser.add_argument("--model_kwargs", type=str, default="{}",
                      help="JSON string of model-specific kwargs")
    parser.add_argument("--selector_kwargs", type=str, default="{}",
                      help="JSON string of selector-specific kwargs")
    parser.add_argument("--replace_existing", action="store_true",
                      help="Replace existing benchmark results")
    parser.add_argument("--bad_start", action="store_true",
                      help="Use bad start initialization")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(repr(args.model_kwargs))
    run_benchmark(
        model_name=args.model,
        selector_name=args.selector,
        data_path=args.data,
        batch_size=args.batch_size,
        initial_size=args.initial_size,
        num_repeats=args.num_repeats,
        model_kwargs=json.loads(args.model_kwargs),
        selector_kwargs=json.loads(args.selector_kwargs),
        replace_existing=args.replace_existing,
        bad_start=args.bad_start
    )
    target = Path(args.data).stem
    print("\nPlotting benchmark results...")
    plot_benchmark(
        target=target,
        oracle_name=args.model,
        selector_name=args.selector,
        initial_size=args.initial_size,
        batch_size=args.batch_size,
        bad_start=args.bad_start,
        selector_kwargs=json.loads(args.selector_kwargs)
    )

