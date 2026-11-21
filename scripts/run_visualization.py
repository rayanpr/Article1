#!/usr/bin/env python3
"""
Visualization script for generating comprehensive visualizations of EMS results.

This script provides:
- Time series visualization with confidence intervals
- Cost comparison and regret analysis
- Battery state of charge visualization
- Feature importance plots
- Uncertainty quantification visualizations
- Model performance comparisons
- Robustness analysis visualizations
- Interactive dashboard generation
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.visualization import EMSVisualizer
from src.optimization.benchmark import BenchmarkRunner
from src.optimization.robustness_analysis import RobustnessAnalyzer
from src.forecasting.tft import TFTQuantile
from src.forecasting.robust_models import RobustTFT, EnsembleTFT
from src.optimization.ems import BatterySpec


def load_results(results_path: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        Dictionary with results
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def create_sample_data(n_samples: int = 1000, horizon: int = 24) -> Dict[str, Any]:
    """
    Create sample data for visualization.
    
    Args:
        n_samples: Number of samples
        horizon: Prediction horizon
        
    Returns:
        Dictionary with sample data
    """
    # Create timestamps
    timestamps = pd.date_range(start='2022-01-01', periods=n_samples, freq='H')
    
    # Create synthetic data
    actual_values = np.random.randn(n_samples) * 10 + 50
    predicted_values = np.random.randn(n_samples) * 8 + 48
    confidence_intervals = np.column_stack([
        predicted_values - 5,
        predicted_values + 5
    ])
    
    # Create battery data
    state_of_charge = np.linspace(20, 80, n_samples)
    charge_schedule = np.random.randn(n_samples) * 5 + 2
    discharge_schedule = np.random.randn(n_samples) * 5 - 2
    
    # Create cost data
    methods = ['TFT', 'XGBoost', 'LightGBM', 'Oracle']
    costs = [1000, 1200, 1100, 800]
    regrets = [0.1, 0.15, 0.12, 0.0]
    
    # Create feature importance data
    feature_names = [f'feature_{i}' for i in range(20)]
    importance_scores = np.random.rand(20)
    
    # Create uncertainty data
    uncertainty_intervals = np.column_stack([
        np.random.rand(n_samples) * 2,
        np.random.rand(n_samples) * 2 + 2
    ])
    
    # Create model performance data
    model_names = ['TFT', 'XGBoost', 'LightGBM']
    metrics = {
        'Accuracy': [0.85, 0.82, 0.88],
        'Precision': [0.83, 0.80, 0.86],
        'Recall': [0.87, 0.84, 0.90],
        'F1-Score': [0.85, 0.82, 0.88]
    }
    
    # Create quantiles for fan chart
    quantiles = np.random.rand(n_samples, 5)
    
    # Create robustness results
    robustness_results = {
        'adversarial_attacks': {
            'fgsm': {'degradation': 0.3},
            'pgd': {'degradation': 0.5}
        },
        'noise_robustness': {
            'noise_0.01': {'crps': 0.1},
            'noise_0.1': {'crps': 0.3}
        },
        'distribution_shift': {
            'mean': {'crps': 0.2},
            'std': {'crps': 0.4}
        }
    }
    
    return {
        'timestamps': timestamps,
        'actual_values': actual_values,
        'predicted_values': predicted_values,
        'confidence_intervals': confidence_intervals,
        'state_of_charge': state_of_charge,
        'charge_schedule': charge_schedule,
        'discharge_schedule': discharge_schedule,
        'methods': methods,
        'costs': costs,
        'regrets': regrets,
        'feature_names': feature_names,
        'importance_scores': importance_scores,
        'uncertainty_intervals': uncertainty_intervals,
        'model_names': model_names,
        'metrics': metrics,
        'quantiles': quantiles,
        'robustness_results': robustness_results
    }


def run_time_series_visualization(visualizer: EMSVisualizer, data: Dict[str, Any],
                                output_dir: Path) -> None:
    """
    Run time series visualization.
    
    Args:
        visualizer: EMSVisualizer instance
        data: Sample data
        output_dir: Output directory
    """
    print("Generating time series visualization...")
    
    # Time series with confidence intervals
    visualizer.plot_time_series_with_confidence(
        data['timestamps'],
        data['actual_values'],
        data['predicted_values'],
        data['confidence_intervals'],
        title="Time Series with Confidence Intervals",
        save_path=str(output_dir / "time_series_with_confidence.png")
    )
    
    # Forecast fan chart
    visualizer.plot_forecast_fan_chart(
        data['timestamps'],
        data['quantiles'],
        title="Forecast Fan Chart",
        save_path=str(output_dir / "forecast_fan_chart.png")
    )
    
    print("Time series visualization completed.")


def run_cost_visualization(visualizer: EMSVisualizer, data: Dict[str, Any],
                          output_dir: Path) -> None:
    """
    Run cost visualization.
    
    Args:
        visualizer: EMSVisualizer instance
        data: Sample data
        output_dir: Output directory
    """
    print("Generating cost visualization...")
    
    # Cost comparison
    visualizer.plot_cost_comparison(
        data['methods'],
        data['costs'],
        data['regrets'],
        title="Cost Comparison and Regret Analysis",
        save_path=str(output_dir / "cost_comparison.png")
    )
    
    print("Cost visualization completed.")


def run_battery_visualization(visualizer: EMSVisualizer, data: Dict[str, Any],
                            output_dir: Path) -> None:
    """
    Run battery visualization.
    
    Args:
        visualizer: EMSVisualizer instance
        data: Sample data
        output_dir: Output directory
    """
    print("Generating battery visualization...")
    
    # Battery state of charge
    visualizer.plot_battery_state(
        data['timestamps'],
        data['state_of_charge'],
        data['charge_schedule'],
        data['discharge_schedule'],
        title="Battery State of Charge and Schedule",
        save_path=str(output_dir / "battery_state.png")
    )
    
    print("Battery visualization completed.")


def run_feature_importance_visualization(visualizer: EMSVisualizer, data: Dict[str, Any],
                                       output_dir: Path) -> None:
    """
    Run feature importance visualization.
    
    Args:
        visualizer: EMSVisualizer instance
        data: Sample data
        output_dir: Output directory
    """
    print("Generating feature importance visualization...")
    
    # Feature importance
    visualizer.plot_feature_importance(
        data['feature_names'],
        data['importance_scores'],
        title="Feature Importance",
        save_path=str(output_dir / "feature_importance.png"),
        top_n=15
    )
    
    print("Feature importance visualization completed.")


def run_uncertainty_visualization(visualizer: EMSVisualizer, data: Dict[str, Any],
                                output_dir: Path) -> None:
    """
    Run uncertainty visualization.
    
    Args:
        visualizer: EMSVisualizer instance
        data: Sample data
        output_dir: Output directory
    """
    print("Generating uncertainty visualization...")
    
    # Uncertainty quantification
    visualizer.plot_uncertainty_quantification(
        data['actual_values'],
        data['predicted_values'],
        data['uncertainty_intervals'],
        title="Uncertainty Quantification",
        save_path=str(output_dir / "uncertainty_quantification.png")
    )
    
    print("Uncertainty visualization completed.")


def run_model_performance_visualization(visualizer: EMSVisualizer, data: Dict[str, Any],
                                      output_dir: Path) -> None:
    """
    Run model performance visualization.
    
    Args:
        visualizer: EMSVisualizer instance
        data: Sample data
        output_dir: Output directory
    """
    print("Generating model performance visualization...")
    
    # Model performance comparison
    visualizer.plot_model_performance_comparison(
        data['model_names'],
        data['metrics'],
        title="Model Performance Comparison",
        save_path=str(output_dir / "model_performance_comparison.png")
    )
    
    print("Model performance visualization completed.")


def run_robustness_visualization(visualizer: EMSVisualizer, data: Dict[str, Any],
                               output_dir: Path) -> None:
    """
    Run robustness visualization.
    
    Args:
        visualizer: EMSVisualizer instance
        data: Sample data
        output_dir: Output directory
    """
    print("Generating robustness visualization...")
    
    # Robustness analysis
    visualizer.plot_robustness_analysis(
        data['robustness_results'],
        title="Robustness Analysis",
        save_path=str(output_dir / "robustness_analysis.png")
    )
    
    print("Robustness visualization completed.")


def run_comprehensive_dashboard(visualizer: EMSVisualizer, data: Dict[str, Any],
                               output_dir: Path) -> None:
    """
    Run comprehensive dashboard.
    
    Args:
        visualizer: EMSVisualizer instance
        data: Sample data
        output_dir: Output directory
    """
    print("Generating comprehensive dashboard...")
    
    # Create dashboard
    dashboard_data = {
        'cost_comparison': {
            'methods': data['methods'],
            'costs': data['costs'],
            'regrets': data['regrets']
        },
        'model_performance': data['metrics'],
        'robustness': {
            'Adversarial Attacks': 0.4,
            'Noise Robustness': 0.2,
            'Distribution Shift': 0.3
        },
        'feature_importance': {
            'features': data['feature_names'][:10],
            'importance': data['importance_scores'][:10]
        },
        'training_metrics': {
            'train_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
            'val_loss': [0.6, 0.5, 0.4, 0.3, 0.2]
        },
        'inference_metrics': {
            'models': data['model_names'],
            'times': [0.1, 0.2, 0.15]
        },
        'memory_metrics': {
            'models': data['model_names'],
            'memory': [100, 150, 120]
        }
    }
    
    visualizer.create_dashboard(
        dashboard_data,
        save_path=str(output_dir / "comprehensive_dashboard.png")
    )
    
    print("Comprehensive dashboard completed.")


def run_all_visualizations(output_dir: str, use_sample_data: bool = True,
                          results_path: Optional[str] = None) -> None:
    """
    Run all visualizations.
    
    Args:
        output_dir: Output directory
        use_sample_data: Whether to use sample data
        results_path: Path to results file (if not using sample data)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create visualizer
    visualizer = EMSVisualizer(output_dir)
    
    # Load or create data
    if use_sample_data:
        data = create_sample_data()
    else:
        if results_path is None:
            raise ValueError("results_path must be provided when use_sample_data is False")
        data = load_results(results_path)
    
    print(f"Starting visualization generation in {output_dir}")
    
    # Run all visualizations
    try:
        run_time_series_visualization(visualizer, data, output_path)
        run_cost_visualization(visualizer, data, output_path)
        run_battery_visualization(visualizer, data, output_path)
        run_feature_importance_visualization(visualizer, data, output_path)
        run_uncertainty_visualization(visualizer, data, output_path)
        run_model_performance_visualization(visualizer, data, output_path)
        run_robustness_visualization(visualizer, data, output_path)
        run_comprehensive_dashboard(visualizer, data, output_path)
        
        print(f"All visualizations completed successfully. Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error during visualization generation: {str(e)}")
        raise


def main():
    """Main function to run visualizations."""
    parser = argparse.ArgumentParser(description="Generate EMS visualizations")
    parser.add_argument("--output-dir", type=str, default="visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--use-sample-data", action="store_true",
                       help="Use sample data for visualization")
    parser.add_argument("--results-path", type=str, default=None,
                       help="Path to results file (if not using sample data)")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples for sample data")
    parser.add_argument("--horizon", type=int, default=24,
                       help="Prediction horizon for sample data")
    
    args = parser.parse_args()
    
    # Run visualizations
    run_all_visualizations(
        args.output_dir,
        args.use_sample_data,
        args.results_path
    )


if __name__ == "__main__":
    main()