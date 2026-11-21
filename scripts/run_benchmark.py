#!/usr/bin/env python3
"""
Benchmark script for comparing different model implementations and configurations.

This script provides comprehensive benchmarking capabilities:
- Performance comparison between different models
- Memory usage analysis
- Training time measurement
- Inference speed testing
- Robustness evaluation
- Scalability analysis
"""

import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import logging
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.spo import spo_plus_loss_torch
from src.optimization.spo_optimized import benchmark_spo_implementations
from src.optimization.robustness_analysis import RobustnessAnalyzer
from src.forecasting.tft import TFTQuantile
from src.forecasting.robust_models import RobustTFT, EnsembleTFT
from src.forecasting.metrics import DEFAULT_TAUS, evaluate_forecast
from src.optimization.ems import BatterySpec


class BenchmarkRunner:
    """
    Comprehensive benchmark runner for model comparison.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize the benchmark runner.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "benchmark.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
    def create_test_data(self, n_samples: int = 1000, horizon: int = 24,
                        n_features: int = 10, n_targets: int = 2) -> Dict[str, torch.Tensor]:
        """
        Create test data for benchmarking.
        
        Args:
            n_samples: Number of samples
            horizon: Prediction horizon
            n_features: Number of input features
            n_targets: Number of target variables
            
        Returns:
            Dictionary with test data
        """
        self.logger.info(f"Creating test data: {n_samples} samples, {horizon} horizon, "
                        f"{n_features} features, {n_targets} targets")
        
        # Create synthetic data
        X = torch.randn(n_samples, horizon, n_features)
        y = torch.randn(n_samples, horizon, n_targets, len(DEFAULT_TAUS))
        
        # Split into train/val/test
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        
        X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
        y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
        
        # Create data loaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader
        }
        
    def benchmark_model_training(self, model_class: nn.Module, model_config: Dict[str, Any],
                               train_loader: DataLoader, val_loader: DataLoader,
                               n_epochs: int = 10) -> Dict[str, Any]:
        """
        Benchmark model training performance.
        
        Args:
            model_class: Model class to benchmark
            model_config: Model configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of training epochs
            
        Returns:
            Dictionary with training benchmark results
        """
        self.logger.info(f"Benchmarking {model_class.__name__} training")
        
        # Initialize model
        model = model_class(**model_config).to(self.device)
        
        # Initialize optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Track metrics
        train_times = []
        memory_usage = []
        train_losses = []
        val_losses = []
        
        # Training loop
        for epoch in range(n_epochs):
            # Memory tracking
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            start_time = time.time()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Training
            model.train()
            epoch_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                
            # Validation
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    epoch_val_loss += loss.item()
            
            # Calculate metrics
            epoch_train_loss /= len(train_loader)
            epoch_val_loss /= len(val_loader)
            
            end_time = time.time()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            train_times.append(end_time - start_time)
            memory_usage.append(final_memory - initial_memory)
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            
            self.logger.info(f"Epoch {epoch+1}/{n_epochs}: "
                           f"Train Loss: {epoch_train_loss:.4f}, "
                           f"Val Loss: {epoch_val_loss:.4f}, "
                           f"Time: {train_times[-1]:.2f}s, "
                           f"Memory: {memory_usage[-1]:.2f}MB")
        
        # Calculate summary statistics
        results = {
            'model_name': model_class.__name__,
            'model_config': model_config,
            'n_epochs': n_epochs,
            'total_time': sum(train_times),
            'avg_time_per_epoch': np.mean(train_times),
            'std_time_per_epoch': np.std(train_times),
            'avg_memory_usage': np.mean(memory_usage),
            'std_memory_usage': np.std(memory_usage),
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_times': train_times,
            'memory_usage': memory_usage
        }
        
        return results
        
    def benchmark_model_inference(self, model: nn.Module, test_loader: DataLoader,
                                n_runs: int = 10) -> Dict[str, Any]:
        """
        Benchmark model inference performance.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            n_runs: Number of inference runs
            
        Returns:
            Dictionary with inference benchmark results
        """
        self.logger.info(f"Benchmarking {model.__class__.__name__} inference")
        
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                _ = model(batch_x)
        
        # Benchmark inference
        inference_times = []
        memory_usage = []
        
        with torch.no_grad():
            for run in range(n_runs):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                start_time = time.time()
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # Run inference on entire test set
                for batch_x, _ in test_loader:
                    batch_x = batch_x.to(self.device)
                    _ = model(batch_x)
                
                end_time = time.time()
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                inference_times.append(end_time - start_time)
                memory_usage.append(final_memory - initial_memory)
        
        # Calculate summary statistics
        results = {
            'model_name': model.__class__.__name__,
            'n_runs': n_runs,
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'avg_memory_usage': np.mean(memory_usage),
            'std_memory_usage': np.std(memory_usage),
            'inference_times': inference_times,
            'memory_usage': memory_usage
        }
        
        return results
        
    def benchmark_spo_performance(self, batch_sizes: List[int] = [8, 16, 32, 64, 128],
                               horizon: int = 24) -> Dict[str, Any]:
        """
        Benchmark SPO performance improvements.
        
        Args:
            batch_sizes: List of batch sizes to test
            horizon: Prediction horizon
            
        Returns:
            Dictionary with SPO benchmark results
        """
        self.logger.info("Benchmarking SPO performance improvements")
        
        results = benchmark_spo_implementations(batch_sizes, horizon=horizon, n_runs=10)
        
        # Calculate summary statistics
        summary = {
            'batch_sizes': batch_sizes,
            'results': results,
            'avg_speedup': np.mean([r['speedup'] for r in results.values()]),
            'max_speedup': max(r['speedup'] for r in results.values()),
            'min_speedup': min(r['speedup'] for r in results.values())
        }
        
        return summary
        
    def benchmark_robustness(self, model: nn.Module, test_loader: DataLoader,
                           n_samples: int = 100) -> Dict[str, Any]:
        """
        Benchmark model robustness.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            n_samples: Number of samples for robustness testing
            
        Returns:
            Dictionary with robustness benchmark results
        """
        self.logger.info(f"Benchmarking {model.__class__.__name__} robustness")
        
        analyzer = RobustnessAnalyzer()
        
        # Extract test data
        test_data = []
        for batch_x, batch_y in test_loader:
            test_data.append((batch_x.cpu().numpy(), batch_y.cpu().numpy()))
        test_data = np.concatenate([x[0] for x in test_data]), np.concatenate([x[1] for x in test_data])
        
        X_test, y_test = test_data[0][:n_samples], test_data[1][:n_samples]
        
        # Convert to tensors
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        
        # Benchmark adversarial attacks
        adversarial_results = analyzer.analyze_adversarial_attacks(
            model, X_tensor, y_tensor, epsilon=0.1
        )
        
        # Benchmark noise robustness
        noise_results = analyzer.analyze_noise_robustness(
            y_test[:, :, 0], y_test[:, :, 0],  # Using same data for simplicity
            noise_levels=[0.01, 0.05, 0.1, 0.2]
        )
        
        # Benchmark distribution shift
        shift_results = analyzer.analyze_distribution_shift(
            y_test[:, :, 0], y_test[:, :, 0],
            shift_types=['mean', 'std', 'seasonal']
        )
        
        results = {
            'adversarial_attacks': adversarial_results,
            'noise_robustness': noise_results,
            'distribution_shift': shift_results
        }
        
        return results
        
    def run_comprehensive_benchmark(self, models_config: List[Dict[str, Any]],
                                  n_samples: int = 1000, horizon: int = 24,
                                  n_epochs: int = 10, n_runs: int = 10) -> Dict[str, Any]:
        """
        Run comprehensive benchmark on multiple models.
        
        Args:
            models_config: List of model configurations
            n_samples: Number of test samples
            horizon: Prediction horizon
            n_epochs: Number of training epochs
            n_runs: Number of inference runs
            
        Returns:
            Dictionary with comprehensive benchmark results
        """
        self.logger.info("Starting comprehensive benchmark")
        
        # Create test data
        test_data = self.create_test_data(n_samples, horizon)
        
        results = {
            'benchmark_info': {
                'n_samples': n_samples,
                'horizon': horizon,
                'n_epochs': n_epochs,
                'n_runs': n_runs,
                'device': str(self.device)
            },
            'models': {}
        }
        
        # Benchmark each model
        for model_config in models_config:
            model_class = model_config['class']
            model_name = model_config['name']
            model_params = model_config['params']
            
            self.logger.info(f"Benchmarking {model_name}")
            
            # Benchmark training
            training_results = self.benchmark_model_training(
                model_class, model_params,
                test_data['train_loader'], test_data['val_loader'],
                n_epochs
            )
            
            # Create model for inference
            model = model_class(**model_params).to(self.device)
            
            # Benchmark inference
            inference_results = self.benchmark_model_inference(
                model, test_data['test_loader'], n_runs
            )
            
            # Benchmark robustness
            robustness_results = self.benchmark_robustness(
                model, test_data['test_loader'], n_samples // 10
            )
            
            # Store results
            results['models'][model_name] = {
                'training': training_results,
                'inference': inference_results,
                'robustness': robustness_results
            }
        
        # Benchmark SPO performance
        spo_results = self.benchmark_spo_performance()
        results['spo_benchmark'] = spo_results
        
        # Save results
        results_path = self.output_dir / "comprehensive_benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Benchmark completed. Results saved to {results_path}")
        
        return results
        
    def generate_benchmark_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Benchmark info
        info = results['benchmark_info']
        report.append("BENCHMARK CONFIGURATION")
        report.append("-" * 40)
        report.append(f"Number of samples: {info['n_samples']}")
        report.append(f"Prediction horizon: {info['horizon']}")
        report.append(f"Training epochs: {info['n_epochs']}")
        report.append(f"Inference runs: {info['n_runs']}")
        report.append(f"Device: {info['device']}")
        report.append("")
        
        # Model comparison
        report.append("MODEL COMPARISON")
        report.append("-" * 40)
        
        training_summary = []
        inference_summary = []
        
        for model_name, model_results in results['models'].items():
            training = model_results['training']
            inference = model_results['inference']
            
            training_summary.append({
                'name': model_name,
                'total_time': training['total_time'],
                'avg_time_per_epoch': training['avg_time_per_epoch'],
                'final_val_loss': training['final_val_loss'],
                'avg_memory': training['avg_memory_usage']
            })
            
            inference_summary.append({
                'name': model_name,
                'avg_inference_time': inference['avg_inference_time'],
                'avg_memory': inference['avg_memory_usage']
            })
        
        # Sort by validation loss (lower is better)
        training_summary.sort(key=lambda x: x['final_val_loss'])
        
        # Training summary table
        report.append("Training Performance (sorted by validation loss):")
        report.append(f"{'Model':<20} {'Total Time (s)':<15} {'Time/Epoch (s)':<15} "
                     f"{'Val Loss':<10} {'Memory (MB)':<12}")
        for model in training_summary:
            report.append(f"{model['name']:<20} {model['total_time']:<15.2f} "
                         f"{model['avg_time_per_epoch']:<15.2f} "
                         f"{model['final_val_loss']:<10.4f} "
                         f"{model['avg_memory']:<12.2f}")
        report.append("")
        
        # Inference summary table
        inference_summary.sort(key=lambda x: x['avg_inference_time'])
        
        report.append("Inference Performance (sorted by inference time):")
        report.append(f"{'Model':<20} {'Avg Time (s)':<15} {'Memory (MB)':<12}")
        for model in inference_summary:
            report.append(f"{model['name']:<20} {model['avg_inference_time']:<15.4f} "
                         f"{model['avg_memory']:<12.2f}")
        report.append("")
        
        # SPO benchmark results
        if 'spo_benchmark' in results:
            report.append("SPO PERFORMANCE IMPROVEMENTS")
            report.append("-" * 40)
            
            spo_results = results['spo_benchmark']
            report.append(f"Average speedup: {spo_results['avg_speedup']:.2f}x")
            report.append(f"Maximum speedup: {spo_results['max_speedup']:.2f}x")
            report.append(f"Minimum speedup: {spo_results['min_speedup']:.2f}x")
            report.append("")
            
            report.append("Batch size performance:")
            for batch_size, metrics in spo_results['results'].items():
                report.append(f"  Batch {batch_size}: {metrics['speedup']:.2f}x speedup "
                            f"({metrics['optimized_mean']:.4f}s vs {metrics['original_mean']:.4f}s)")
            report.append("")
        
        # Robustness analysis
        report.append("ROBUSTNESS ANALYSIS")
        report.append("-" * 40)
        
        for model_name, model_results in results['models'].items():
            robustness = model_results['robustness']
            
            report.append(f"{model_name}:")
            
            # Adversarial attacks
            if 'adversarial_attacks' in robustness:
                attacks = robustness['adversarial_attacks']
                worst_attack = max(attacks.items(), key=lambda x: x[1]['degradation'])
                report.append(f"  Most severe adversarial attack: {worst_attack[0]} "
                            f"(degradation: {worst_attack[1]['degradation']:.2%})")
            
            # Noise robustness
            if 'noise_robustness' in robustness:
                noise = robustness['noise_robustness']
                worst_noise = max(noise.items(), key=lambda x: x[1]['crps'])
                report.append(f"  Most affected noise level: {worst_noise[0]} "
                            f"(CRPS: {worst_noise[1]['crps']:.4f})")
            
            # Distribution shift
            if 'distribution_shift' in robustness:
                shift = robustness['distribution_shift']
                worst_shift = min(shift.items(), key=lambda x: x[1]['crps'])
                report.append(f"  Most affected distribution shift: {worst_shift[0]} "
                            f"(CRPS: {worst_shift[1]['crps']:.4f})")
            
            report.append("")
        
        # Overall recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        # Best model for training
        best_training = min(training_summary, key=lambda x: x['final_val_loss'])
        report.append(f"Best model for training: {best_training['name']} "
                    f"(validation loss: {best_training['final_val_loss']:.4f})")
        
        # Best model for inference
        best_inference = min(inference_summary, key=lambda x: x['avg_inference_time'])
        report.append(f"Best model for inference: {best_inference['name']} "
                    f"(inference time: {best_inference['avg_inference_time']:.4f}s)")
        
        # Overall best trade-off
        trade_off_scores = []
        for model in training_summary:
            # Combine validation loss and inference time (normalized)
            val_loss_norm = (model['final_val_loss'] - min(t['final_val_loss'] for t in training_summary)) / \
                          (max(t['final_val_loss'] for t in training_summary) - min(t['final_val_loss'] for t in training_summary) + 1e-8)
            
            inference_time = next(i['avg_inference_time'] for i in inference_summary if i['name'] == model['name'])
            inference_time_norm = (inference_time - min(i['avg_inference_time'] for i in inference_summary)) / \
                                (max(i['avg_inference_time'] for i in inference_summary) - min(i['avg_inference_time'] for i in inference_summary) + 1e-8)
            
            # Combined score (lower is better)
            combined_score = val_loss_norm + inference_time_norm
            trade_off_scores.append((model['name'], combined_score))
        
        best_trade_off = min(trade_off_scores, key=lambda x: x[1])
        report.append(f"Best overall trade-off: {best_trade_off[0]} (score: {best_trade_off[1]:.4f})")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / "benchmark_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        return report_text


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Run comprehensive model benchmarks")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Directory to save benchmark results")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of test samples")
    parser.add_argument("--horizon", type=int, default=24,
                       help="Prediction horizon")
    parser.add_argument("--n-epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--n-runs", type=int, default=10,
                       help="Number of inference runs")
    parser.add_argument("--models", type=str, nargs='+',
                       default=["TFT", "RobustTFT", "EnsembleTFT"],
                       help="Models to benchmark")
    
    args = parser.parse_args()
    
    # Model configurations
    models_config = [
        {
            'class': TFTQuantile,
            'name': 'TFT',
            'params': {
                'n_features': 10,
                'horizon': args.horizon,
                'n_targets': 2,
                'hidden': 64,
                'taus': DEFAULT_TAUS
            }
        },
        {
            'class': RobustTFT,
            'name': 'RobustTFT',
            'params': {
                'n_features': 10,
                'horizon': args.horizon,
                'n_targets': 2,
                'hidden': 64,
                'taus': DEFAULT_TAUS,
                'dropout_rate': 0.1,
                'use_mc_dropout': True
            }
        },
        {
            'class': EnsembleTFT,
            'name': 'EnsembleTFT',
            'params': {
                'n_models': 3,
                'n_features': 10,
                'horizon': args.horizon,
                'n_targets': 2,
                'hidden': 64,
                'taus': DEFAULT_TAUS
            }
        }
    ]
    
    # Filter models based on command line arguments
    if args.models:
        models_config = [m for m in models_config if m['name'] in args.models]
    
    # Create benchmark runner
    benchmark = BenchmarkRunner(args.output_dir)
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(
        models_config,
        args.n_samples,
        args.horizon,
        args.n_epochs,
        args.n_runs
    )
    
    # Generate report
    report = benchmark.generate_benchmark_report(results)
    print(report)
    
    print(f"\nBenchmark completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()