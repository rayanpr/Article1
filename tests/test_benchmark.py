import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.benchmark import BenchmarkRunner
from src.forecasting.tft import TFTQuantile
from src.forecasting.robust_models import RobustTFT, EnsembleTFT
from src.optimization.ems import BatterySpec


class TestBenchmarkRunner:
    """Test suite for BenchmarkRunner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for benchmark results
        self.temp_dir = tempfile.mkdtemp()
        self.runner = BenchmarkRunner(self.temp_dir)
        
        # Create test data
        self.n_samples = 100
        self.horizon = 24
        self.n_features = 10
        self.n_targets = 2
        
        # Test model configurations
        self.models_config = [
            {
                'class': TFTQuantile,
                'name': 'TFT',
                'params': {
                    'n_features': self.n_features,
                    'horizon': self.horizon,
                    'n_targets': self.n_targets,
                    'hidden': 64,
                    'taus': [0.1, 0.25, 0.5, 0.75, 0.9]
                }
            },
            {
                'class': RobustTFT,
                'name': 'RobustTFT',
                'params': {
                    'n_features': self.n_features,
                    'horizon': self.horizon,
                    'n_targets': self.n_targets,
                    'hidden': 64,
                    'taus': [0.1, 0.25, 0.5, 0.75, 0.9],
                    'dropout_rate': 0.1,
                    'use_mc_dropout': True
                }
            }
        ]
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def test_init(self):
        """Test BenchmarkRunner initialization."""
        assert self.runner.output_dir == Path(self.temp_dir)
        assert self.runner.device is not None
        
    def test_create_test_data(self):
        """Test test data creation."""
        data = self.runner.create_test_data(
            n_samples=self.n_samples,
            horizon=self.horizon,
            n_features=self.n_features,
            n_targets=self.n_targets
        )
        
        # Check data structure
        assert 'X_train' in data
        assert 'y_train' in data
        assert 'X_val' in data
        assert 'y_val' in data
        assert 'X_test' in data
        assert 'y_test' in data
        assert 'train_loader' in data
        assert 'val_loader' in data
        assert 'test_loader' in data
        
        # Check data shapes
        assert data['X_train'].shape == (70, self.horizon, self.n_features)
        assert data['y_train'].shape == (70, self.horizon, self.n_targets, 5)
        assert data['X_val'].shape == (15, self.horizon, self.n_features)
        assert data['y_val'].shape == (15, self.horizon, self.n_targets, 5)
        assert data['X_test'].shape == (15, self.horizon, self.n_features)
        assert data['y_test'].shape == (15, self.horizon, self.n_targets, 5)
        
    def test_benchmark_model_training(self):
        """Test model training benchmark."""
        # Create test data
        data = self.runner.create_test_data(n_samples=50)  # Smaller dataset for faster testing
        
        # Benchmark training
        results = self.runner.benchmark_model_training(
            self.models_config[0]['class'],
            self.models_config[0]['params'],
            data['train_loader'],
            data['val_loader'],
            n_epochs=2  # Fewer epochs for faster testing
        )
        
        # Check results structure
        assert 'model_name' in results
        assert 'model_config' in results
        assert 'n_epochs' in results
        assert 'total_time' in results
        assert 'avg_time_per_epoch' in results
        assert 'std_time_per_epoch' in results
        assert 'avg_memory_usage' in results
        assert 'std_memory_usage' in results
        assert 'final_train_loss' in results
        assert 'final_val_loss' in results
        assert 'train_losses' in results
        assert 'val_losses' in results
        assert 'train_times' in results
        assert 'memory_usage' in results
        
        # Check values
        assert results['model_name'] == 'TFT'
        assert results['n_epochs'] == 2
        assert len(results['train_losses']) == 2
        assert len(results['val_losses']) == 2
        assert len(results['train_times']) == 2
        assert len(results['memory_usage']) == 2
        
    def test_benchmark_model_inference(self):
        """Test model inference benchmark."""
        # Create test data
        data = self.runner.create_test_data(n_samples=50)  # Smaller dataset for faster testing
        
        # Create a simple model for testing
        model = nn.Sequential(
            nn.Linear(self.n_features, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_targets * 5)  # 5 quantiles
        )
        
        # Benchmark inference
        results = self.runner.benchmark_model_inference(
            model,
            data['test_loader'],
            n_runs=2  # Fewer runs for faster testing
        )
        
        # Check results structure
        assert 'model_name' in results
        assert 'n_runs' in results
        assert 'avg_inference_time' in results
        assert 'std_inference_time' in results
        assert 'min_inference_time' in results
        assert 'max_inference_time' in results
        assert 'avg_memory_usage' in results
        assert 'std_memory_usage' in results
        assert 'inference_times' in results
        assert 'memory_usage' in results
        
        # Check values
        assert results['model_name'] == 'Sequential'
        assert results['n_runs'] == 2
        assert len(results['inference_times']) == 2
        assert len(results['memory_usage']) == 2
        
    def test_benchmark_spo_performance(self):
        """Test SPO performance benchmark."""
        results = self.runner.benchmark_spo_performance(
            batch_sizes=[8, 16],
            horizon=24
        )
        
        # Check results structure
        assert 'batch_sizes' in results
        assert 'results' in results
        assert 'avg_speedup' in results
        assert 'max_speedup' in results
        assert 'min_speedup' in results
        
        # Check values
        assert results['batch_sizes'] == [8, 16]
        assert len(results['results']) == 2
        assert results['avg_speedup'] > 0
        assert results['max_speedup'] > 0
        assert results['min_speedup'] > 0
        
    def test_benchmark_robustness(self):
        """Test robustness benchmark."""
        # Create test data
        data = self.runner.create_test_data(n_samples=50)  # Smaller dataset for faster testing
        
        # Create a simple model for testing
        model = nn.Sequential(
            nn.Linear(self.n_features, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_targets * 5)  # 5 quantiles
        )
        
        # Benchmark robustness
        results = self.runner.benchmark_robustness(
            model,
            data['test_loader'],
            n_samples=10  # Fewer samples for faster testing
        )
        
        # Check results structure
        assert 'adversarial_attacks' in results
        assert 'noise_robustness' in results
        assert 'distribution_shift' in results
        
        # Check adversarial attacks results
        assert 'fgsm' in results['adversarial_attacks']
        assert 'pgd' in results['adversarial_attacks']
        
        # Check noise robustness results
        for key in results['noise_robustness'].keys():
            assert key.startswith('noise_')
            
        # Check distribution shift results
        assert 'mean' in results['distribution_shift']
        assert 'std' in results['distribution_shift']
        assert 'seasonal' in results['distribution_shift']
        
    def test_run_comprehensive_benchmark(self):
        """Test comprehensive benchmark."""
        # Run comprehensive benchmark with smaller dataset and fewer epochs
        results = self.runner.run_comprehensive_benchmark(
            self.models_config,
            n_samples=50,  # Smaller dataset
            horizon=24,
            n_epochs=2,    # Fewer epochs
            n_runs=2      # Fewer runs
        )
        
        # Check results structure
        assert 'benchmark_info' in results
        assert 'models' in results
        assert 'spo_benchmark' in results
        
        # Check benchmark info
        info = results['benchmark_info']
        assert info['n_samples'] == 50
        assert info['horizon'] == 24
        assert info['n_epochs'] == 2
        assert info['n_runs'] == 2
        
        # Check models results
        for model_name in ['TFT', 'RobustTFT']:
            assert model_name in results['models']
            model_results = results['models'][model_name]
            
            assert 'training' in model_results
            assert 'inference' in model_results
            assert 'robustness' in model_results
            
        # Check SPO benchmark results
        spo_results = results['spo_benchmark']
        assert 'batch_sizes' in spo_results
        assert 'results' in spo_results
        assert 'avg_speedup' in spo_results
        
        # Check that results file was created
        results_file = Path(self.temp_dir) / "comprehensive_benchmark_results.json"
        assert results_file.exists()
        
        # Check that results file contains valid JSON
        with open(results_file, 'r') as f:
            loaded_results = json.load(f)
        assert 'benchmark_info' in loaded_results
        assert 'models' in loaded_results
        assert 'spo_benchmark' in loaded_results
        
    def test_generate_benchmark_report(self):
        """Test benchmark report generation."""
        # Create mock results
        results = {
            'benchmark_info': {
                'n_samples': 100,
                'horizon': 24,
                'n_epochs': 10,
                'n_runs': 10,
                'device': 'cpu'
            },
            'models': {
                'TFT': {
                    'training': {
                        'model_name': 'TFT',
                        'total_time': 100.0,
                        'avg_time_per_epoch': 10.0,
                        'final_val_loss': 0.1,
                        'avg_memory_usage': 100.0
                    },
                    'inference': {
                        'model_name': 'TFT',
                        'avg_inference_time': 0.1,
                        'avg_memory_usage': 10.0
                    },
                    'robustness': {
                        'adversarial_attacks': {
                            'fgsm': {'degradation': 0.1},
                            'pgd': {'degradation': 0.2}
                        },
                        'noise_robustness': {
                            'noise_0.01': {'crps': 0.1},
                            'noise_0.1': {'crps': 0.2}
                        },
                        'distribution_shift': {
                            'mean': {'crps': 0.1},
                            'std': {'crps': 0.2}
                        }
                    }
                },
                'RobustTFT': {
                    'training': {
                        'model_name': 'RobustTFT',
                        'total_time': 120.0,
                        'avg_time_per_epoch': 12.0,
                        'final_val_loss': 0.09,
                        'avg_memory_usage': 120.0
                    },
                    'inference': {
                        'model_name': 'RobustTFT',
                        'avg_inference_time': 0.12,
                        'avg_memory_usage': 12.0
                    },
                    'robustness': {
                        'adversarial_attacks': {
                            'fgsm': {'degradation': 0.05},
                            'pgd': {'degradation': 0.1}
                        },
                        'noise_robustness': {
                            'noise_0.01': {'crps': 0.09},
                            'noise_0.1': {'crps': 0.18}
                        },
                        'distribution_shift': {
                            'mean': {'crps': 0.09},
                            'std': {'crps': 0.18}
                        }
                    }
                }
            },
            'spo_benchmark': {
                'batch_sizes': [8, 16, 32],
                'results': {
                    8: {'speedup': 1.5},
                    16: {'speedup': 2.0},
                    32: {'speedup': 2.5}
                },
                'avg_speedup': 2.0,
                'max_speedup': 2.5,
                'min_speedup': 1.5
            }
        }
        
        # Generate report
        report = self.runner.generate_benchmark_report(results)
        
        # Check report structure
        assert isinstance(report, str)
        assert 'COMPREHENSIVE BENCHMARK REPORT' in report
        assert 'BENCHMARK CONFIGURATION' in report
        assert 'MODEL COMPARISON' in report
        assert 'Training Performance' in report
        assert 'Inference Performance' in report
        assert 'SPO PERFORMANCE IMPROVEMENTS' in report
        assert 'ROBUSTNESS ANALYSIS' in report
        assert 'RECOMMENDATIONS' in report
        
        # Check that report file was created
        report_file = Path(self.temp_dir) / "benchmark_report.txt"
        assert report_file.exists()
        
        # Check that report file contains the report
        with open(report_file, 'r') as f:
            loaded_report = f.read()
        assert loaded_report == report
        
    def test_integration_with_models(self):
        """Test integration with different model types."""
        # Create test data
        data = self.runner.create_test_data(n_samples=50)
        
        # Test with different model types
        for model_config in self.models_config:
            model_class = model_config['class']
            model_params = model_config['params']
            
            # Create model
            model = model_class(**model_params)
            
            # Test training benchmark
            training_results = self.runner.benchmark_model_training(
                model_class,
                model_params,
                data['train_loader'],
                data['val_loader'],
                n_epochs=1  # Single epoch for testing
            )
            
            # Test inference benchmark
            inference_results = self.runner.benchmark_model_inference(
                model,
                data['test_loader'],
                n_runs=1  # Single run for testing
            )
            
            # Test robustness benchmark
            robustness_results = self.runner.benchmark_robustness(
                model,
                data['test_loader'],
                n_samples=10
            )
            
            # Check that all benchmarks completed successfully
            assert training_results is not None
            assert inference_results is not None
            assert robustness_results is not None
            
            # Check model-specific results
            assert training_results['model_name'] == model_config['name']
            assert inference_results['model_name'] == model_config['name']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])