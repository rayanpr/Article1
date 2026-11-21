import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import json

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.robustness_analysis import (
    RobustnessAnalyzer,
    run_comprehensive_robustness_analysis
)
from src.optimization.ems import BatterySpec
from src.forecasting.tft import TFTQuantile


class TestRobustnessAnalyzer:
    """Test suite for RobustnessAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = RobustnessAnalyzer()
        self.batt = BatterySpec()
        
        # Create test data
        self.n_samples = 100
        self.predictions = np.random.randn(self.n_samples)
        self.true_values = np.random.randn(self.n_samples)
        
        # Create test model
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Create test tensors
        self.X = torch.randn(10, 24, 10)
        self.y = torch.randn(10, 24, 1)
        
    def test_init(self):
        """Test RobustnessAnalyzer initialization."""
        assert self.analyzer.battery_spec is not None
        
    def test_analyze_distribution_shift(self):
        """Test distribution shift analysis."""
        results = self.analyzer.analyze_distribution_shift(
            self.predictions, self.true_values
        )
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'mean' in results
        assert 'std' in results
        assert 'seasonal' in results
        
        # Check result values
        for shift_type, metrics in results.items():
            assert 'crps' in metrics
            assert 'shift_magnitude' in metrics
            assert 'relative_error' in metrics
            assert isinstance(metrics['crps'], float)
            assert isinstance(metrics['shift_magnitude'], float)
            assert isinstance(metrics['relative_error'], float)
            
    def test_analyze_adversarial_attacks(self):
        """Test adversarial attack analysis."""
        results = self.analyzer.analyze_adversarial_attacks(
            self.model, self.X, self.y
        )
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'fgsm' in results
        assert 'pgd' in results
        
        # Check result values
        for attack_type, metrics in results.items():
            assert 'mse_clean' in metrics
            assert 'mse_adv' in metrics
            assert 'degradation' in metrics
            assert 'perturbation_magnitude' in metrics
            assert isinstance(metrics['mse_clean'], float)
            assert isinstance(metrics['mse_adv'], float)
            assert isinstance(metrics['degradation'], float)
            assert isinstance(metrics['perturbation_magnitude'], float)
            
    def test_analyze_noise_robustness(self):
        """Test noise robustness analysis."""
        results = self.analyzer.analyze_noise_robustness(
            self.predictions, self.true_values
        )
        
        # Check results structure
        assert isinstance(results, dict)
        for key in results.keys():
            assert key.startswith('noise_')
            
        # Check result values
        for noise_type, metrics in results.items():
            assert 'crps' in metrics
            assert 'noise_level' in metrics
            assert 'signal_to_noise_ratio' in metrics
            assert isinstance(metrics['crps'], float)
            assert isinstance(metrics['noise_level'], float)
            assert isinstance(metrics['signal_to_noise_ratio'], float)
            
    def test_analyze_parameter_sensitivity(self):
        """Test parameter sensitivity analysis."""
        # Define parameter ranges and base values
        param_ranges = {
            'learning_rate': [0.001, 0.01, 0.1],
            'hidden_size': [32, 64, 128]
        }
        base_params = {
            'learning_rate': 0.01,
            'hidden_size': 64
        }
        
        # Define evaluation function
        def evaluate_func(params):
            return np.random.rand()  # Random evaluation for testing
            
        results = self.analyzer.analyze_parameter_sensitivity(
            param_ranges, base_params, evaluate_func
        )
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'learning_rate' in results
        assert 'hidden_size' in results
        
        # Check result values
        for param, metrics in results.items():
            assert 'values' in metrics
            assert 'performance' in metrics
            assert 'sensitivity' in metrics
            assert 'max_change' in metrics
            assert isinstance(metrics['values'], list)
            assert isinstance(metrics['performance'], list)
            assert isinstance(metrics['sensitivity'], float)
            assert isinstance(metrics['max_change'], float)
            
    def test_detect_outliers(self):
        """Test outlier detection."""
        # Test with synthetic data containing outliers
        normal_data = np.random.randn(90)
        outlier_data = np.random.randn(10) * 5 + 10
        data = np.concatenate([normal_data, outlier_data])
        
        # Test different methods
        for method in ['isolation', 'zscore', 'iqr']:
            outlier_mask, outlier_stats = self.analyzer.detect_outliers(
                data, method=method, contamination=0.1
            )
            
            # Check results
            assert isinstance(outlier_mask, np.ndarray)
            assert isinstance(outlier_stats, dict)
            assert 'n_outliers' in outlier_stats
            assert 'outlier_ratio' in outlier_stats
            assert 'outlier_values' in outlier_stats
            assert 'non_outlier_values' in outlier_stats
            assert isinstance(outlier_stats['n_outliers'], int)
            assert isinstance(outlier_stats['outlier_ratio'], float)
            assert isinstance(outlier_stats['outlier_values'], list)
            assert isinstance(outlier_stats['non_outlier_values'], list)
            
    def test_analyze_scenario_robustness(self):
        """Test scenario robustness analysis."""
        # Create test scenarios
        scenarios = [
            {
                'type': 'normal',
                'net_load': np.random.randn(24) * 100 + 500,
                'price': np.random.randn(24) * 0.1 + 0.2,
                'oracle_cost': 50.0
            },
            {
                'type': 'peak',
                'net_load': np.random.randn(24) * 150 + 700,
                'price': np.random.randn(24) * 0.2 + 0.5,
                'oracle_cost': 80.0
            }
        ]
        
        results = self.analyzer.analyze_scenario_robustness(
            scenarios, self.model, lambda net, price, batt: 
            {'cost': np.random.rand() * 100, 'status': 'optimal'}
        )
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'scenario_0' in results
        assert 'scenario_1' in results
        
        # Check result values
        for scenario, metrics in results.items():
            assert 'scenario_type' in metrics
            assert 'cost' in metrics
            assert 'robustness_score' in metrics
            assert 'solution_status' in metrics
            assert 'net_load_stats' in metrics
            assert 'price_stats' in metrics
            assert isinstance(metrics['scenario_type'], str)
            assert isinstance(metrics['cost'], float)
            assert isinstance(metrics['robustness_score'], float)
            assert isinstance(metrics['solution_status'], str)
            assert isinstance(metrics['net_load_stats'], dict)
            assert isinstance(metrics['price_stats'], dict)
            
    def test_generate_robustness_report(self):
        """Test robustness report generation."""
        # Create mock results
        results = {
            'distribution_shift': {
                'mean': {'crps': 0.1, 'shift_magnitude': 0.05, 'relative_error': 0.1}
            },
            'adversarial_attacks': {
                'fgsm': {'mse_clean': 0.1, 'mse_adv': 0.2, 'degradation': 1.0, 'perturbation_magnitude': 0.1}
            },
            'noise_robustness': {
                'noise_0.01': {'crps': 0.1, 'noise_level': 0.01, 'signal_to_noise_ratio': 10.0}
            },
            'outlier_detection': {
                'isolation': {'n_outliers': 10, 'outlier_ratio': 0.1, 'outlier_values': [1, 2, 3], 'non_outlier_values': [4, 5, 6]}
            },
            'scenario_robustness': {
                'scenario_0': {'scenario_type': 'normal', 'cost': 50.0, 'robustness_score': 0.8, 'solution_status': 'optimal'}
            }
        }
        
        # Generate report
        report = self.analyzer.generate_robustness_report(results)
        
        # Check report structure
        assert isinstance(report, str)
        assert 'ROBUSTNESS ANALYSIS REPORT' in report
        assert 'DISTRIBUTION SHIFT ANALYSIS' in report
        assert 'ADVERSARIAL ATTACK ANALYSIS' in report
        assert 'NOISE ROBUSTNESS ANALYSIS' in report
        assert 'OUTLIER DETECTION' in report
        assert 'SCENARIO ROBUSTNESS ANALYSIS' in report
        assert 'OVERALL SUMMARY' in report
        
    def test_robustness_score_calculation(self):
        """Test robustness score calculation."""
        # Test optimal solution
        solution = {'cost': 50.0, 'constraint_violations': 0}
        scenario = {'oracle_cost': 40.0}
        score = self.analyzer._calculate_robustness_score(solution, scenario)
        assert 0.0 <= score <= 1.0
        
        # Test infeasible solution
        solution = {'cost': np.inf, 'constraint_violations': 0}
        score = self.analyzer._calculate_robustness_score(solution, scenario)
        assert score == 0.0
        
        # Test solution with constraint violations
        solution = {'cost': 50.0, 'constraint_violations': 1}
        score = self.analyzer._calculate_robustness_score(solution, scenario)
        assert score < 0.8  # Should be penalized


class TestComprehensiveRobustnessAnalysis:
    """Test suite for comprehensive robustness analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test model
        self.model = TFTQuantile(n_features=10, horizon=24, n_targets=2, hidden=64)
        
        # Create test data
        self.test_data = {
            'predictions': np.random.randn(100, 24, 2, 5),
            'true_values': np.random.randn(100, 24, 2),
            'X_tensor': torch.randn(10, 24, 10),
            'y_tensor': torch.randn(10, 24, 2, 5),
            'scenarios': [
                {
                    'type': 'normal',
                    'net_load': np.random.randn(24) * 100 + 500,
                    'price': np.random.randn(24) * 0.1 + 0.2,
                    'oracle_cost': 50.0
                },
                {
                    'type': 'peak',
                    'net_load': np.random.randn(24) * 150 + 700,
                    'price': np.random.randn(24) * 0.2 + 0.5,
                    'oracle_cost': 80.0
                }
            ]
        }
        
    def test_run_comprehensive_robustness_analysis(self):
        """Test comprehensive robustness analysis."""
        results = run_comprehensive_robustness_analysis(
            self.model, self.test_data, output_dir="test_robustness_results"
        )
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'distribution_shift' in results
        assert 'adversarial_attacks' in results
        assert 'noise_robustness' in results
        assert 'outlier_detection' in results
        assert 'scenario_robustness' in results
        
        # Check that files were created
        output_path = Path("test_robustness_results")
        assert (output_path / "robustness_report.txt").exists()
        assert (output_path / "robustness_results.json").exists()
        
        # Clean up
        import shutil
        shutil.rmtree(output_path)
        
    def test_integration_with_ems(self):
        """Test integration with EMS."""
        from src.optimization.ems import solve_ems
        
        # Create test data
        net_load = np.random.randn(24) * 100 + 500
        price = np.random.randn(24) * 0.1 + 0.2
        
        # Test EMS solver
        solution = solve_ems(net_load, price, BatterySpec())
        
        # Check solution
        assert isinstance(solution, dict)
        assert 'status' in solution
        assert 'cost' in solution
        assert 'discharge' in solution
        assert 'charge' in solution
        assert isinstance(solution['status'], str)
        assert isinstance(solution['cost'], float)
        assert isinstance(solution['discharge'], np.ndarray)
        assert isinstance(solution['charge'], np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])