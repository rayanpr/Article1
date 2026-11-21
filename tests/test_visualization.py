import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import shutil
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.visualization import EMSVisualizer


class TestEMSVisualizer:
    """Test suite for EMSVisualizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for visualizations
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = EMSVisualizer(self.temp_dir)
        
        # Create test data
        self.n_samples = 100
        self.horizon = 24
        self.timestamps = pd.date_range(start='2022-01-01', periods=self.n_samples, freq='H')
        
        # Create test data for visualizations
        self.actual_values = np.random.randn(self.n_samples) * 10 + 50
        self.predicted_values = np.random.randn(self.n_samples) * 8 + 48
        self.confidence_intervals = np.column_stack([
            self.predicted_values - 5,
            self.predicted_values + 5
        ])
        
        # Create test data for cost comparison
        self.methods = ['TFT', 'XGBoost', 'Oracle']
        self.costs = [1000, 1200, 800]
        self.regrets = [0.1, 0.15, 0.0]
        
        # Create test data for battery visualization
        self.state_of_charge = np.linspace(20, 80, self.n_samples)
        self.charge_schedule = np.random.randn(self.n_samples) * 5 + 2
        self.discharge_schedule = np.random.randn(self.n_samples) * 5 - 2
        
        # Create test data for feature importance
        self.feature_names = [f'feature_{i}' for i in range(20)]
        self.importance_scores = np.random.rand(20)
        
        # Create test data for uncertainty quantification
        self.uncertainty_intervals = np.column_stack([
            np.random.rand(self.n_samples) * 2,
            np.random.rand(self.n_samples) * 2 + 2
        ])
        
        # Create test data for model performance comparison
        self.model_names = ['TFT', 'XGBoost', 'LightGBM']
        self.metrics = {
            'Accuracy': [0.85, 0.82, 0.88],
            'Precision': [0.83, 0.80, 0.86],
            'Recall': [0.87, 0.84, 0.90],
            'F1-Score': [0.85, 0.82, 0.88]
        }
        
        # Create test data for forecast fan chart
        self.quantiles = np.random.rand(self.n_samples, 5)
        
        # Create test data for robustness analysis
        self.robustness_results = {
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
        
    def teardown_method(self):
        """Clean up test fixtures."""
        plt.close('all')  # Close all matplotlib figures
        shutil.rmtree(self.temp_dir)
        
    def test_init(self):
        """Test EMSVisualizer initialization."""
        assert self.visualizer.output_dir == Path(self.temp_dir)
        assert self.visualizer.colors is not None
        
    def test_plot_time_series_with_confidence(self):
        """Test time series with confidence intervals plotting."""
        save_path = self.temp_dir / "time_series.png"
        
        # Test with confidence intervals
        self.visualizer.plot_time_series_with_confidence(
            self.timestamps,
            self.actual_values,
            self.predicted_values,
            self.confidence_intervals,
            save_path=str(save_path)
        )
        
        # Check that plot was saved
        assert save_path.exists()
        
        # Test without confidence intervals
        save_path_no_ci = self.temp_dir / "time_series_no_ci.png"
        self.visualizer.plot_time_series_with_confidence(
            self.timestamps,
            self.actual_values,
            self.predicted_values,
            None,
            save_path=str(save_path_no_ci)
        )
        
        assert save_path_no_ci.exists()
        
    def test_plot_cost_comparison(self):
        """Test cost comparison plotting."""
        save_path = self.temp_dir / "cost_comparison.png"
        
        self.visualizer.plot_cost_comparison(
            self.methods,
            self.costs,
            self.regrets,
            save_path=str(save_path)
        )
        
        # Check that plot was saved
        assert save_path.exists()
        
    def test_plot_battery_state(self):
        """Test battery state plotting."""
        save_path = self.temp_dir / "battery_state.png"
        
        self.visualizer.plot_battery_state(
            self.timestamps,
            self.state_of_charge,
            self.charge_schedule,
            self.discharge_schedule,
            save_path=str(save_path)
        )
        
        # Check that plot was saved
        assert save_path.exists()
        
    def test_plot_feature_importance(self):
        """Test feature importance plotting."""
        save_path = self.temp_dir / "feature_importance.png"
        
        self.visualizer.plot_feature_importance(
            self.feature_names,
            self.importance_scores,
            save_path=str(save_path),
            top_n=10
        )
        
        # Check that plot was saved
        assert save_path.exists()
        
    def test_plot_uncertainty_quantification(self):
        """Test uncertainty quantification plotting."""
        save_path = self.temp_dir / "uncertainty_quantification.png"
        
        self.visualizer.plot_uncertainty_quantification(
            self.actual_values,
            self.predicted_values,
            self.uncertainty_intervals,
            save_path=str(save_path)
        )
        
        # Check that plot was saved
        assert save_path.exists()
        
    def test_plot_model_performance_comparison(self):
        """Test model performance comparison plotting."""
        save_path = self.temp_dir / "model_performance_comparison.png"
        
        self.visualizer.plot_model_performance_comparison(
            self.model_names,
            self.metrics,
            save_path=str(save_path)
        )
        
        # Check that plot was saved
        assert save_path.exists()
        
    def test_plot_forecast_fan_chart(self):
        """Test forecast fan chart plotting."""
        save_path = self.temp_dir / "forecast_fan_chart.png"
        
        self.visualizer.plot_forecast_fan_chart(
            self.timestamps,
            self.quantiles,
            save_path=str(save_path)
        )
        
        # Check that plot was saved
        assert save_path.exists()
        
    def test_plot_robustness_analysis(self):
        """Test robustness analysis plotting."""
        save_path = self.temp_dir / "robustness_analysis.png"
        
        self.visualizer.plot_robustness_analysis(
            self.robustness_results,
            save_path=str(save_path)
        )
        
        # Check that plot was saved
        assert save_path.exists()
        
    def test_create_dashboard(self):
        """Test dashboard creation."""
        save_path = self.temp_dir / "dashboard.png"
        
        # Create mock results
        results = {
            'cost_comparison': {
                'methods': self.methods,
                'costs': self.costs,
                'regrets': self.regrets
            },
            'model_performance': {
                'model_names': self.model_names,
                'Accuracy': [0.85, 0.82, 0.88],
                'Precision': [0.83, 0.80, 0.86],
                'Recall': [0.87, 0.84, 0.90],
                'F1-Score': [0.85, 0.82, 0.88]
            },
            'robustness': {
                'adversarial_attacks': 0.4,
                'noise_robustness': 0.2,
                'distribution_shift': 0.3
            },
            'feature_importance': {
                'features': self.feature_names[:10],
                'importance': self.importance_scores[:10]
            },
            'training_metrics': {
                'train_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
                'val_loss': [0.6, 0.5, 0.4, 0.3, 0.2]
            },
            'inference_metrics': {
                'models': self.model_names,
                'times': [0.1, 0.2, 0.15]
            },
            'memory_metrics': {
                'models': self.model_names,
                'memory': [100, 150, 120]
            }
        }
        
        self.visualizer.create_dashboard(results, str(save_path))
        
        # Check that dashboard was saved
        assert save_path.exists()
        
    def test_integration_with_different_data_types(self):
        """Test integration with different data types."""
        # Test with different data types
        save_path = self.temp_dir / "integration_test.png"
        
        # Test with pandas Series
        actual_series = pd.Series(self.actual_values, index=self.timestamps)
        predicted_series = pd.Series(self.predicted_values, index=self.timestamps)
        
        self.visualizer.plot_time_series_with_confidence(
            self.timestamps,
            actual_series,
            predicted_series,
            save_path=str(save_path)
        )
        
        # Check that plot was saved
        assert save_path.exists()
        
    def test_error_handling(self):
        """Test error handling."""
        # Test with empty data
        empty_timestamps = pd.DatetimeIndex([])
        empty_values = np.array([])
        
        # Should not raise an exception
        try:
            self.visualizer.plot_time_series_with_confidence(
                empty_timestamps,
                empty_values,
                empty_values,
                save_path=None
            )
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")
            
    def test_customization_options(self):
        """Test customization options."""
        save_path = self.temp_dir / "customization_test.png"
        
        # Test with custom title and figsize
        self.visualizer.plot_cost_comparison(
            self.methods,
            self.costs,
            self.regrets,
            title="Custom Title",
            save_path=str(save_path),
            figsize=(10, 8)
        )
        
        # Check that plot was saved
        assert save_path.exists()
        
    def test_multiple_plots(self):
        """Test creating multiple plots."""
        # Create multiple plots
        save_paths = []
        
        # Time series plot
        save_path1 = self.temp_dir / "plot1.png"
        self.visualizer.plot_time_series_with_confidence(
            self.timestamps,
            self.actual_values,
            self.predicted_values,
            save_path=str(save_path1)
        )
        save_paths.append(save_path1)
        
        # Cost comparison plot
        save_path2 = self.temp_dir / "plot2.png"
        self.visualizer.plot_cost_comparison(
            self.methods,
            self.costs,
            self.regrets,
            save_path=str(save_path2)
        )
        save_paths.append(save_path2)
        
        # Battery state plot
        save_path3 = self.temp_dir / "plot3.png"
        self.visualizer.plot_battery_state(
            self.timestamps,
            self.state_of_charge,
            self.charge_schedule,
            self.discharge_schedule,
            save_path=str(save_path3)
        )
        save_paths.append(save_path3)
        
        # Check that all plots were saved
        for path in save_paths:
            assert path.exists()
            
    def test_color_schemes(self):
        """Test color schemes."""
        # Check that colors are defined
        assert 'actual' in self.visualizer.colors
        assert 'predicted' in self.visualizer.colors
        assert 'oracle' in self.visualizer.colors
        assert 'battery_charge' in self.visualizer.colors
        assert 'battery_discharge' in self.visualizer.colors
        
        # Check color values
        assert isinstance(self.visualizer.colors['actual'], str)
        assert isinstance(self.visualizer.colors['predicted'], str)
        
    def test_figure_management(self):
        """Test figure management."""
        # Create multiple figures
        self.visualizer.plot_time_series_with_confidence(
            self.timestamps,
            self.actual_values,
            self.predicted_values,
            save_path=None
        )
        
        self.visualizer.plot_cost_comparison(
            self.methods,
            self.costs,
            self.regrets,
            save_path=None
        )
        
        # Should not have memory leaks
        import gc
        gc.collect()
        
        # Test that figures are properly closed
        assert plt.get_fignums() == []  # No open figures


if __name__ == "__main__":
    pytest.main([__file__, "-v"])