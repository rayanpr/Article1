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

from src.optimization.mlflow_integration import (
    MLflowExperimentTracker,
    create_mlflow_experiment_pipeline,
    run_mlflow_experiment
)
from src.optimization.ems import BatterySpec
from src.forecasting.tft import TFTQuantile


class TestMLflowExperimentTracker:
    """Test suite for MLflowExperimentTracker."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for MLflow tracking
        self.temp_dir = tempfile.mkdtemp()
        self.tracking_uri = f"file://{self.temp_dir}"
        
        # Create experiment tracker
        self.tracker = MLflowExperimentTracker(
            experiment_name="Test_Experiment",
            tracking_uri=self.tracking_uri
        )
        
        # Create test data
        self.test_params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'hidden_size': 64
        }
        
        self.test_metrics = {
            'train_loss': 0.1,
            'val_loss': 0.2,
            'test_loss': 0.15
        }
        
        # Create test model
        self.test_model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Create test config
        self.test_config = {
            'model_type': 'TFT',
            'horizon': 24,
            'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9]
        }
        
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_init(self):
        """Test MLflowExperimentTracker initialization."""
        assert self.tracker.experiment_name == "Test_Experiment"
        assert self.tracker.tracking_uri == self.tracking_uri
        assert self.tracker.experiment_id is not None
        
    def test_start_end_run(self):
        """Test starting and ending MLflow runs."""
        # Start run
        self.tracker.start_run(run_name="test_run")
        assert hasattr(self.tracker, 'run')
        assert self.tracker.run is not None
        
        # End run
        self.tracker.end_run()
        # Should not raise any exceptions
        
    def test_log_params(self):
        """Test parameter logging."""
        with patch('mlflow.log_params') as mock_log_params:
            self.tracker.log_params(self.test_params)
            mock_log_params.assert_called_once_with(self.test_params)
            
    def test_log_metrics(self):
        """Test metric logging."""
        with patch('mlflow.log_metrics') as mock_log_metrics:
            self.tracker.log_metrics(self.test_metrics, step=1)
            mock_log_metrics.assert_called_once_with(self.test_metrics, step=1)
            
    def test_log_artifacts(self):
        """Test artifact logging."""
        with patch('mlflow.log_artifacts') as mock_log_artifacts:
            self.tracker.log_artifacts("test_path", "artifact_path")
            mock_log_artifacts.assert_called_once_with("test_path", "artifact_path")
            
    def test_log_model(self):
        """Test model logging."""
        with patch('mlflow.pytorch.log_model') as mock_log_model:
            self.tracker.log_model(self.test_model, "model_path")
            mock_log_model.assert_called_once()
            
    def test_log_json_config(self):
        """Test JSON config logging."""
        with patch('mlflow.log_artifact') as mock_log_artifact:
            self.tracker.log_json_config(self.test_config, "config.json")
            mock_log_artifact.assert_called_once()
            
            # Check that config file was created
            config_path = Path(self.temp_dir) / "config.json"
            assert config_path.exists()
            
            # Check config content
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            assert loaded_config == self.test_config
            
    def test_log_training_results(self):
        """Test training results logging."""
        train_metrics = {'train_loss': 0.1}
        val_metrics = {'val_loss': 0.2}
        test_metrics = {'test_loss': 0.15}
        
        with patch.object(self.tracker, 'log_metrics') as mock_log_metrics, \
             patch.object(self.tracker, 'log_model') as mock_log_model, \
             patch.object(self.tracker, 'log_json_config') as mock_log_config:
             
            self.tracker.log_training_results(
                train_metrics, val_metrics, test_metrics,
                self.test_model, self.test_config
            )
            
            # Check that metrics were logged
            expected_metrics = {
                'train_train_loss': 0.1,
                'val_val_loss': 0.2,
                'test_test_loss': 0.15
            }
            mock_log_metrics.assert_called_once_with(expected_metrics)
            
            # Check that model was logged
            mock_log_model.assert_called_once()
            
            # Check that config was logged
            mock_log_config.assert_called_once_with(self.test_config, "config.json")
            
    def test_log_ems_evaluation(self):
        """Test EMS evaluation logging."""
        # Create test data
        predictions = np.random.randn(10, 24, 2, 5)
        true_values = np.random.randn(10, 24, 2)
        price_hat = np.random.randn(10, 24)
        price_true = np.random.randn(10, 24)
        scaler_mean = np.array([0, 0])
        scaler_std = np.array([1, 1])
        battery_spec = BatterySpec()
        
        with patch.object(self.tracker, 'log_metrics') as mock_log_metrics:
            metrics = self.tracker.log_ems_evaluation(
                predictions, true_values, price_hat, price_true,
                scaler_mean, scaler_std, battery_spec
            )
            
            # Check that metrics were logged
            mock_log_metrics.assert_called_once()
            
            # Check that metrics dictionary is returned
            assert isinstance(metrics, dict)
            assert 'pinball_mean' in metrics
            assert 'crps' in metrics
            assert 'regret' in metrics
            assert 'regret_pct' in metrics
            
    def test_log_hyperparameter_search(self):
        """Test hyperparameter search logging."""
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [32, 64, 128]
        }
        
        results = [
            {'params': {'learning_rate': 0.001, 'batch_size': 32}, 'score': 0.9},
            {'params': {'learning_rate': 0.01, 'batch_size': 64}, 'score': 0.95},
            {'params': {'learning_rate': 0.1, 'batch_size': 128}, 'score': 0.85}
        ]
        
        with patch.object(self.tracker, 'log_json_config') as mock_log_config, \
             patch.object(self.tracker, 'log_artifacts') as mock_log_artifacts, \
             patch.object(self.tracker, 'log_params') as mock_log_params, \
             patch.object(self.tracker, 'log_metrics') as mock_log_metrics:
             
            self.tracker.log_hyperparameter_search(param_grid, results)
            
            # Check that parameter grid was logged
            mock_log_config.assert_called_once_with({"param_grid": param_grid}, "param_grid.json")
            
            # Check that results were logged
            mock_log_artifacts.assert_called_once()
            
            # Check that best parameters were logged
            mock_log_params.assert_called_once_with({'learning_rate': 0.01, 'batch_size': 64})
            mock_log_metrics.assert_called_once_with({'best_score': 0.95})
            
    def test_log_dataset_info(self):
        """Test dataset info logging."""
        dataset_info = {
            'name': 'CityLearn 2022',
            'n_samples': 8760,
            'n_features': 10,
            'n_targets': 2
        }
        
        with patch.object(self.tracker, 'log_json_config') as mock_log_config:
            self.tracker.log_dataset_info(dataset_info)
            mock_log_config.assert_called_once_with(dataset_info, "dataset_info.json")
            
    def test_log_model_comparison(self):
        """Test model comparison logging."""
        model_results = {
            'TFT': {'crps': 0.1, 'regret': 0.05},
            'XGBoost': {'crps': 0.15, 'regret': 0.08},
            'LightGBM': {'crps': 0.12, 'regret': 0.06}
        }
        
        with patch.object(self.tracker, 'log_artifacts') as mock_log_artifacts, \
             patch.object(self.tracker, 'log_metrics') as mock_log_metrics:
             
            self.tracker.log_model_comparison(model_results)
            
            # Check that comparison was logged
            mock_log_artifacts.assert_called_once()
            
            # Check that best model was logged
            mock_log_metrics.assert_called_once_with({
                'best_model': 'TFT',
                'best_regret': 0.05
            })
            
    def test_create_model_registry_entry(self):
        """Test model registry entry creation."""
        model_name = "test_model"
        model_uri = "runs:/123456/test_model"
        description = "Test model description"
        tags = {"version": "1.0", "environment": "test"}
        
        with patch('mlflow.register_model') as mock_register:
            self.tracker.create_model_registry_entry(
                model_name, model_uri, description, tags
            )
            mock_register.assert_called_once_with(model_uri, model_name, description=description, tags=tags)
            
    def test_load_model(self):
        """Test model loading."""
        model_uri = "runs:/123456/test_model"
        
        with patch('mlflow.pytorch.load_model') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            loaded_model = self.tracker.load_model(model_uri)
            
            mock_load.assert_called_once_with(model_uri)
            assert loaded_model == mock_model
            
    def test_get_run_history(self):
        """Test getting run history."""
        with patch('mlflow.search_runs') as mock_search:
            mock_search.return_value = MagicMock()
            mock_search.return_value.to_dict.return_value = [{'run_id': '123'}]
            
            runs = self.tracker.get_run_history()
            
            mock_search.assert_called_once()
            assert runs == [{'run_id': '123'}]
            
    def test_get_best_run(self):
        """Test getting best run."""
        runs = [
            {'metrics.test_loss': 0.1},
            {'metrics.test_loss': 0.2},
            {'metrics.test_loss': 0.05}
        ]
        
        with patch.object(self.tracker, 'get_run_history') as mock_get_history:
            mock_get_history.return_value = runs
            
            best_run = self.tracker.get_best_run('test_loss', ascending=True)
            
            assert best_run == {'metrics.test_loss': 0.05}


class TestCreateMLflowExperimentPipeline:
    """Test suite for create_mlflow_experiment_pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracking_uri = f"file://{self.temp_dir}"
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def test_create_pipeline(self):
        """Test creating MLflow experiment pipeline."""
        tracker = create_mlflow_experiment_pipeline(
            experiment_name="Test_Pipeline",
            tracking_uri=self.tracking_uri
        )
        
        assert tracker.experiment_name == "Test_Pipeline"
        assert tracker.tracking_uri == self.tracking_uri
        assert tracker.experiment_id is not None
        
        # Check that environment info was logged
        with patch('mlflow.log_params') as mock_log_params:
            # This should have been called during initialization
            assert mock_log_params.call_count > 0


class TestRunMLflowExperiment:
    """Test suite for run_mlflow_experiment."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracking_uri = f"file://{self.temp_dir}"
        
        # Mock training and evaluation functions
        self.mock_train_func = MagicMock()
        self.mock_train_func.return_value = {
            'train_metrics': {'loss': 0.1},
            'val_metrics': {'loss': 0.2},
            'test_metrics': {'loss': 0.15},
            'model': nn.Sequential(nn.Linear(10, 1))
        }
        
        self.mock_eval_func = MagicMock()
        self.mock_eval_func.return_value = {'eval_metric': 0.05}
        
        self.test_config = {
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def test_run_experiment(self):
        """Test running an experiment."""
        results = run_mlflow_experiment(
            config=self.test_config,
            train_func=self.mock_train_func,
            evaluate_func=self.mock_eval_func,
            experiment_name="Test_Run_Experiment",
            tracking_uri=self.tracking_uri
        )
        
        # Check results structure
        assert 'train_results' in results
        assert 'eval_results' in results
        assert 'run_id' in results
        
        # Check that functions were called
        self.mock_train_func.assert_called_once_with(self.test_config)
        self.mock_eval_func.assert_called_once()
        
    def test_run_experiment_with_exception(self):
        """Test running an experiment with exception."""
        # Mock training function that raises exception
        def failing_train_func(config):
            raise ValueError("Training failed")
            
        with pytest.raises(ValueError, match="Training failed"):
            run_mlflow_experiment(
                config=self.test_config,
                train_func=failing_train_func,
                evaluate_func=self.mock_eval_func,
                experiment_name="Test_Failed_Experiment",
                tracking_uri=self.tracking_uri
            )


class TestIntegration:
    """Integration tests for MLflow integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracking_uri = f"file://{self.temp_dir}"
        
        # Create test model
        self.model = TFTQuantile(n_features=10, horizon=24, n_targets=2, hidden=64)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
        
    def test_integration_with_tft_model(self):
        """Test integration with TFT model."""
        # Create experiment tracker
        tracker = MLflowExperimentTracker(
            experiment_name="TFT_Integration_Test",
            tracking_uri=self.tracking_uri
        )
        
        # Start run
        tracker.start_run(run_name="tft_test")
        
        # Log model
        tracker.log_model(self.model, "tft_model")
        
        # Log test metrics
        test_metrics = {
            'test_crps': 0.1,
            'test_regret': 0.05
        }
        tracker.log_metrics(test_metrics)
        
        # End run
        tracker.end_run()
        
        # Check that run was created
        runs = tracker.get_run_history()
        assert len(runs) > 0
        
    def test_full_experiment_workflow(self):
        """Test full experiment workflow."""
        # Create experiment tracker
        tracker = MLflowExperimentTracker(
            experiment_name="Full_Workflow_Test",
            tracking_uri=self.tracking_uri
        )
        
        # Start run
        tracker.start_run(run_name="full_workflow")
        
        # Log configuration
        config = {
            'model_type': 'TFT',
            'horizon': 24,
            'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
            'learning_rate': 0.001,
            'batch_size': 32
        }
        tracker.log_params(config)
        
        # Log model
        tracker.log_model(self.model, "model")
        
        # Log training results
        train_metrics = {'train_loss': 0.1}
        val_metrics = {'val_loss': 0.2}
        test_metrics = {'test_loss': 0.15}
        tracker.log_training_results(
            train_metrics, val_metrics, test_metrics,
            self.model, config
        )
        
        # Log evaluation results
        eval_metrics = {'eval_crps': 0.1, 'eval_regret': 0.05}
        tracker.log_metrics(eval_metrics)
        
        # Log dataset info
        dataset_info = {
            'name': 'CityLearn 2022',
            'n_samples': 8760,
            'n_features': 10,
            'n_targets': 2
        }
        tracker.log_dataset_info(dataset_info)
        
        # End run
        tracker.end_run()
        
        # Check that run was created and has all logged data
        runs = tracker.get_run_history()
        assert len(runs) > 0
        
        # Check that we can retrieve the best run
        best_run = tracker.get_best_run('eval_loss' if 'eval_loss' in runs[0] else 'test_loss')
        assert best_run is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])