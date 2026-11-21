from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from datetime import datetime
import logging

from src.optimization.ems import BatterySpec, solve_ems
from src.optimization.spo import spo_plus_loss_torch
from src.forecasting.metrics import evaluate_forecast, crps_from_scenarios


class MLflowExperimentTracker:
    """
    MLflow integration for experiment tracking and model management.
    
    This class provides:
    - Automatic experiment tracking
    - Model versioning
    - Hyperparameter logging
    - Metrics tracking
    - Artifact management
    - Model registry integration
    """
    
    def __init__(self, experiment_name: str = "EMS_Experiments",
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None):
        """
        Initialize MLflow experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
        # Set or create experiment
        self.experiment_id = mlflow.set_experiment(experiment_name)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def start_run(self, run_name: Optional[str] = None,
                 tags: Optional[Dict[str, str]] = None) -> None:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags to attach to the run
        """
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        self.logger.info(f"Started MLflow run: {self.run.info.run_id}")
        
    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        self.logger.info("Ended MLflow run")
        
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
        self.logger.info(f"Logged {len(params)} parameters")
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number for the metrics
        """
        mlflow.log_metrics(metrics, step=step)
        self.logger.info(f"Logged {len(metrics)} metrics at step {step}")
        
    def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log artifacts to MLflow.
        
        Args:
            local_path: Local path to artifacts
            artifact_path: Artifact path in MLflow
        """
        mlflow.log_artifacts(local_path, artifact_path)
        self.logger.info(f"Logged artifacts from {local_path}")
        
    def log_model(self, model: nn.Module, artifact_path: str,
                 conda_env: Optional[Dict] = None,
                 code_paths: Optional[List[str]] = None) -> None:
        """
        Log a PyTorch model to MLflow.
        
        Args:
            model: PyTorch model to log
            artifact_path: Path to save the model
            conda_env: Conda environment specification
            code_paths: Paths to additional code files
        """
        mlflow.pytorch.log_model(
            model, 
            artifact_path,
            conda_env=conda_env,
            code_paths=code_paths
        )
        self.logger.info(f"Logged model to {artifact_path}")
        
    def log_json_config(self, config: Dict[str, Any], artifact_path: str) -> None:
        """
        Log a JSON configuration file.
        
        Args:
            config: Configuration dictionary
            artifact_path: Path to save the config
        """
        config_path = Path(artifact_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        mlflow.log_artifact(str(config_path), artifact_path)
        self.logger.info(f"Logged JSON config to {artifact_path}")
        
    def log_training_results(self, train_metrics: Dict[str, float],
                          val_metrics: Dict[str, float],
                          test_metrics: Dict[str, float],
                          model: nn.Module,
                          config: Dict[str, Any]) -> None:
        """
        Log comprehensive training results.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            test_metrics: Test metrics
            model: Trained model
            config: Configuration dictionary
        """
        # Log all metrics
        all_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        all_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        
        self.log_metrics(all_metrics)
        
        # Log model
        self.log_model(model, "model")
        
        # Log config
        self.log_json_config(config, "config.json")
        
    def log_ems_evaluation(self, predictions: np.ndarray, true_values: np.ndarray,
                          price_hat: np.ndarray, price_true: np.ndarray,
                          scaler_mean: np.ndarray, scaler_std: np.ndarray,
                          battery_spec: BatterySpec) -> Dict[str, float]:
        """
        Log EMS evaluation results.
        
        Args:
            predictions: Model predictions
            true_values: True values
            price_hat: Predicted prices
            price_true: True prices
            scaler_mean: Scaler mean values
            scaler_std: Scaler std values
            battery_spec: Battery specifications
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert to real scale
        y_real = true_values * scaler_std + scaler_mean
        med = predictions[:, :, :, 2] * scaler_std + scaler_mean
        net_hat = med[:, :, 0]
        price_hat_real = price_hat * scaler_std[1] + scaler_mean[1]
        price_true_real = price_true * scaler_std[1] + scaler_mean[1]
        
        # Evaluate forecasts
        net_metrics = evaluate_forecast(y_real[:, :, 0], predictions[:, :, 0, :])
        price_metrics = evaluate_forecast(y_real[:, :, 1], predictions[:, :, 1, :])
        
        # Evaluate operational performance
        cost_pred, cost_true, cost_oracle = [], [], []
        step = max(1, len(y_real) // 240)
        
        for i in range(0, len(y_real), step):
            sol = solve_ems(net_hat[i], price_hat_real[i], battery_spec)
            if sol["status"] not in ("optimal", "optimal_inaccurate"):
                continue
                
            sched = sol["discharge"] - sol["charge"]
            cp_ = self._evaluate_schedule(sched, y_real[i, :, 0], price_hat_real[i], battery_spec)
            co = self._evaluate_schedule(sched, y_real[i, :, 0], price_true_real[i], battery_spec)
            ref = solve_ems(y_real[i, :, 0], price_true_real[i], battery_spec)
            
            cost_pred.append(cp_)
            cost_true.append(co)
            cost_oracle.append(ref["cost"] if ref["status"] in ("optimal", "optimal_inaccurate") else np.nan)
        
        cost_pred = np.array(cost_pred)
        cost_true = np.array(cost_true)
        cost_oracle = np.array(cost_oracle)
        valid = ~np.isnan(cost_oracle)
        
        # Calculate metrics
        operational_metrics = {
            "cost_pred": float(np.mean(cost_pred)),
            "cost_true": float(np.mean(cost_true)),
            "cost_oracle": float(np.mean(cost_oracle[valid])),
            "regret": float(np.mean(cost_true[valid] - cost_oracle[valid])),
            "regret_pct": float(np.mean((cost_true[valid] - cost_oracle[valid]) / 
                                       (np.abs(cost_oracle[valid]) + 1e-9) * 100)),
            "cost_std": float(np.std(cost_true)),
            "n_windows": int(valid.sum())
        }
        
        # Log all metrics
        all_metrics = {
            **net_metrics,
            **price_metrics,
            **operational_metrics
        }
        
        self.log_metrics(all_metrics)
        
        return all_metrics
        
    def _evaluate_schedule(self, schedule: np.ndarray, net_load: np.ndarray,
                          price: np.ndarray, battery_spec: BatterySpec) -> float:
        """
        Evaluate a battery schedule.
        
        Args:
            schedule: Battery schedule
            net_load: Net load profile
            price: Price profile
            battery_spec: Battery specifications
            
        Returns:
            Total cost
        """
        cost = 0.0
        energy = battery_spec.e0
        
        for t in range(len(schedule)):
            # Apply battery constraints
            discharge = min(max(schedule[t], 0), battery_spec.p_dis)
            charge = min(max(-schedule[t], 0), battery_spec.p_ch)
            
            # Update energy level
            energy += discharge - charge
            energy = max(energy, battery_spec.e_min * battery_spec.e_max)
            energy = min(energy, battery_spec.e_max)
            
            # Calculate cost
            net = net_load[t] + discharge - charge
            cost += net * price[t]
            
        return cost
        
    def log_hyperparameter_search(self, param_grid: Dict[str, List[Any]],
                                results: List[Dict[str, Any]]) -> None:
        """
        Log hyperparameter search results.
        
        Args:
            param_grid: Parameter grid
            results: List of result dictionaries
        """
        # Log parameter grid
        self.log_json_config({"param_grid": param_grid}, "param_grid.json")
        
        # Log results
        results_df = pd.DataFrame(results)
        results_path = Path("hyperparameter_search_results.csv")
        results_df.to_csv(results_path, index=False)
        self.log_artifacts(str(results_path), "hyperparameter_search")
        
        # Log best parameters
        if results:
            best_result = max(results, key=lambda x: x.get('score', 0))
            self.log_params(best_result['params'])
            self.log_metrics({"best_score": best_result['score']})
            
    def log_dataset_info(self, dataset_info: Dict[str, Any]) -> None:
        """
        Log dataset information.
        
        Args:
            dataset_info: Dictionary with dataset information
        """
        self.log_json_config(dataset_info, "dataset_info.json")
        
    def log_model_comparison(self, model_results: Dict[str, Dict[str, float]]) -> None:
        """
        Log model comparison results.
        
        Args:
            model_results: Dictionary with model results
        """
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(model_results).T
        comparison_path = Path("model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=True)
        self.log_artifacts(str(comparison_path), "model_comparison")
        
        # Log best model
        if model_results:
            best_model = min(model_results.items(), key=lambda x: x[1].get('regret', float('inf')))
            self.log_metrics({"best_model": best_model[0], "best_regret": best_model[1]['regret']})
            
    def create_model_registry_entry(self, model_name: str, model_uri: str,
                                  description: str = "",
                                  tags: Optional[Dict[str, str]] = None) -> None:
        """
        Create an entry in the model registry.
        
        Args:
            model_name: Name of the model
            model_uri: URI of the model
            description: Model description
            tags: Model tags
        """
        if self.registry_uri:
            mlflow.register_model(model_uri, model_name, description=description, tags=tags)
            self.logger.info(f"Registered model {model_name} at {model_uri}")
            
    def load_model(self, model_uri: str) -> nn.Module:
        """
        Load a model from MLflow.
        
        Args:
            model_uri: URI of the model
            
        Returns:
            Loaded PyTorch model
        """
        model = mlflow.pytorch.load_model(model_uri)
        self.logger.info(f"Loaded model from {model_uri}")
        return model
        
    def get_run_history(self, run_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get run history for an experiment.
        
        Args:
            run_name: Optional run name filter
            
        Returns:
            List of run dictionaries
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return []
            
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_names=[run_name] if run_name else None,
            filter_string="",
            max_results=1000,
            order_by=["start_time DESC"]
        )
        
        return runs.to_dict('records')
        
    def get_best_run(self, metric_name: str, ascending: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.
        
        Args:
            metric_name: Name of the metric
            ascending: Whether to sort in ascending order
            
        Returns:
            Best run dictionary or None
        """
        runs = self.get_run_history()
        if not runs:
            return None
            
        # Sort runs by metric
        sorted_runs = sorted(
            runs,
            key=lambda x: x.get(f'metrics.{metric_name}', float('-inf' if ascending else float('inf'))),
            reverse=not ascending
        )
        
        return sorted_runs[0] if sorted_runs else None


def create_mlflow_experiment_pipeline(experiment_name: str = "EMS_Experiments",
                                    tracking_uri: Optional[str] = None,
                                    registry_uri: Optional[str] = None) -> MLflowExperimentTracker:
    """
    Create an MLflow experiment tracker with standard configuration.
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking server URI
        registry_uri: MLflow model registry URI
        
    Returns:
        Configured MLflow experiment tracker
    """
    # Create tracker
    tracker = MLflowExperimentTracker(experiment_name, tracking_uri, registry_uri)
    
    # Log environment information
    import platform
    import torch
    import mlflow
    
    env_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "mlflow_version": mlflow.__version__,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    
    tracker.log_params(env_info)
    
    return tracker


def run_mlflow_experiment(config: Dict[str, Any],
                         train_func: callable,
                         evaluate_func: callable,
                         experiment_name: str = "EMS_Experiments",
                         tracking_uri: Optional[str] = None,
                         registry_uri: Optional[str] = None) -> Dict[str, Any]:
    """
    Run an experiment with MLflow tracking.
    
    Args:
        config: Configuration dictionary
        train_func: Training function
        evaluate_func: Evaluation function
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking server URI
        registry_uri: MLflow model registry URI
        
    Returns:
        Dictionary with experiment results
    """
    # Create experiment tracker
    tracker = create_mlflow_experiment_pipeline(experiment_name, tracking_uri, registry_uri)
    
    try:
        # Start run
        tracker.start_run(run_name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Log configuration
        tracker.log_params(config)
        
        # Run training
        train_results = train_func(config)
        
        # Log training results
        tracker.log_training_results(
            train_results['train_metrics'],
            train_results['val_metrics'],
            train_results['test_metrics'],
            train_results['model'],
            config
        )
        
        # Run evaluation
        eval_results = evaluate_func(train_results['model'], config)
        
        # Log evaluation results
        tracker.log_metrics(eval_results)
        
        # Return results
        return {
            'train_results': train_results,
            'eval_results': eval_results,
            'run_id': tracker.run.info.run_id
        }
        
    except Exception as e:
        tracker.logger.error(f"Experiment failed: {str(e)}")
        raise
    finally:
        tracker.end_run()