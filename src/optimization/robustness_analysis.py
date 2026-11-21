from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.optimization.ems import BatterySpec, solve_ems, evaluate_schedule
from src.optimization.spo import spo_plus_loss_torch
from src.forecasting.metrics import crps_from_scenarios, evaluate_forecast


class RobustnessAnalyzer:
    """
    Comprehensive robustness analysis for energy management systems.
    
    This class provides methods to analyze:
    - Distribution shift robustness
    - Adversarial attacks
    - Noise robustness
    - Parameter sensitivity
    - Outlier detection
    - Scenario-based robustness
    """
    
    def __init__(self, battery_spec: BatterySpec = None):
        """
        Initialize the robustness analyzer.
        
        Args:
            battery_spec: Battery specifications for EMS
        """
        self.battery_spec = battery_spec or BatterySpec()
        
    def analyze_distribution_shift(self, predictions: np.ndarray, 
                                  true_values: np.ndarray,
                                  shift_types: List[str] = ['mean', 'std', 'seasonal']) -> Dict[str, Dict]:
        """
        Analyze model performance under various distribution shifts.
        
        Args:
            predictions: Model predictions
            true_values: True values
            shift_types: Types of distribution shifts to test
            
        Returns:
            Dictionary with shift analysis results
        """
        results = {}
        
        for shift_type in shift_types:
            if shift_type == 'mean':
                # Shift mean of predictions
                shifted_predictions = predictions + 0.1 * np.std(predictions)
                
            elif shift_type == 'std':
                # Shift standard deviation of predictions
                shifted_predictions = predictions * 1.2
                
            elif shift_type == 'seasonal':
                # Add seasonal pattern
                t = np.arange(len(predictions))
                seasonal_pattern = 0.05 * np.sin(2 * np.pi * t / (24 * 7))  # Weekly pattern
                shifted_predictions = predictions + seasonal_pattern
                
            else:
                continue
                
            # Evaluate performance
            crps = crps_from_scenarios(true_values, shifted_predictions)
            
            results[shift_type] = {
                'crps': float(crps),
                'shift_magnitude': float(np.mean(np.abs(shifted_predictions - predictions))),
                'relative_error': float(np.mean(np.abs(shifted_predictions - true_values)) / 
                                     np.mean(np.abs(true_values)))
            }
            
        return results
    
    def analyze_adversarial_attacks(self, model: nn.Module, 
                                   X: torch.Tensor, y: torch.Tensor,
                                   attack_methods: List[str] = ['fgsm', 'pgd'],
                                   epsilon: float = 0.1) -> Dict[str, Dict]:
        """
        Analyze model performance under adversarial attacks.
        
        Args:
            model: Trained model
            X: Input data
            y: Target data
            attack_methods: List of attack methods
            epsilon: Attack magnitude
            
        Returns:
            Dictionary with adversarial analysis results
        """
        results = {}
        
        model.eval()
        
        for attack_method in attack_methods:
            if attack_method == 'fgsm':
                # Fast Gradient Sign Method
                X_adv = self._fgsm_attack(model, X, y, epsilon)
                
            elif attack_method == 'pgd':
                # Projected Gradient Descent
                X_adv = self._pgd_attack(model, X, y, epsilon, alpha=0.01, steps=10)
                
            else:
                continue
                
            # Evaluate performance
            with torch.no_grad():
                pred_clean = model(X)
                pred_adv = model(X_adv)
                
            # Calculate metrics
            mse_clean = torch.nn.functional.mse_loss(pred_clean, y).item()
            mse_adv = torch.nn.functional.mse_loss(pred_adv, y).item()
            
            results[attack_method] = {
                'mse_clean': mse_clean,
                'mse_adv': mse_adv,
                'degradation': (mse_adv - mse_clean) / mse_clean,
                'perturbation_magnitude': float(torch.norm(X_adv - X).item())
            }
            
        return results
    
    def _fgsm_attack(self, model: nn.Module, X: torch.Tensor, 
                     y: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Implement FGSM attack."""
        X_adv = X.clone().detach()
        X_adv.requires_grad = True
        
        loss = torch.nn.functional.mse_loss(model(X_adv), y)
        loss.backward()
        
        # Calculate perturbation
        perturbation = epsilon * X_adv.grad.sign()
        X_adv = X_adv.detach() + perturbation
        X_adv = torch.clamp(X_adv, 0, 1)  # Assuming input is normalized
        
        return X_adv
    
    def _pgd_attack(self, model: nn.Module, X: torch.Tensor, 
                    y: torch.Tensor, epsilon: float, 
                    alpha: float, steps: int) -> torch.Tensor:
        """Implement PGD attack."""
        X_adv = X.clone().detach()
        
        for _ in range(steps):
            X_adv.requires_grad = True
            
            loss = torch.nn.functional.mse_loss(model(X_adv), y)
            loss.backward()
            
            # Calculate perturbation
            perturbation = alpha * X_adv.grad.sign()
            X_adv = X_adv.detach() + perturbation
            
            # Project back to epsilon-ball
            delta = X_adv - X
            delta = torch.clamp(delta, -epsilon, epsilon)
            X_adv = X + delta
            X_adv = torch.clamp(X_adv, 0, 1)
            
        return X_adv
    
    def analyze_noise_robustness(self, predictions: np.ndarray, 
                                true_values: np.ndarray,
                                noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]) -> Dict[str, Dict]:
        """
        Analyze model robustness to noise.
        
        Args:
            predictions: Model predictions
            true_values: True values
            noise_levels: List of noise levels to test
            
        Returns:
            Dictionary with noise robustness results
        """
        results = {}
        
        for noise_level in noise_levels:
            # Add noise to predictions
            noise = np.random.normal(0, noise_level, predictions.shape)
            noisy_predictions = predictions + noise
            
            # Evaluate performance
            crps = crps_from_scenarios(true_values, noisy_predictions)
            
            results[f'noise_{noise_level}'] = {
                'crps': float(crps),
                'noise_level': noise_level,
                'signal_to_noise_ratio': float(np.std(predictions) / (noise_level + 1e-8))
            }
            
        return results
    
    def analyze_parameter_sensitivity(self, param_ranges: Dict[str, List[float]],
                                  base_params: Dict[str, float],
                                  evaluate_func: callable) -> Dict[str, Dict]:
        """
        Analyze sensitivity to parameter changes.
        
        Args:
            param_ranges: Dictionary of parameter ranges
            base_params: Base parameter values
            evaluate_func: Function to evaluate performance
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        results = {}
        
        for param, values in param_ranges.items():
            sensitivity = []
            
            for value in values:
                # Create modified parameters
                modified_params = base_params.copy()
                modified_params[param] = value
                
                # Evaluate performance
                performance = evaluate_func(modified_params)
                sensitivity.append(performance)
                
            # Calculate sensitivity metrics
            sensitivity = np.array(sensitivity)
            base_performance = evaluate_func(base_params)
            
            results[param] = {
                'values': values,
                'performance': sensitivity.tolist(),
                'sensitivity': float(np.std(sensitivity) / (base_performance + 1e-8)),
                'max_change': float(np.max(np.abs(sensitivity - base_performance)) / base_performance)
            }
            
        return results
    
    def detect_outliers(self, data: np.ndarray, method: str = 'isolation',
                       contamination: float = 0.1) -> Tuple[np.ndarray, Dict]:
        """
        Detect outliers in the data.
        
        Args:
            data: Input data
            method: Outlier detection method
            contamination: Expected proportion of outliers
            
        Returns:
            Tuple of (outlier_mask, outlier_stats)
        """
        if method == 'isolation':
            from sklearn.ensemble import IsolationForest
            
            clf = IsolationForest(contamination=contamination, random_state=42)
            outlier_mask = clf.fit_predict(data) == -1
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outlier_mask = z_scores > 3
            
        elif method == 'iqr':
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            outlier_mask = (data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
            
        # Calculate outlier statistics
        outlier_stats = {
            'n_outliers': int(np.sum(outlier_mask)),
            'outlier_ratio': float(np.mean(outlier_mask)),
            'outlier_values': data[outlier_mask].tolist(),
            'non_outlier_values': data[~outlier_mask].tolist()
        }
        
        return outlier_mask, outlier_stats
    
    def analyze_scenario_robustness(self, scenarios: List[Dict],
                                   prediction_model: nn.Module,
                                   ems_solver: callable) -> Dict[str, Dict]:
        """
        Analyze robustness across different scenarios.
        
        Args:
            scenarios: List of scenario dictionaries
            prediction_model: Trained prediction model
            ems_solver: EMS solver function
            
        Returns:
            Dictionary with scenario robustness results
        """
        results = {}
        
        for i, scenario in enumerate(scenarios):
            # Extract scenario data
            net_load = scenario['net_load']
            price = scenario['price']
            
            # Make predictions
            with torch.no_grad():
                X = torch.tensor(np.column_stack([net_load, price]), dtype=torch.float32).unsqueeze(0)
                pred = prediction_model(X)
                
            # Solve EMS
            solution = ems_solver(net_load, price, self.battery_spec)
            
            # Calculate robustness metrics
            if solution['status'] == 'optimal':
                cost = solution['cost']
                robustness_score = self._calculate_robustness_score(solution, scenario)
            else:
                cost = np.inf
                robustness_score = 0.0
                
            results[f'scenario_{i}'] = {
                'scenario_type': scenario.get('type', 'unknown'),
                'cost': float(cost),
                'robustness_score': robustness_score,
                'solution_status': solution['status'],
                'net_load_stats': {
                    'mean': float(np.mean(net_load)),
                    'std': float(np.std(net_load)),
                    'max': float(np.max(net_load)),
                    'min': float(np.min(net_load))
                },
                'price_stats': {
                    'mean': float(np.mean(price)),
                    'std': float(np.std(price)),
                    'max': float(np.max(price)),
                    'min': float(np.min(price))
                }
            }
            
        return results
    
    def _calculate_robustness_score(self, solution: Dict, scenario: Dict) -> float:
        """
        Calculate robustness score for a solution.
        
        Args:
            solution: EMS solution
            scenario: Scenario data
            
        Returns:
            Robustness score
        """
        # Simple robustness score based on solution feasibility and cost
        if solution['status'] != 'optimal':
            return 0.0
            
        # Calculate cost relative to oracle
        oracle_cost = scenario.get('oracle_cost', solution['cost'])
        cost_ratio = solution['cost'] / oracle_cost
        
        # Penalize high costs and infeasible solutions
        robustness_score = 1.0 / (1.0 + cost_ratio)
        
        # Add penalty for constraint violations
        if solution.get('constraint_violations', 0) > 0:
            robustness_score *= 0.5
            
        return robustness_score
    
    def generate_robustness_report(self, results: Dict[str, Dict],
                                 output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive robustness report.
        
        Args:
            results: Dictionary with robustness analysis results
            output_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 60)
        report.append("ROBUSTNESS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Distribution shift analysis
        if 'distribution_shift' in results:
            report.append("1. DISTRIBUTION SHIFT ANALYSIS")
            report.append("-" * 40)
            for shift_type, metrics in results['distribution_shift'].items():
                report.append(f"Shift Type: {shift_type}")
                report.append(f"  CRPS: {metrics['crps']:.4f}")
                report.append(f"  Shift Magnitude: {metrics['shift_magnitude']:.4f}")
                report.append(f"  Relative Error: {metrics['relative_error']:.4f}")
                report.append("")
        
        # Adversarial attack analysis
        if 'adversarial_attacks' in results:
            report.append("2. ADVERSARIAL ATTACK ANALYSIS")
            report.append("-" * 40)
            for attack_type, metrics in results['adversarial_attacks'].items():
                report.append(f"Attack Type: {attack_type}")
                report.append(f"  MSE (Clean): {metrics['mse_clean']:.4f}")
                report.append(f"  MSE (Adversarial): {metrics['mse_adv']:.4f}")
                report.append(f"  Degradation: {metrics['degradation']:.2%}")
                report.append(f"  Perturbation Magnitude: {metrics['perturbation_magnitude']:.4f}")
                report.append("")
        
        # Noise robustness analysis
        if 'noise_robustness' in results:
            report.append("3. NOISE ROBUSTNESS ANALYSIS")
            report.append("-" * 40)
            for noise_type, metrics in results['noise_robustness'].items():
                report.append(f"Noise Level: {metrics['noise_level']:.2f}")
                report.append(f"  CRPS: {metrics['crps']:.4f}")
                report.append(f"  SNR: {metrics['signal_to_noise_ratio']:.2f}")
                report.append("")
        
        # Parameter sensitivity analysis
        if 'parameter_sensitivity' in results:
            report.append("4. PARAMETER SENSITIVITY ANALYSIS")
            report.append("-" * 40)
            for param, metrics in results['parameter_sensitivity'].items():
                report.append(f"Parameter: {param}")
                report.append(f"  Sensitivity: {metrics['sensitivity']:.4f}")
                report.append(f"  Max Change: {metrics['max_change']:.2%}")
                report.append("")
        
        # Outlier detection
        if 'outlier_detection' in results:
            report.append("5. OUTLIER DETECTION")
            report.append("-" * 40)
            for method, stats in results['outlier_detection'].items():
                report.append(f"Method: {method}")
                report.append(f"  Number of Outliers: {stats['n_outliers']}")
                report.append(f"  Outlier Ratio: {stats['outlier_ratio']:.2%}")
                report.append("")
        
        # Scenario robustness analysis
        if 'scenario_robustness' in results:
            report.append("6. SCENARIO ROBUSTNESS ANALYSIS")
            report.append("-" * 40)
            for scenario, metrics in results['scenario_robustness'].items():
                report.append(f"Scenario: {scenario}")
                report.append(f"  Type: {metrics['scenario_type']}")
                report.append(f"  Cost: {metrics['cost']:.2f}")
                report.append(f"  Robustness Score: {metrics['robustness_score']:.4f}")
                report.append(f"  Status: {metrics['solution_status']}")
                report.append("")
        
        # Overall summary
        report.append("7. OVERALL SUMMARY")
        report.append("-" * 40)
        report.append("Key Findings:")
        
        if 'distribution_shift' in results:
            worst_shift = min(results['distribution_shift'].items(), 
                            key=lambda x: x[1]['crps'])
            report.append(f"- Most affected distribution shift: {worst_shift[0]} "
                        f"(CRPS: {worst_shift[1]['crps']:.4f})")
        
        if 'adversarial_attacks' in results:
            worst_attack = max(results['adversarial_attacks'].items(), 
                             key=lambda x: x[1]['degradation'])
            report.append(f"- Most severe adversarial attack: {worst_attack[0]} "
                        f"(degradation: {worst_attack[1]['degradation']:.2%})")
        
        if 'noise_robustness' in results:
            worst_noise = max(results['noise_robustness'].items(), 
                            key=lambda x: x[1]['crps'])
            report.append(f"- Most affected noise level: {worst_noise[0]} "
                        f"(CRPS: {worst_noise[1]['crps']:.4f})")
        
        if 'scenario_robustness' in results:
            avg_robustness = np.mean([m['robustness_score'] 
                                    for m in results['scenario_robustness'].values()])
            report.append(f"- Average scenario robustness score: {avg_robustness:.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
                
        return report_text


def run_comprehensive_robustness_analysis(model: nn.Module, 
                                       test_data: Dict[str, np.ndarray],
                                       output_dir: str = "robustness_results") -> Dict[str, Dict]:
    """
    Run comprehensive robustness analysis.
    
    Args:
        model: Trained model
        test_data: Dictionary with test data
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all robustness analysis results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = RobustnessAnalyzer()
    
    # Collect results
    results = {}
    
    # 1. Distribution shift analysis
    results['distribution_shift'] = analyzer.analyze_distribution_shift(
        test_data['predictions'], test_data['true_values']
    )
    
    # 2. Adversarial attack analysis
    results['adversarial_attacks'] = analyzer.analyze_adversarial_attacks(
        model, test_data['X_tensor'], test_data['y_tensor']
    )
    
    # 3. Noise robustness analysis
    results['noise_robustness'] = analyzer.analyze_noise_robustness(
        test_data['predictions'], test_data['true_values']
    )
    
    # 4. Outlier detection
    outlier_mask, outlier_stats = analyzer.detect_outliers(test_data['true_values'])
    results['outlier_detection'] = {'isolation': outlier_stats}
    
    # 5. Scenario robustness analysis
    scenarios = test_data.get('scenarios', [])
    if scenarios:
        results['scenario_robustness'] = analyzer.analyze_scenario_robustness(
            scenarios, model, lambda net, price, batt: solve_ems(net, price, batt)
        )
    
    # Generate report
    report_path = output_path / "robustness_report.txt"
    report = analyzer.generate_robustness_report(results, str(report_path))
    
    # Save results
    import json
    with open(output_path / "robustness_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results