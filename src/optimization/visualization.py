from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.ems import BatterySpec
from src.forecasting.metrics import DEFAULT_TAUS


class EMSVisualizer:
    """
    Comprehensive visualization tools for Energy Management Systems.
    
    This class provides methods to create various visualizations:
    - Time series plots with confidence intervals
    - Cost comparisons and regret analysis
    - Battery state of charge visualization
    - Feature importance plots
    - Uncertainty quantification visualizations
    - Model performance comparisons
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Color schemes
        self.colors = {
            'actual': '#1f77b4',
            'predicted': '#ff7f0e',
            'upper': '#2ca02c',
            'lower': '#d62728',
            'oracle': '#9467bd',
            'baseline': '#8c564b',
            'battery_charge': '#2ca02c',
            'battery_discharge': '#d62728',
            'grid_import': '#ff7f0e',
            'grid_export': '#1f77b4'
        }
        
    def plot_time_series_with_confidence(self, timestamps: pd.DatetimeIndex,
                                       actual_values: np.ndarray,
                                       predicted_values: np.ndarray,
                                       confidence_intervals: Optional[np.ndarray] = None,
                                       title: str = "Time Series with Confidence Intervals",
                                       save_path: Optional[str] = None,
                                       figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot time series with confidence intervals.
        
        Args:
            timestamps: Datetime index
            actual_values: Actual values
            predicted_values: Predicted values
            confidence_intervals: Confidence intervals (lower, upper)
            title: Plot title
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual values
        ax.plot(timestamps, actual_values, 
                label='Actual', color=self.colors['actual'], linewidth=2)
        
        # Plot predicted values
        ax.plot(timestamps, predicted_values, 
                label='Predicted', color=self.colors['predicted'], linewidth=2, linestyle='--')
        
        # Plot confidence intervals
        if confidence_intervals is not None:
            lower, upper = confidence_intervals
            ax.fill_between(timestamps, lower, upper, 
                          alpha=0.2, color=self.colors['predicted'],
                          label='Confidence Interval')
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_cost_comparison(self, methods: List[str], costs: List[float],
                           regrets: List[float],
                           title: str = "Cost Comparison and Regret Analysis",
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot cost comparison between different methods.
        
        Args:
            methods: List of method names
            costs: List of costs
            regrets: List of regrets
            title: Plot title
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Cost comparison
        bars = ax1.bar(methods, costs, color=[self.colors['predicted'] if m != 'Oracle' else self.colors['oracle'] 
                                              for m in methods])
        ax1.set_title('Cost Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cost (€)', fontsize=12)
        ax1.set_xlabel('Method', fontsize=12)
        
        # Add value labels on bars
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cost:.2f}', ha='center', va='bottom')
        
        # Regret analysis
        bars = ax2.bar(methods, regrets, color=[self.colors['predicted'] if m != 'Oracle' else self.colors['oracle'] 
                                              for m in methods])
        ax2.set_title('Regret Analysis', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Regret (%)', fontsize=12)
        ax2.set_xlabel('Method', fontsize=12)
        
        # Add value labels on bars
        for bar, regret in zip(bars, regrets):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{regret:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_battery_state(self, timestamps: pd.DatetimeIndex,
                          state_of_charge: np.ndarray,
                          charge_schedule: np.ndarray,
                          discharge_schedule: np.ndarray,
                          title: str = "Battery State of Charge and Schedule",
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot battery state of charge and charging/discharging schedule.
        
        Args:
            timestamps: Datetime index
            state_of_charge: State of charge values
            charge_schedule: Charging schedule
            discharge_schedule: Discharging schedule
            title: Plot title
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # State of charge
        ax1.plot(timestamps, state_of_charge, 
                label='State of Charge', color=self.colors['actual'], linewidth=2)
        ax1.set_title('Battery State of Charge', fontsize=14, fontweight='bold')
        ax1.set_ylabel('State of Charge (%)', fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # Add SOC limits
        ax1.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Min SOC')
        ax1.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='Max SOC')
        ax1.legend()
        
        # Charge/discharge schedule
        ax2.fill_between(timestamps, 0, charge_schedule, 
                        where=(charge_schedule > 0), color=self.colors['battery_charge'], 
                        alpha=0.7, label='Charging')
        ax2.fill_between(timestamps, 0, discharge_schedule, 
                        where=(discharge_schedule < 0), color=self.colors['battery_discharge'], 
                        alpha=0.7, label='Discharging')
        ax2.set_title('Battery Charge/Discharge Schedule', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Power (kW)', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray,
                              title: str = "Feature Importance",
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8),
                              top_n: int = 20) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores
            title: Plot title
            save_path: Path to save the plot
            figsize: Figure size
            top_n: Number of top features to display
        """
        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1][:top_n]
        sorted_names = [feature_names[i] for i in indices]
        sorted_scores = importance_scores[indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(range(len(sorted_names)), sorted_scores)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Color bars
        for i, bar in enumerate(bars):
            if i < len(sorted_scores) * 0.2:  # Top 20%
                bar.set_color('#2ca02c')  # Green
            elif i < len(sorted_scores) * 0.5:  # Next 30%
                bar.set_color('#ff7f0e')  # Orange
            else:  # Bottom 50%
                bar.set_color('#d62728')  # Red
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_uncertainty_quantification(self, actual_values: np.ndarray,
                                     predicted_values: np.ndarray,
                                     uncertainty_intervals: np.ndarray,
                                     title: str = "Uncertainty Quantification",
                                     save_path: Optional[str] = None,
                                     figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot uncertainty quantification.
        
        Args:
            actual_values: Actual values
            predicted_values: Predicted values
            uncertainty_intervals: Uncertainty intervals
            title: Plot title
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot
        scatter = ax.scatter(predicted_values, actual_values, 
                           alpha=0.6, c=uncertainty_intervals[:, 0], 
                           cmap='viridis', s=50)
        
        # Plot perfect prediction line
        min_val = min(np.min(actual_values), np.min(predicted_values))
        max_val = max(np.max(actual_values), np.max(predicted_values))
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', alpha=0.8, label='Perfect Prediction')
        
        # Formatting
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Actual Values', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Uncertainty', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_model_performance_comparison(self, model_names: List[str], 
                                       metrics: Dict[str, List[float]],
                                       title: str = "Model Performance Comparison",
                                       save_path: Optional[str] = None,
                                       figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot model performance comparison using radar chart.
        
        Args:
            model_names: List of model names
            metrics: Dictionary of metrics for each model
            title: Plot title
            save_path: Path to save the plot
            figsize: Figure size
        """
        # Number of variables
        categories = list(metrics.keys())
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Plot each model
        for i, model_name in enumerate(model_names):
            values = [metrics[cat][i] for cat in categories]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)  # Assuming normalized metrics
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_forecast_fan_chart(self, timestamps: pd.DatetimeIndex,
                               quantiles: np.ndarray,
                               title: str = "Forecast Fan Chart",
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot forecast fan chart showing quantiles.
        
        Args:
            timestamps: Datetime index
            quantiles: Quantile values (n_timestamps, n_quantiles)
            title: Plot title
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort quantiles for proper coloring
        sorted_indices = np.argsort(quantiles, axis=1)
        sorted_quantiles = np.take_along_axis(quantiles, sorted_indices, axis=1)
        
        # Create fan chart
        for i in range(len(timestamps) - 1):
            # Get quantiles for this timestamp
            current_quantiles = sorted_quantiles[i]
            next_quantiles = sorted_quantiles[i + 1]
            
            # Create polygon for each interval
            for j in range(len(current_quantiles) - 1):
                vertices = [
                    (timestamps[i], current_quantiles[j]),
                    (timestamps[i], current_quantiles[j + 1]),
                    (timestamps[i + 1], next_quantiles[j + 1]),
                    (timestamps[i + 1], next_quantiles[j])
                ]
                
                # Color based on quantile level
                alpha = 0.3 + (j / (len(current_quantiles) - 1)) * 0.5
                color = plt.cm.Blues(j / (len(current_quantiles) - 1))
                
                polygon = plt.Polygon(vertices, facecolor=color, alpha=alpha, edgecolor='none')
                ax.add_patch(polygon)
        
        # Add median line
        median_idx = len(quantiles[0]) // 2
        ax.plot(timestamps, quantiles[:, median_idx], 
                color='red', linewidth=2, label='Median')
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_robustness_analysis(self, robustness_results: Dict[str, Dict],
                               title: str = "Robustness Analysis",
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot comprehensive robustness analysis.
        
        Args:
            robustness_results: Dictionary with robustness results
            title: Plot title
            save_path: Path to save the plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Adversarial attacks
        ax1 = axes[0, 0]
        if 'adversarial_attacks' in robustness_results:
            attacks = robustness_results['adversarial_attacks']
            methods = list(attacks.keys())
            degradations = [attacks[method]['degradation'] for method in methods]
            
            bars = ax1.bar(methods, degradations)
            ax1.set_title('Adversarial Attack Robustness', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Degradation', fontsize=10)
            ax1.set_xlabel('Attack Method', fontsize=10)
            
            # Color bars
            for bar, degradation in zip(bars, degradations):
                if degradation > 0.5:
                    bar.set_color('#d62728')  # Red for high degradation
                elif degradation > 0.2:
                    bar.set_color('#ff7f0e')  # Orange for medium degradation
                else:
                    bar.set_color('#2ca02c')  # Green for low degradation
        
        # 2. Noise robustness
        ax2 = axes[0, 1]
        if 'noise_robustness' in robustness_results:
            noise = robustness_results['noise_robustness']
            noise_levels = [float(key.split('_')[1]) for key in noise.keys()]
            crps_values = [noise[key]['crps'] for key in noise.keys()]
            
            ax2.plot(noise_levels, crps_values, 'o-', linewidth=2, markersize=8)
            ax2.set_title('Noise Robustness', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Noise Level', fontsize=10)
            ax2.set_ylabel('CRPS', fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        # 3. Distribution shift
        ax3 = axes[1, 0]
        if 'distribution_shift' in robustness_results:
            shift = robustness_results['distribution_shift']
            shift_types = list(shift.keys())
            crps_values = [shift[shift_type]['crps'] for shift_type in shift_types]
            
            bars = ax3.bar(shift_types, crps_values)
            ax3.set_title('Distribution Shift Robustness', fontsize=12, fontweight='bold')
            ax3.set_ylabel('CRPS', fontsize=10)
            ax3.set_xlabel('Shift Type', fontsize=10)
            
            # Color bars
            for bar, crps in zip(bars, crps_values):
                if crps > 0.5:
                    bar.set_color('#d62728')  # Red for high CRPS
                elif crps > 0.2:
                    bar.set_color('#ff7f0e')  # Orange for medium CRPS
                else:
                    bar.set_color('#2ca02c')  # Green for low CRPS
        
        # 4. Overall robustness score
        ax4 = axes[1, 1]
        overall_scores = []
        score_labels = []
        
        for category, results in robustness_results.items():
            if category in ['adversarial_attacks', 'noise_robustness', 'distribution_shift']:
                # Calculate category score (lower is better for CRPS, higher is worse for degradation)
                if category == 'adversarial_attacks':
                    category_scores = [results[method]['degradation'] for method in results.keys()]
                    score = np.mean(category_scores)
                    score_type = 'Degradation'
                else:
                    category_scores = [results[key]['crps'] for key in results.keys()]
                    score = np.mean(category_scores)
                    score_type = 'CRPS'
                
                overall_scores.append(score)
                score_labels.append(f'{category}\n({score_type})')
        
        bars = ax4.bar(range(len(overall_scores)), overall_scores)
        ax4.set_title('Overall Robustness Scores', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Score', fontsize=10)
        ax4.set_xticks(range(len(score_labels)))
        ax4.set_xticklabels(score_labels, fontsize=8)
        
        # Color bars
        for bar, score in zip(bars, overall_scores):
            if isinstance(score, float) and score > 0.5:
                bar.set_color('#d62728')  # Red for high scores
            elif isinstance(score, float) and score > 0.2:
                bar.set_color('#ff7f0e')  # Orange for medium scores
            else:
                bar.set_color('#2ca02c')  # Green for low scores
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def create_dashboard(self, results: Dict[str, Any], 
                        dashboard_path: Optional[str] = None) -> None:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            results: Dictionary with results
            dashboard_path: Path to save the dashboard
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Cost comparison
        ax1 = plt.subplot(3, 3, 1)
        if 'cost_comparison' in results:
            methods = results['cost_comparison']['methods']
            costs = results['cost_comparison']['costs']
            regrets = results['cost_comparison']['regrets']
            
            bars = ax1.bar(methods, costs, color=[self.colors['predicted'] if m != 'Oracle' else self.colors['oracle'] 
                                                  for m in methods])
            ax1.set_title('Cost Comparison', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Cost (€)', fontsize=10)
            
            # Add value labels
            for bar, cost in zip(bars, costs):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{cost:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Regret analysis
        ax2 = plt.subplot(3, 3, 2)
        if 'cost_comparison' in results:
            methods = results['cost_comparison']['methods']
            regrets = results['cost_comparison']['regrets']
            
            bars = ax2.bar(methods, regrets, color=[self.colors['predicted'] if m != 'Oracle' else self.colors['oracle'] 
                                                  for m in methods])
            ax2.set_title('Regret Analysis', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Regret (%)', fontsize=10)
            
            # Add value labels
            for bar, regret in zip(bars, regrets):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{regret:.2f}%', ha='center', va='bottom', fontsize=8)
        
        # 3. Model performance radar
        ax3 = plt.subplot(3, 3, 3, projection='polar')
        if 'model_performance' in results:
            metrics = results['model_performance']
            categories = list(metrics.keys())
            N = len(categories)
            
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            for i, model_name in enumerate(metrics['model_names']):
                values = [metrics[cat][i] for cat in categories]
                values += values[:1]
                
                ax3.plot(angles, values, linewidth=2, label=model_name)
                ax3.fill(angles, values, alpha=0.1)
            
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(categories)
            ax3.set_ylim(0, 1)
            ax3.set_title('Model Performance', fontsize=12, fontweight='bold')
            ax3.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=8)
        
        # 4. Robustness analysis
        ax4 = plt.subplot(3, 3, 4)
        if 'robustness' in results:
            robustness = results['robustness']
            categories = list(robustness.keys())
            scores = [robustness[cat] for cat in categories]
            
            bars = ax4.bar(categories, scores)
            ax4.set_title('Robustness Scores', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Score', fontsize=10)
            
            # Color bars
            for bar, score in zip(bars, scores):
                if score > 0.7:
                    bar.set_color('#d62728')
                elif score > 0.4:
                    bar.set_color('#ff7f0e')
                else:
                    bar.set_color('#2ca02c')
        
        # 5. Feature importance
        ax5 = plt.subplot(3, 3, 5)
        if 'feature_importance' in results:
            features = results['feature_importance']['features']
            importance = results['feature_importance']['importance']
            
            # Sort by importance
            indices = np.argsort(importance)[::-1][:10]
            sorted_features = [features[i] for i in indices]
            sorted_importance = importance[indices]
            
            bars = ax5.barh(range(len(sorted_features)), sorted_importance)
            ax5.set_yticks(range(len(sorted_features)))
            ax5.set_yticklabels(sorted_features)
            ax5.set_xlabel('Importance', fontsize=10)
            ax5.set_title('Top 10 Features', fontsize=12, fontweight='bold')
            
            # Color bars
            for i, bar in enumerate(bars):
                if i < 3:
                    bar.set_color('#2ca02c')
                elif i < 7:
                    bar.set_color('#ff7f0e')
                else:
                    bar.set_color('#d62728')
        
        # 6. Training metrics
        ax6 = plt.subplot(3, 3, 6)
        if 'training_metrics' in results:
            metrics = results['training_metrics']
            epochs = range(len(metrics['train_loss']))
            
            ax6.plot(epochs, metrics['train_loss'], label='Train Loss', linewidth=2)
            ax6.plot(epochs, metrics['val_loss'], label='Val Loss', linewidth=2)
            ax6.set_title('Training Metrics', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Epoch', fontsize=10)
            ax6.set_ylabel('Loss', fontsize=10)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Inference time
        ax7 = plt.subplot(3, 3, 7)
        if 'inference_metrics' in results:
            models = results['inference_metrics']['models']
            times = results['inference_metrics']['times']
            
            bars = ax7.bar(models, times)
            ax7.set_title('Inference Time', fontsize=12, fontweight='bold')
            ax7.set_ylabel('Time (s)', fontsize=10)
            
            # Add value labels
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height,
                        f'{time:.3f}s', ha='center', va='bottom', fontsize=8)
        
        # 8. Memory usage
        ax8 = plt.subplot(3, 3, 8)
        if 'memory_metrics' in results:
            models = results['memory_metrics']['models']
            memory = results['memory_metrics']['memory']
            
            bars = ax8.bar(models, memory)
            ax8.set_title('Memory Usage', fontsize=12, fontweight='bold')
            ax8.set_ylabel('Memory (MB)', fontsize=10)
            
            # Add value labels
            for bar, mem in zip(bars, memory):
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mem:.1f}MB', ha='center', va='bottom', fontsize=8)
        
        # 9. Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Create summary text
        summary_text = "SUMMARY STATISTICS\n" + "="*20 + "\n\n"
        
        if 'cost_comparison' in results:
            best_method = results['cost_comparison']['methods'][np.argmin(results['cost_comparison']['costs'])]
            summary_text += f"Best Cost Method: {best_method}\n"
            summary_text += f"Best Cost: {min(results['cost_comparison']['costs']):.2f}€\n\n"
        
        if 'robustness' in results:
            avg_robustness = np.mean(list(results['robustness'].values()))
            summary_text += f"Average Robustness: {avg_robustness:.3f}\n\n"
        
        if 'inference_metrics' in results:
            avg_inference = np.mean(results['inference_metrics']['times'])
            summary_text += f"Average Inference Time: {avg_inference:.3f}s\n\n"
        
        if 'memory_metrics' in results:
            avg_memory = np.mean(results['memory_metrics']['memory'])
            summary_text += f"Average Memory Usage: {avg_memory:.1f}MB"
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('EMS Performance Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if dashboard_path:
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        
        plt.show()