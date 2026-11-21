from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
import torch.nn.functional as F

from src.forecasting.tft import TFTQuantile
from src.forecasting.metrics import DEFAULT_TAUS, evaluate_forecast


class RobustTFT(TFTQuantile):
    """
    Robust TFT implementation with additional features for uncertainty quantification.
    
    This extends the base TFT model with:
    - Monte Carlo dropout for uncertainty estimation
    - Ensemble training for improved robustness
    - Adversarial training capabilities
    - Distribution shift detection
    """
    
    def __init__(self, n_features: int, horizon: int, n_targets: int, 
                 hidden: int = 64, taus: List[float] = DEFAULT_TAUS,
                 dropout_rate: float = 0.1, use_mc_dropout: bool = True,
                 n_ensemble: int = 1):
        """
        Initialize Robust TFT.
        
        Args:
            n_features: Number of input features
            horizon: Prediction horizon
            n_targets: Number of target variables
            hidden: Hidden layer size
            taus: Quantile levels
            dropout_rate: Dropout rate
            use_mc_dropout: Whether to use Monte Carlo dropout
            n_ensemble: Number of models in ensemble
        """
        super().__init__(n_features, horizon, n_targets, hidden, taus)
        
        self.dropout_rate = dropout_rate
        self.use_mc_dropout = use_mc_dropout
        self.n_ensemble = n_ensemble
        
        # Add dropout layers if enabled
        if use_mc_dropout:
            self.dropout = nn.Dropout(dropout_rate)
            
        # Additional layers for robustness
        self.robustness_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
        
    def forward(self, x: torch.Tensor, return_uncertainty: bool = False) -> torch.Tensor:
        """
        Forward pass with optional uncertainty estimation.
        
        Args:
            x: Input tensor
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Output tensor with optional uncertainty
        """
        # Base TFT forward pass
        output = super().forward(x)
        
        if return_uncertainty and self.use_mc_dropout:
            # Monte Carlo dropout for uncertainty estimation
            uncertainty = []
            for _ in range(10):  # Number of MC samples
                mc_output = super().forward(self.dropout(x))
                uncertainty.append(mc_output)
            uncertainty = torch.stack(uncertainty)
            uncertainty_std = uncertainty.std(dim=0)
            
            # Combine prediction and uncertainty
            output = torch.cat([output, uncertainty_std], dim=-1)
            
        return output
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (predictions, uncertainty)
        """
        predictions = []
        uncertainties = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x, return_uncertainty=True)
                predictions.append(pred)
                uncertainties.append(pred[..., -1:])  # Last dimension is uncertainty
                
        predictions = torch.stack(predictions)
        uncertainties = torch.stack(uncertainties)
        
        # Return mean prediction and uncertainty
        return predictions.mean(dim=0), uncertainties.mean(dim=0)


class EnsembleTFT(nn.Module):
    """
    Ensemble of TFT models for improved robustness.
    """
    
    def __init__(self, n_models: int, n_features: int, horizon: int, 
                 n_targets: int, hidden: int = 64, taus: List[float] = DEFAULT_TAUS):
        """
        Initialize ensemble of TFT models.
        
        Args:
            n_models: Number of models in ensemble
            n_features: Number of input features
            horizon: Prediction horizon
            n_targets: Number of target variables
            hidden: Hidden layer size
            taus: Quantile levels
        """
        super().__init__()
        
        self.n_models = n_models
        self.models = nn.ModuleList([
            RobustTFT(n_features, horizon, n_targets, hidden, taus)
            for _ in range(n_models)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
            
        return torch.stack(predictions)
    
    def predict_ensemble(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with ensemble statistics.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean predictions, prediction uncertainty)
        """
        predictions = self.forward(x)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred


class AdversarialTFT(nn.Module):
    """
    TFT model with adversarial training capabilities.
    """
    
    def __init__(self, n_features: int, horizon: int, n_targets: int,
                 hidden: int = 64, taus: List[float] = DEFAULT_TAUS):
        """
        Initialize adversarial TFT.
        
        Args:
            n_features: Number of input features
            horizon: Prediction horizon
            n_targets: Number of target variables
            hidden: Hidden layer size
            taus: Quantile levels
        """
        super().__init__()
        
        self.tft = TFTQuantile(n_features, horizon, n_targets, hidden, taus)
        self.adversarial_head = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with adversarial prediction.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (main prediction, adversarial prediction)
        """
        main_pred = self.tft(x)
        adv_pred = self.adversarial_head(x)
        
        return main_pred, adv_pred


class DistributionShiftDetector(nn.Module):
    """
    Model to detect distribution shifts in input data.
    """
    
    def __init__(self, n_features: int, hidden: int = 64):
        """
        Initialize distribution shift detector.
        
        Args:
            n_features: Number of input features
            hidden: Hidden layer size
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect distribution shift.
        
        Args:
            x: Input tensor
            
        Returns:
            Shift probability
        """
        return self.encoder(x)


def train_robust_tft(train_loader: DataLoader, val_loader: DataLoader,
                    n_features: int, horizon: int, n_targets: int,
                    hidden: int = 64, taus: List[float] = DEFAULT_TAUS,
                    n_epochs: int = 50, lr: float = 1e-3,
                    dropout_rate: float = 0.1, use_mc_dropout: bool = True,
                    n_ensemble: int = 1, device: str = 'cpu') -> RobustTFT:
    """
    Train a robust TFT model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        n_features: Number of input features
        horizon: Prediction horizon
        n_targets: Number of target variables
        hidden: Hidden layer size
        taus: Quantile levels
        n_epochs: Number of training epochs
        lr: Learning rate
        dropout_rate: Dropout rate
        use_mc_dropout: Whether to use Monte Carlo dropout
        n_ensemble: Number of models in ensemble
        device: Device to train on
        
    Returns:
        Trained robust TFT model
    """
    model = RobustTFT(n_features, horizon, n_targets, hidden, taus,
                     dropout_rate, use_mc_dropout, n_ensemble)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    model.load_state_dict(best_model_state)
    return model


def train_ensemble_tft(train_loader: DataLoader, val_loader: DataLoader,
                      n_features: int, horizon: int, n_targets: int,
                      hidden: int = 64, taus: List[float] = DEFAULT_TAUS,
                      n_models: int = 5, n_epochs: int = 50, lr: float = 1e-3,
                      device: str = 'cpu') -> EnsembleTFT:
    """
    Train an ensemble of TFT models.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        n_features: Number of input features
        horizon: Prediction horizon
        n_targets: Number of target variables
        hidden: Hidden layer size
        taus: Quantile levels
        n_models: Number of models in ensemble
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Trained ensemble TFT model
    """
    ensemble = EnsembleTFT(n_models, n_features, horizon, n_targets, hidden, taus)
    ensemble = ensemble.to(device)
    
    # Train each model in the ensemble
    for i, model in enumerate(ensemble.models):
        print(f"Training model {i+1}/{n_models}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
        
        model.load_state_dict(best_model_state)
    
    return ensemble


def evaluate_robustness(model: nn.Module, test_loader: DataLoader,
                        n_samples: int = 100, device: str = 'cpu') -> Dict[str, float]:
    """
    Evaluate model robustness using Monte Carlo dropout.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        n_samples: Number of MC samples
        device: Device to evaluate on
        
    Returns:
        Dictionary with robustness metrics
    """
    model.eval()
    
    predictions = []
    uncertainties = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            
            # Get predictions with uncertainty
            if hasattr(model, 'predict_with_uncertainty'):
                pred, unc = model.predict_with_uncertainty(batch_x, n_samples)
                predictions.append(pred.cpu())
                uncertainties.append(unc.cpu())
            else:
                # For models without uncertainty estimation
                pred = model(batch_x)
                predictions.append(pred.cpu())
                uncertainties.append(torch.zeros_like(pred.cpu()))
    
    predictions = torch.cat(predictions)
    uncertainties = torch.cat(uncertainties)
    
    # Calculate robustness metrics
    metrics = {
        'mean_prediction': float(predictions.mean().item()),
        'std_prediction': float(predictions.std().item()),
        'mean_uncertainty': float(uncertainties.mean().item()),
        'std_uncertainty': float(uncertainties.std().item()),
        'uncertainty_coverage': float((uncertainties > 0).float().mean().item())
    }
    
    return metrics


def detect_distribution_shift(model: DistributionShiftDetector,
                             data_loader: DataLoader,
                             threshold: float = 0.5,
                             device: str = 'cpu') -> Dict[str, float]:
    """
    Detect distribution shifts in input data.
    
    Args:
        model: Trained distribution shift detector
        data_loader: Data loader to analyze
        threshold: Shift probability threshold
        device: Device to evaluate on
        
    Returns:
        Dictionary with shift detection metrics
    """
    model.eval()
    
    shift_probabilities = []
    
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            shift_prob = model(batch_x)
            shift_probabilities.append(shift_prob.cpu())
    
    shift_probabilities = torch.cat(shift_probabilities)
    
    # Calculate shift detection metrics
    metrics = {
        'mean_shift_probability': float(shift_probabilities.mean().item()),
        'max_shift_probability': float(shift_probabilities.max().item()),
        'shift_ratio': float((shift_probabilities > threshold).float().mean().item()),
        'shift_ratio_75': float((shift_probabilities > 0.75).float().mean().item()),
        'shift_ratio_90': float((shift_probabilities > 0.9).float().mean().item())
    }
    
    return metrics