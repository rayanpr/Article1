import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forecasting.robust_models import (
    RobustTFT,
    EnsembleTFT,
    AdversarialTFT,
    DistributionShiftDetector,
    train_robust_tft,
    train_ensemble_tft,
    evaluate_robustness,
    detect_distribution_shift
)
from src.forecasting.tft import TFTQuantile
from src.forecasting.metrics import DEFAULT_TAUS


class TestRobustTFT:
    """Test suite for Robust TFT implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_features = 10
        self.horizon = 24
        self.n_targets = 2
        self.hidden = 64
        self.taus = DEFAULT_TAUS
        
        # Create test data
        self.batch_size = 32
        self.X = torch.randn(self.batch_size, self.horizon, self.n_features)
        self.y = torch.randn(self.batch_size, self.horizon, self.n_targets, len(self.taus))
        
    def test_init(self):
        """Test RobustTFT initialization."""
        model = RobustTFT(self.n_features, self.horizon, self.n_targets, 
                         self.hidden, self.taus, dropout_rate=0.1, 
                         use_mc_dropout=True, n_ensemble=1)
        
        assert model.n_features == self.n_features
        assert model.horizon == self.horizon
        assert model.n_targets == self.n_targets
        assert model.hidden == self.hidden
        assert model.dropout_rate == 0.1
        assert model.use_mc_dropout == True
        assert model.n_ensemble == 1
        
    def test_forward(self):
        """Test forward pass."""
        model = RobustTFT(self.n_features, self.horizon, self.n_targets, 
                         self.hidden, self.taus, dropout_rate=0.1, 
                         use_mc_dropout=True, n_ensemble=1)
        
        # Standard forward pass
        output = model(self.X)
        assert output.shape == (self.batch_size, self.horizon, self.n_targets, len(self.taus))
        
        # Forward pass with uncertainty
        output_unc = model(self.X, return_uncertainty=True)
        expected_shape = (self.batch_size, self.horizon, self.n_targets, len(self.taus) + 1)
        assert output_unc.shape == expected_shape
        
    def test_predict_with_uncertainty(self):
        """Test prediction with uncertainty estimation."""
        model = RobustTFT(self.n_features, self.horizon, self.n_targets, 
                         self.hidden, self.taus, dropout_rate=0.1, 
                         use_mc_dropout=True, n_ensemble=1)
        
        pred, unc = model.predict_with_uncertainty(self.X, n_samples=10)
        
        # Check shapes
        assert pred.shape == (self.batch_size, self.horizon, self.n_targets, len(self.taus))
        assert unc.shape == (self.batch_size, self.horizon, self.n_targets, 1)
        
    def test_dropout_training(self):
        """Test dropout during training."""
        model = RobustTFT(self.n_features, self.horizon, self.n_targets, 
                         self.hidden, self.taus, dropout_rate=0.1, 
                         use_mc_dropout=True, n_ensemble=1)
        
        # Training mode should have dropout
        model.train()
        assert model.training == True
        
        # Evaluation mode should not have dropout for MC
        model.eval()
        assert model.training == False


class TestEnsembleTFT:
    """Test suite for Ensemble TFT implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_models = 3
        self.n_features = 10
        self.horizon = 24
        self.n_targets = 2
        self.hidden = 64
        self.taus = DEFAULT_TAUS
        
        # Create test data
        self.batch_size = 32
        self.X = torch.randn(self.batch_size, self.horizon, self.n_features)
        
    def test_init(self):
        """Test EnsembleTFT initialization."""
        ensemble = EnsembleTFT(self.n_models, self.n_features, self.horizon, 
                              self.n_targets, self.hidden, self.taus)
        
        assert ensemble.n_models == self.n_models
        assert len(ensemble.models) == self.n_models
        
    def test_forward(self):
        """Test forward pass."""
        ensemble = EnsembleTFT(self.n_models, self.n_features, self.horizon, 
                              self.n_targets, self.hidden, self.taus)
        
        output = ensemble(self.X)
        expected_shape = (self.n_models, self.batch_size, self.horizon, 
                         self.n_targets, len(self.taus))
        assert output.shape == expected_shape
        
    def test_predict_ensemble(self):
        """Test ensemble prediction."""
        ensemble = EnsembleTFT(self.n_models, self.n_features, self.horizon, 
                              self.n_targets, self.hidden, self.taus)
        
        mean_pred, std_pred = ensemble.predict_ensemble(self.X)
        
        # Check shapes
        assert mean_pred.shape == (self.batch_size, self.horizon, self.n_targets, len(self.taus))
        assert std_pred.shape == (self.batch_size, self.horizon, self.n_targets, len(self.taus))


class TestAdversarialTFT:
    """Test suite for Adversarial TFT implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_features = 10
        self.horizon = 24
        self.n_targets = 2
        self.hidden = 64
        self.taus = DEFAULT_TAUS
        
        # Create test data
        self.batch_size = 32
        self.X = torch.randn(self.batch_size, self.horizon, self.n_features)
        
    def test_init(self):
        """Test AdversarialTFT initialization."""
        model = AdversarialTFT(self.n_features, self.horizon, self.n_targets, 
                              self.hidden, self.taus)
        
        assert model.tft is not None
        assert model.adversarial_head is not None
        
    def test_forward(self):
        """Test forward pass."""
        model = AdversarialTFT(self.n_features, self.horizon, self.n_targets, 
                              self.hidden, self.taus)
        
        main_pred, adv_pred = model(self.X)
        
        # Check shapes
        assert main_pred.shape == (self.batch_size, self.horizon, self.n_targets, len(self.taus))
        assert adv_pred.shape == (self.batch_size, self.horizon, 1)


class TestDistributionShiftDetector:
    """Test suite for Distribution Shift Detector implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_features = 10
        self.hidden = 64
        
        # Create test data
        self.batch_size = 32
        self.X = torch.randn(self.batch_size, self.n_features)
        
    def test_init(self):
        """Test DistributionShiftDetector initialization."""
        detector = DistributionShiftDetector(self.n_features, self.hidden)
        
        assert detector.encoder is not None
        
    def test_forward(self):
        """Test forward pass."""
        detector = DistributionShiftDetector(self.n_features, self.hidden)
        
        output = detector(self.X)
        
        # Check output shape
        assert output.shape == (self.batch_size, 1)
        
        # Check output range
        assert torch.all(output >= 0) and torch.all(output <= 1)


class TestTrainingFunctions:
    """Test suite for training functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_features = 10
        self.horizon = 24
        self.n_targets = 2
        self.hidden = 64
        self.taus = DEFAULT_TAUS
        self.batch_size = 32
        
        # Create test data loaders
        train_data = TensorDataset(
            torch.randn(100, self.horizon, self.n_features),
            torch.randn(100, self.horizon, self.n_targets, len(self.taus))
        )
        val_data = TensorDataset(
            torch.randn(50, self.horizon, self.n_features),
            torch.randn(50, self.horizon, self.n_targets, len(self.taus))
        )
        
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size)
        
    def test_train_robust_tft(self):
        """Test robust TFT training."""
        model = train_robust_tft(
            self.train_loader, self.val_loader,
            self.n_features, self.horizon, self.n_targets,
            hidden=self.hidden, taus=self.taus,
            n_epochs=2, lr=1e-3
        )
        
        assert model is not None
        assert isinstance(model, RobustTFT)
        
    def test_train_ensemble_tft(self):
        """Test ensemble TFT training."""
        ensemble = train_ensemble_tft(
            self.train_loader, self.val_loader,
            self.n_features, self.horizon, self.n_targets,
            hidden=self.hidden, taus=self.taus,
            n_models=2, n_epochs=2, lr=1e-3
        )
        
        assert ensemble is not None
        assert isinstance(ensemble, EnsembleTFT)
        assert len(ensemble.models) == 2


class TestEvaluationFunctions:
    """Test suite for evaluation functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_features = 10
        self.horizon = 24
        self.n_targets = 2
        self.hidden = 64
        self.taus = DEFAULT_TAUS
        self.batch_size = 32
        
        # Create test data loaders
        test_data = TensorDataset(
            torch.randn(50, self.horizon, self.n_features),
            torch.randn(50, self.horizon, self.n_targets, len(self.taus))
        )
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size)
        
    def test_evaluate_robustness(self):
        """Test robustness evaluation."""
        # Create a simple model for testing
        model = RobustTFT(self.n_features, self.horizon, self.n_targets, 
                         self.hidden, self.taus, dropout_rate=0.1, 
                         use_mc_dropout=True, n_ensemble=1)
        
        metrics = evaluate_robustness(model, self.test_loader, n_samples=10)
        
        # Check metrics
        assert isinstance(metrics, dict)
        assert 'mean_prediction' in metrics
        assert 'std_prediction' in metrics
        assert 'mean_uncertainty' in metrics
        assert 'std_uncertainty' in metrics
        assert 'uncertainty_coverage' in metrics
        
        # Check value ranges
        assert 0 <= metrics['uncertainty_coverage'] <= 1
        
    def test_detect_distribution_shift(self):
        """Test distribution shift detection."""
        # Create a simple detector for testing
        detector = DistributionShiftDetector(self.n_features, self.hidden)
        
        metrics = detect_distribution_shift(detector, self.test_loader, threshold=0.5)
        
        # Check metrics
        assert isinstance(metrics, dict)
        assert 'mean_shift_probability' in metrics
        assert 'max_shift_probability' in metrics
        assert 'shift_ratio' in metrics
        assert 'shift_ratio_75' in metrics
        assert 'shift_ratio_90' in metrics
        
        # Check value ranges
        assert 0 <= metrics['mean_shift_probability'] <= 1
        assert 0 <= metrics['max_shift_probability'] <= 1
        assert 0 <= metrics['shift_ratio'] <= 1
        assert 0 <= metrics['shift_ratio_75'] <= 1
        assert 0 <= metrics['shift_ratio_90'] <= 1


class TestIntegration:
    """Integration tests for robust models."""
    
    def test_integration_with_base_tft(self):
        """Test that robust models integrate well with base TFT."""
        n_features = 10
        horizon = 24
        n_targets = 2
        hidden = 64
        taus = DEFAULT_TAUS
        
        # Create base TFT
        base_tft = TFTQuantile(n_features, horizon, n_targets, hidden, taus)
        
        # Create robust TFT
        robust_tft = RobustTFT(n_features, horizon, n_targets, hidden, taus)
        
        # Test that they can be created with same parameters
        assert base_tft.n_features == robust_tft.n_features
        assert base_tft.horizon == robust_tft.horizon
        assert base_tft.n_targets == robust_tft.n_targets
        assert base_tft.hidden == robust_tft.hidden
        assert base_tft.taus == robust_tft.taus
        
    def test_model_compatibility(self):
        """Test that robust models are compatible with existing code."""
        n_features = 10
        horizon = 24
        n_targets = 2
        hidden = 64
        taus = DEFAULT_TAUS
        
        # Create test data
        X = torch.randn(10, horizon, n_features)
        
        # Test different model types
        robust_tft = RobustTFT(n_features, horizon, n_targets, hidden, taus)
        ensemble_tft = EnsembleTFT(3, n_features, horizon, n_targets, hidden, taus)
        adversarial_tft = AdversarialTFT(n_features, horizon, n_targets, hidden, taus)
        detector = DistributionShiftDetector(n_features, hidden)
        
        # Test forward passes
        robust_out = robust_tft(X)
        ensemble_out = ensemble_tft(X)
        adversarial_out = adversarial_tft(X)
        detector_out = detector(X)
        
        # Check output shapes
        assert robust_out.shape == (10, horizon, n_targets, len(taus))
        assert ensemble_out.shape == (3, 10, horizon, n_targets, len(taus))
        assert adversarial_out[0].shape == (10, horizon, n_targets, len(taus))
        assert adversarial_out[1].shape == (10, horizon, 1)
        assert detector_out.shape == (10, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])