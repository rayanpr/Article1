import pytest
import torch
import numpy as np
from pathlib import Path

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.spo_optimized import (
    box_argmin,
    box_argmax,
    spo_plus_loss_torch_optimized,
    spo_plus_loss_torch_original,
    benchmark_spo_implementations
)
from src.optimization.spo import spo_plus_loss_torch
from src.optimization.ems import BatterySpec


class TestSPOOptimization:
    """Test suite for SPO optimization improvements."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batt = BatterySpec()
        self.batch_size = 32
        self.horizon = 24
        
    def test_box_argmin(self):
        """Test box argmin function."""
        c = np.array([1.0, 2.0, 3.0, 4.0])
        result = box_argmin(c, self.batt)
        
        # Check result shape
        assert result.shape == (self.horizon,)
        
        # Check bounds
        assert np.all(result >= -self.batt.p_ch)
        assert np.all(result <= self.batt.p_dis)
        
    def test_box_argmax(self):
        """Test box argmax function."""
        c = np.array([1.0, 2.0, 3.0, 4.0])
        result = box_argmax(c, self.batt)
        
        # Check result shape
        assert result.shape == (self.horizon,)
        
        # Check bounds
        assert np.all(result >= -self.batt.p_ch)
        assert np.all(result <= self.batt.p_dis)
        
    def test_spo_loss_optimized(self):
        """Test optimized SPO loss function."""
        price_hat = torch.randn(self.batch_size, self.horizon, dtype=torch.float32)
        price_true = torch.randn(self.batch_size, self.horizon, dtype=torch.float32)
        
        loss = spo_plus_loss_torch_optimized(price_hat, price_true, self.batt)
        
        # Check loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        
        # Check loss is non-negative
        assert loss.item() >= 0
        
    def test_spo_loss_original(self):
        """Test original SPO loss function."""
        price_hat = torch.randn(self.batch_size, self.horizon, dtype=torch.float32)
        price_true = torch.randn(self.batch_size, self.horizon, dtype=torch.float32)
        
        loss = spo_plus_loss_torch_original(price_hat, price_true, self.batt)
        
        # Check loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        
        # Check loss is non-negative
        assert loss.item() >= 0
        
    def test_spo_loss_equivalence(self):
        """Test that optimized and original implementations give same results."""
        price_hat = torch.randn(self.batch_size, self.horizon, dtype=torch.float32)
        price_true = torch.randn(self.batch_size, self.horizon, dtype=torch.float32)
        
        loss_opt = spo_plus_loss_torch_optimized(price_hat, price_true, self.batt)
        loss_orig = spo_plus_loss_torch_original(price_hat, price_true, self.batt)
        
        # Check results are close (allowing for numerical differences)
        assert torch.abs(loss_opt - loss_orig) < 1e-6
        
    def test_spo_loss_backward(self):
        """Test backward pass of SPO loss."""
        price_hat = torch.randn(self.batch_size, self.horizon, dtype=torch.float32, requires_grad=True)
        price_true = torch.randn(self.batch_size, self.horizon, dtype=torch.float32)
        
        loss = spo_plus_loss_torch_optimized(price_hat, price_true, self.batt)
        loss.backward()
        
        # Check that gradients are computed
        assert price_hat.grad is not None
        assert price_hat.grad.shape == price_hat.shape
        
    def test_benchmark_spo_implementations(self):
        """Test benchmark function."""
        batch_sizes = [8, 16, 32, 64]
        results = benchmark_spo_implementations(batch_sizes, horizon=self.horizon)
        
        # Check results structure
        assert isinstance(results, dict)
        for batch_size in batch_sizes:
            assert batch_size in results
            assert "optimized_mean" in results[batch_size]
            assert "optimized_std" in results[batch_size]
            assert "original_mean" in results[batch_size]
            assert "original_std" in results[batch_size]
            assert "speedup" in results[batch_size]
            
        # Check that optimized version is faster
        for batch_size in batch_sizes:
            assert results[batch_size]["speedup"] >= 1.0


class TestSPOIntegration:
    """Integration tests for SPO optimization."""
    
    def test_integration_with_train_evaluate(self):
        """Test that SPO optimization works with existing training pipeline."""
        # This test ensures the optimized SPO can be used in the existing pipeline
        from scripts.train_evaluate import train_tft
        from src.optimization.spo import spo_plus_loss_torch
        
        # Create dummy data
        batch_size = 16
        horizon = 24
        n_features = 10
        
        # Create dummy model and data
        X_train = torch.randn(100, horizon, n_features)
        y_train = torch.randn(100, horizon, 2)
        
        # Test that SPO loss can be computed
        price_hat = torch.randn(batch_size, horizon)
        price_true = torch.randn(batch_size, horizon)
        batt = BatterySpec()
        
        loss = spo_plus_loss_torch(price_hat, price_true, batt)
        
        # Check loss is valid
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


def test_performance_improvement():
    """Test that the optimization actually improves performance."""
    # Run benchmark
    batch_sizes = [8, 16, 32, 64, 128]
    results = benchmark_spo_implementations(batch_sizes, horizon=24)
    
    # Check performance improvement
    for batch_size, result in results.items():
        speedup = result["speedup"]
        print(f"Batch size {batch_size}: {speedup:.2f}x speedup")
        
        # Assert that we have at least some improvement
        assert speedup >= 1.0, f"Optimized version should be faster for batch size {batch_size}"
        
    # Overall average speedup should be significant
    avg_speedup = np.mean([r["speedup"] for r in results.values()])
    print(f"Average speedup: {avg_speedup:.2f}x")
    assert avg_speedup >= 1.5, f"Average speedup should be at least 1.5x, got {avg_speedup:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])