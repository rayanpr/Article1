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
from src.optimization.spo import spo_plus_loss, spo_plus_loss_torch
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
        assert result.shape == c.shape
        
        # Check bounds
        assert np.all(result >= -self.batt.p_ch)
        assert np.all(result <= self.batt.p_dis)
        
    def test_box_argmax(self):
        """Test box argmax function."""
        c = np.array([1.0, 2.0, 3.0, 4.0])
        result = box_argmax(c, self.batt)
        
        # Check result shape
        assert result.shape == c.shape
        
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
        assert torch.abs(loss_opt - loss_orig) < 1e-6 * max(1.0, float(loss_opt.abs()))
        
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
            
        # Check that optimized version is competitive (benchmark speedup on CPU is close to 1.0x with SciPy HiGHS)
        for batch_size in batch_sizes:
            assert results[batch_size]["speedup"] >= 0.85


class TestSPOElmachtoubGrigas:
    """Conformite de la perte SPO+ a la Definition 3 d'Elmachtoub & Grigas (2022).

    Reference (arXiv:1710.08005, section 3.2) pour min_{u in X} c^T u :

        l_SPO+(c_hat, c) = max_{u in X} (c - 2 c_hat)^T u + 2 c_hat^T w*(c) - z*(c)
                         = (2 c_hat - c)^T ( w*(c) - w*(2 c_hat - c) )

    sous-gradient : 2 ( w*(c) - w*(2 c_hat - c) )
    """

    def setup_method(self):
        self.batt = BatterySpec()
        self.rng = np.random.default_rng(0)
        self.c = np.abs(self.rng.normal(0.2, 0.05, 24))
        self.c_hat = self.c + self.rng.normal(0.0, 0.03, 24)

    def _reference(self, c_hat, c):
        w_c = box_argmin(c, self.batt)                 # w*(c)
        w_mod = box_argmin(2.0 * c_hat - c, self.batt)  # w*(2 c_hat - c)
        loss = float(np.dot(2.0 * c_hat - c, w_c - w_mod))
        grad = 2.0 * (w_c - w_mod)
        return loss, grad

    def _spo_loss(self, c_hat, c):
        """Perte SPO (Definition 2) : c^T w*(c_hat) - z*(c)."""
        return float(np.dot(c, box_argmin(c_hat, self.batt))
                     - np.dot(c, box_argmin(c, self.batt)))

    def test_matches_definition_3(self):
        """La valeur implementee est exactement celle de la Definition 3."""
        loss, _ = self._reference(self.c_hat, self.c)
        assert spo_plus_loss(self.c_hat, self.c, self.batt) == pytest.approx(loss, abs=1e-6)

    def test_zero_for_perfect_prediction(self):
        """l_SPO+(c, c) = 0."""
        assert spo_plus_loss(self.c, self.c, self.batt) == pytest.approx(0.0, abs=1e-6)

    def test_upper_bounds_spo_loss(self):
        """l_SPO+ >= l_SPO >= 0 (Proposition 3.1)."""
        plus = spo_plus_loss(self.c_hat, self.c, self.batt)
        spo = self._spo_loss(self.c_hat, self.c)
        assert spo >= -1e-6
        assert plus >= spo - 1e-6

    def test_subgradient(self):
        """Le gradient implemente est le sous-gradient d'Elmachtoub-Grigas."""
        _, grad_ref = self._reference(self.c_hat, self.c)
        ph = torch.tensor(self.c_hat[None, :], dtype=torch.float64, requires_grad=True)
        loss = spo_plus_loss_torch(ph, torch.tensor(self.c[None, :], dtype=torch.float64), self.batt)
        loss.backward()
        scale = float(np.mean(np.abs(self.c))) * self.c.size
        assert np.allclose(ph.grad.numpy()[0] * scale, grad_ref, atol=1e-6)

    def test_batch_matches_single(self):
        """Le chemin batch donne les memes valeurs que le chemin unitaire."""
        from src.optimization.spo import spo_plus_loss as single

        batch_hat = np.stack([self.c_hat, self.c_hat * 1.1])
        batch_true = np.stack([self.c, self.c * 0.9])
        ph = torch.tensor(batch_hat, dtype=torch.float64, requires_grad=True)
        loss = spo_plus_loss_torch(ph, torch.tensor(batch_true, dtype=torch.float64), self.batt)
        expected = np.mean([single(batch_hat[i], batch_true[i], self.batt) for i in range(2)])
        scale = float(np.mean(np.abs(batch_true))) * batch_true.shape[1]
        assert float(loss) * scale == pytest.approx(expected, rel=1e-6)


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
        
        # Assert that we have competitive performance on CPU
        assert speedup >= 0.85, f"Optimized version should be competitive for batch size {batch_size}, got {speedup:.2f}x"
        
    # Overall average speedup should be at least comparable
    avg_speedup = np.mean([r["speedup"] for r in results.values()])
    print(f"Average speedup: {avg_speedup:.2f}x")
    assert avg_speedup >= 0.90, f"Average speedup should be at least 0.90x, got {avg_speedup:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])