import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import torch

# Add the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forecasting.advanced_features import (
    AdvancedFeatureEngineer,
    create_advanced_features
)


class TestAdvancedFeatureEngineer:
    """Test suite for advanced feature engineering."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engineer = AdvancedFeatureEngineer(country='FR')
        
        # Create test timestamps
        self.start_date = pd.Timestamp('2022-01-01')
        self.end_date = pd.Timestamp('2022-01-31')
        self.timestamps = pd.date_range(start=self.start_date, end=self.end_date, freq='H')
        
        # Create test data
        self.data = pd.DataFrame({
            'net_load': np.random.randn(len(self.timestamps)) * 100 + 500,
            'price': np.random.randn(len(self.timestamps)) * 0.1 + 0.2
        }, index=self.timestamps)
        
    def test_create_temporal_features(self):
        """Test temporal feature creation."""
        features = self.engineer.create_temporal_features(self.timestamps)
        
        # Check basic features
        assert 'hour' in features.columns
        assert 'day_of_week' in features.columns
        assert 'month' in features.columns
        assert 'year' in features.columns
        
        # Check cyclical features
        assert 'hour_sin' in features.columns
        assert 'hour_cos' in features.columns
        assert 'day_sin' in features.columns
        assert 'day_cos' in features.columns
        
        # Check binary features
        assert 'is_weekend' in features.columns
        assert 'is_business_day' in features.columns
        assert 'is_holiday' in features.columns
        
        # Check shape
        assert features.shape[0] == len(self.timestamps)
        
    def test_create_fourier_features(self):
        """Test Fourier feature creation."""
        features = self.engineer.create_fourier_features(self.timestamps)
        
        # Check that features are created
        assert 'fourier_sin_24_1' in features.columns
        assert 'fourier_cos_24_1' in features.columns
        assert 'fourier_sin_168_1' in features.columns
        assert 'fourier_cos_168_1' in features.columns
        
        # Check shape
        assert features.shape[0] == len(self.timestamps)
        
    def test_create_lag_features(self):
        """Test lag feature creation."""
        lags = [1, 2, 3, 24, 48]
        features = self.engineer.create_lag_features(self.data['net_load'], lags)
        
        # Check lag columns
        for lag in lags:
            assert f'lag_{lag}' in features.columns
            
        # Check shape
        assert features.shape[0] == len(self.timestamps)
        
    def test_create_rolling_features(self):
        """Test rolling feature creation."""
        windows = [3, 6, 12, 24]
        functions = ['mean', 'std', 'min', 'max']
        features = self.engineer.create_rolling_features(self.data['net_load'], windows, functions)
        
        # Check rolling columns
        for window in windows:
            for func in functions:
                assert f'rolling_{func}_{window}' in features.columns
                
        # Check shape
        assert features.shape[0] == len(self.timestamps)
        
    def test_create_diff_features(self):
        """Test difference feature creation."""
        periods = [1, 24, 168]
        features = self.engineer.create_diff_features(self.data['net_load'], periods)
        
        # Check diff columns
        for period in periods:
            assert f'diff_{period}' in features.columns
            
        # Check shape
        assert features.shape[0] == len(self.timestamps)
        
    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        # Create some basic features first
        temporal_features = self.engineer.create_temporal_features(self.timestamps)
        features = self.engineer.create_interaction_features(temporal_features)
        
        # Check interaction columns
        assert 'hour_weekday' in features.columns
        assert 'hour_month' in features.columns
        assert 'hour_business_day' in features.columns
        assert 'hour_summer' in features.columns
        assert 'hour_winter' in features.columns
        assert 'weekend_hour' in features.columns
        
        # Check shape
        assert features.shape[0] == len(self.timestamps)
        
    def test_create_all_features(self):
        """Test comprehensive feature creation."""
        features = self.engineer.create_all_features(
            self.data, 
            self.timestamps,
            target_columns=['net_load', 'price'],
            lags=[1, 2, 3, 24],
            rolling_windows=[3, 6, 12, 24]
        )
        
        # Check that we have features
        assert features.shape[0] == len(self.timestamps)
        assert features.shape[1] > 0
        
        # Check that we have temporal features
        assert 'hour' in features.columns
        assert 'month' in features.columns
        
        # Check that we have lag features
        assert 'lag_1_net_load' in features.columns
        assert 'lag_24_price' in features.columns
        
        # Check that we have rolling features
        assert 'rolling_mean_3_net_load' in features.columns
        assert 'rolling_std_6_price' in features.columns
        
    def test_create_weather_features(self):
        """Test weather feature creation."""
        # Create weather data
        weather_data = pd.DataFrame({
            'temperature': np.random.randn(len(self.timestamps)) * 10 + 15,
            'humidity': np.random.rand(len(self.timestamps)) * 100,
            'wind_speed': np.random.rand(len(self.timestamps)) * 20,
            'solar_radiation': np.random.rand(len(self.timestamps)) * 1000
        }, index=self.timestamps)
        
        features = self.engineer.create_weather_features(weather_data)
        
        # Check weather features
        assert 'temp' in features.columns
        assert 'temp_sin' in features.columns
        assert 'temp_cos' in features.columns
        assert 'humidity' in features.columns
        assert 'wind_speed' in features.columns
        assert 'solar_radiation' in features.columns
        
        # Check weather categories
        assert 'is_hot' in features.columns
        assert 'is_cold' in features.columns
        assert 'is_humid' in features.columns
        assert 'is_windy' in features.columns
        assert 'is_sunny' in features.columns
        
        # Check shape
        assert features.shape[0] == len(self.timestamps)
        
    def test_weather_features_disabled(self):
        """Test weather features when disabled."""
        engineer = AdvancedFeatureEngineer(country='FR', include_weather=False)
        features = engineer.create_weather_features(pd.DataFrame())
        
        # Should return empty DataFrame
        assert features.empty


class TestAdvancedFeaturesIntegration:
    """Integration tests for advanced features."""
    
    def test_integration_with_preprocessing(self):
        """Test integration with existing preprocessing."""
        # Create test data
        timestamps = pd.date_range(start='2022-01-01', end='2022-01-31', freq='H')
        data = pd.DataFrame({
            'net_load': np.random.randn(len(timestamps)) * 100 + 500,
            'price': np.random.randn(len(timestamps)) * 0.1 + 0.2
        }, index=timestamps)
        
        # Create advanced features
        features = create_advanced_features(data, timestamps, target_columns=['net_load', 'price'])
        
        # Check that features are created
        assert not features.empty
        assert features.shape[0] == len(timestamps)
        
        # Check that we have the right columns
        assert 'hour' in features.columns
        assert 'month' in features.columns
        assert 'lag_1_net_load' in features.columns
        assert 'rolling_mean_3_net_load' in features.columns
        
    def test_integration_with_tft(self):
        """Test integration with TFT model."""
        # This test ensures that advanced features work with TFT
        from src.forecasting.tft import TFTQuantile
        
        # Create test data
        timestamps = pd.date_range(start='2022-01-01', end='2022-01-31', freq='H')
        data = pd.DataFrame({
            'net_load': np.random.randn(len(timestamps)) * 100 + 500,
            'price': np.random.randn(len(timestamps)) * 0.1 + 0.2
        }, index=timestamps)
        
        # Create advanced features
        features = create_advanced_features(data, timestamps, target_columns=['net_load', 'price'])
        
        # Create TFT model
        model = TFTQuantile(
            n_features=features.shape[1], 
            horizon=24, 
            n_targets=2, 
            hidden=64, 
            taus=[0.1, 0.25, 0.5, 0.75, 0.9]
        )
        
        # Test that model can be created with advanced features
        assert model is not None
        
        # Test forward pass
        X = torch.randn(10, 24, features.shape[1])
        output = model(X)
        
        # Check output shape
        assert output.shape == (10, 24, 2, 5)  # batch, horizon, targets, quantiles


def test_feature_quality():
    """Test the quality of generated features."""
    # Create test data
    timestamps = pd.date_range(start='2022-01-01', end='2022-12-31', freq='H')
    data = pd.DataFrame({
        'net_load': np.random.randn(len(timestamps)) * 100 + 500,
        'price': np.random.randn(len(timestamps)) * 0.1 + 0.2
    }, index=timestamps)
    
    # Create advanced features
    engineer = AdvancedFeatureEngineer(country='FR')
    features = engineer.create_all_features(data, timestamps, target_columns=['net_load', 'price'])
    
    # Check for NaN values
    assert features.isnull().sum().sum() == 0, "Features should not contain NaN values"
    
    # Check for infinite values
    assert np.isfinite(features.values).all(), "Features should not contain infinite values"
    
    # Check feature diversity
    assert features.shape[1] > 50, "Should have created a diverse set of features"
    
    # Check that temporal features make sense
    assert features['hour'].min() == 0
    assert features['hour'].max() == 23
    assert features['day_of_week'].min() == 0
    assert features['day_of_week'].max() == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])