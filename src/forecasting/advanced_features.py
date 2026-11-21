from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import holidays

from src.forecasting.preprocessing import create_time_features


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for energy forecasting.
    
    This class provides methods to create advanced features including:
    - Temporal features (hourly, daily, weekly, seasonal patterns)
    - Weather-based features
    - External event features
    - Lag features
    - Rolling statistics
    - Fourier features for seasonality
    """
    
    def __init__(self, country: str = 'FR', include_weather: bool = True):
        """
        Initialize the feature engineer.
        
        Args:
            country: Country code for holidays
            include_weather: Whether to include weather-based features
        """
        self.country = country
        self.include_weather = include_weather
        self.holidays = holidays.CountryHoliday(country)
        
    def create_temporal_features(self, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Create advanced temporal features.
        
        Args:
            timestamps: Datetime index for the data
            
        Returns:
            DataFrame with temporal features
        """
        features = pd.DataFrame(index=timestamps)
        
        # Basic temporal features
        features['hour'] = timestamps.hour
        features['day_of_week'] = timestamps.dayofweek
        features['day_of_month'] = timestamps.day
        features['month'] = timestamps.month
        features['year'] = timestamps.year
        features['quarter'] = timestamps.quarter
        features['week_of_year'] = timestamps.isocalendar().week
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['quarter_sin'] = np.sin(2 * np.pi * features['quarter'] / 4)
        features['quarter_cos'] = np.cos(2 * np.pi * features['quarter'] / 4)
        
        # Business/weekend indicators
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        features['is_business_day'] = (~features.index.isin(self.holidays) & 
                                      (features['is_weekend'] == 0)).astype(int)
        
        # Holiday indicators
        features['is_holiday'] = features.index.isin(self.holidays).astype(int)
        
        # Special periods
        features['is_summer'] = features['month'].isin([6, 7, 8]).astype(int)
        features['is_winter'] = features['month'].isin([12, 1, 2]).astype(int)
        features['is_spring'] = features['month'].isin([3, 4, 5]).astype(int)
        features['is_autumn'] = features['month'].isin([9, 10, 11]).astype(int)
        
        # Day type indicators
        features['is_morning'] = features['hour'].between(6, 11).astype(int)
        features['is_afternoon'] = features['hour'].between(12, 17).astype(int)
        features['is_evening'] = features['hour'].between(18, 23).astype(int)
        features['is_night'] = features['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
        
        # Weekend/holiday combinations
        features['is_weekend_holiday'] = (features['is_weekend'] & features['is_holiday']).astype(int)
        features['is_weekday_holiday'] = ((~features['is_weekend']) & features['is_holiday']).astype(int)
        
        return features
    
    def create_fourier_features(self, timestamps: pd.DatetimeIndex, 
                               periods: List[int] = [24, 168, 8760], 
                               order: int = 3) -> pd.DataFrame:
        """
        Create Fourier features for capturing seasonality.
        
        Args:
            timestamps: Datetime index
            periods: List of periods to capture (daily, weekly, yearly)
            order: Order of Fourier terms
            
        Returns:
            DataFrame with Fourier features
        """
        features = pd.DataFrame(index=timestamps)
        
        # Convert to hours since start
        hours_since_start = (timestamps - timestamps[0]).total_seconds() / 3600
        
        for period in periods:
            for i in range(1, order + 1):
                features[f'fourier_sin_{period}_{i}'] = np.sin(2 * np.pi * i * hours_since_start / period)
                features[f'fourier_cos_{period}_{i}'] = np.cos(2 * np.pi * i * hours_since_start / period)
        
        return features
    
    def create_lag_features(self, data: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """
        Create lag features for time series.
        
        Args:
            data: Time series data
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        features = pd.DataFrame(index=data.index)
        
        for lag in lags:
            features[f'lag_{lag}'] = data.shift(lag)
            
        return features
    
    def create_rolling_features(self, data: pd.DataFrame, windows: List[int], 
                              functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            data: Time series data
            windows: List of window sizes
            functions: List of aggregation functions
            
        Returns:
            DataFrame with rolling features
        """
        features = pd.DataFrame(index=data.index)
        
        for window in windows:
            for func in functions:
                col_name = f'rolling_{func}_{window}'
                if func == 'mean':
                    features[col_name] = data.rolling(window=window).mean()
                elif func == 'std':
                    features[col_name] = data.rolling(window=window).std()
                elif func == 'min':
                    features[col_name] = data.rolling(window=window).min()
                elif func == 'max':
                    features[col_name] = data.rolling(window=window).max()
                elif func == 'median':
                    features[col_name] = data.rolling(window=window).median()
                
                # Forward fill NaN values
                features[col_name] = features[col_name].ffill()
        
        return features
    
    def create_diff_features(self, data: pd.DataFrame, periods: List[int] = [1, 24, 168]) -> pd.DataFrame:
        """
        Create difference features.
        
        Args:
            data: Time series data
            periods: List of differencing periods
            
        Returns:
            DataFrame with difference features
        """
        features = pd.DataFrame(index=data.index)
        
        for period in periods:
            features[f'diff_{period}'] = data.diff(period)
            
        return features
    
    def create_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between existing features.
        
        Args:
            features: DataFrame with existing features
            
        Returns:
            DataFrame with interaction features
        """
        interaction_features = pd.DataFrame(index=features.index)
        
        # Time-based interactions
        interaction_features['hour_weekday'] = features['hour'] * features['day_of_week']
        interaction_features['hour_month'] = features['hour'] * features['month']
        interaction_features['hour_business_day'] = features['hour'] * features['is_business_day']
        
        # Seasonal interactions
        interaction_features['hour_summer'] = features['hour'] * features['is_summer']
        interaction_features['hour_winter'] = features['hour'] * features['is_winter']
        
        # Weekend interactions
        interaction_features['weekend_hour'] = features['is_weekend'] * features['hour']
        
        return interaction_features
    
    def create_all_features(self, data: pd.DataFrame, timestamps: pd.DatetimeIndex,
                           target_columns: List[str] = ['net_load', 'price'],
                           lags: List[int] = [1, 2, 3, 24, 48, 168],
                           rolling_windows: List[int] = [3, 6, 12, 24, 48, 168]) -> pd.DataFrame:
        """
        Create all advanced features.
        
        Args:
            data: DataFrame with time series data
            timestamps: Datetime index
            target_columns: List of target column names
            lags: List of lag periods
            rolling_windows: List of rolling window sizes
            
        Returns:
            DataFrame with all features
        """
        all_features = pd.DataFrame(index=timestamps)
        
        # Temporal features
        temporal_features = self.create_temporal_features(timestamps)
        all_features = pd.concat([all_features, temporal_features], axis=1)
        
        # Fourier features
        fourier_features = self.create_fourier_features(timestamps)
        all_features = pd.concat([all_features, fourier_features], axis=1)
        
        # Target-based features
        for target in target_columns:
            if target in data.columns:
                # Lag features
                lag_features = self.create_lag_features(data[target], lags)
                all_features = pd.concat([all_features, lag_features.add_suffix(f'_{target}')], axis=1)
                
                # Rolling features
                rolling_features = self.create_rolling_features(data[target], rolling_windows)
                all_features = pd.concat([all_features, rolling_features.add_suffix(f'_{target}')], axis=1)
                
                # Difference features
                diff_features = self.create_diff_features(data[target])
                all_features = pd.concat([all_features, diff_features.add_suffix(f'_{target}')], axis=1)
        
        # Interaction features
        interaction_features = self.create_interaction_features(all_features)
        all_features = pd.concat([all_features, interaction_features], axis=1)
        
        return all_features
    
    def create_weather_features(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-based features.
        
        Args:
            weather_data: DataFrame with weather data
            
        Returns:
            DataFrame with weather features
        """
        if not self.include_weather:
            return pd.DataFrame()
            
        features = pd.DataFrame(index=weather_data.index)
        
        # Basic weather features
        if 'temperature' in weather_data.columns:
            features['temp'] = weather_data['temperature']
            features['temp_sin'] = np.sin(2 * np.pi * features['temp'] / 30)  # Monthly cycle
            features['temp_cos'] = np.cos(2 * np.pi * features['temp'] / 30)
            
            # Temperature categories
            features['is_hot'] = (features['temp'] > 25).astype(int)
            features['is_cold'] = (features['temp'] < 10).astype(int)
            features['is_mild'] = ((features['temp'] >= 10) & (features['temp'] <= 25)).astype(int)
        
        # Humidity features
        if 'humidity' in weather_data.columns:
            features['humidity'] = weather_data['humidity']
            features['is_humid'] = (features['humidity'] > 70).astype(int)
            features['is_dry'] = (features['humidity'] < 30).astype(int)
        
        # Wind features
        if 'wind_speed' in weather_data.columns:
            features['wind_speed'] = weather_data['wind_speed']
            features['is_windy'] = (features['wind_speed'] > 10).astype(int)
            features['is_calm'] = (features['wind_speed'] < 2).astype(int)
        
        # Solar features
        if 'solar_radiation' in weather_data.columns:
            features['solar_radiation'] = weather_data['solar_radiation']
            features['is_sunny'] = (features['solar_radiation'] > 100).astype(int)
            features['is_cloudy'] = (features['solar_radiation'] < 50).astype(int)
        
        # Weather combinations
        if 'temperature' in features.columns and 'humidity' in features.columns:
            features['heat_index'] = features['temp'] + 0.5 * features['humidity']
        
        return features


def create_advanced_features(data: pd.DataFrame, timestamps: pd.DatetimeIndex,
                           target_columns: List[str] = ['net_load', 'price'],
                           weather_data: Optional[pd.DataFrame] = None,
                           country: str = 'FR') -> pd.DataFrame:
    """
    Convenience function to create advanced features.
    
    Args:
        data: DataFrame with time series data
        timestamps: Datetime index
        target_columns: List of target column names
        weather_data: Optional DataFrame with weather data
        country: Country code for holidays
        
    Returns:
        DataFrame with all advanced features
    """
    engineer = AdvancedFeatureEngineer(country=country, 
                                     include_weather=weather_data is not None)
    
    # Create all features
    features = engineer.create_all_features(data, timestamps, target_columns)
    
    # Add weather features if available
    if weather_data is not None:
        weather_features = engineer.create_weather_features(weather_data)
        features = pd.concat([features, weather_features], axis=1)
    
    return features