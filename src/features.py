import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features...")
    # 1. Categorical Encoding (Congestion)
    if 'congestion_level' in df.columns:
        congestion_map = {'unknown': 0, 'free-flow': 1, 'moderate': 2, 'heavy': 3, 'severe': 4}
        df['congestion_level'] = df['congestion_level'].astype(str).str.lower().map(congestion_map).fillna(0).astype(int)
        
    # 2. Categorical Encoding (Weather)
    if 'weather_condition' in df.columns:
        df = pd.get_dummies(df, columns=['weather_condition'], prefix='weather')
        
    # 3. Time-window features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
    # 4. Ratios
    if 'traffic_volume' in df.columns and 'road_capacity' in df.columns:
        df['volume_capacity_ratio'] = df['traffic_volume'] / df['road_capacity'].replace(0, np.nan)
        
    # 5. Flags
    if 'rainfall' in df.columns:
        df['is_raining'] = (df['rainfall'] > 0).astype(int)
    if 'visibility' in df.columns:
        df['low_visibility'] = (df['visibility'] < 1000).astype(int)
    if 'public_holiday' in df.columns:
        df['is_holiday'] = (df['public_holiday'] > 0).astype(int)
        
    # 6. Lag and Rolling Window Statistics
    if 'road_id' in df.columns and 'timestamp' in df.columns:
        df = df.sort_values(['road_id', 'timestamp'])
        # 1. Lag Features (t-1, t-2, t-48 for volume and speed)
        for lag in [1, 2, 48]:
            if 'traffic_volume' in df.columns:
                df[f'volume_lag_{lag}'] = df.groupby('road_id')['traffic_volume'].shift(lag)
            if 'avg_speed' in df.columns:
                df[f'speed_lag_{lag}'] = df.groupby('road_id')['avg_speed'].shift(lag)
            
        # 2. Rolling Statistics (Mean and Std for windows 4 and 8)
        for window in [4, 8]:
            for col in ['traffic_volume', 'avg_speed']:
                if col in df.columns:
                # Rolling Mean
                    df[f'{col}_rolling_mean_{window}'] = (
                    df.groupby('road_id')[col]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                )
                # Rolling Standard Deviation
                df[f'{col}_rolling_std_{window}'] = (
                    df.groupby('road_id')[col]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).std())
                )
    return df
def missing_values_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Removing missing values...")
    
    # Drop any rows that contain missing values
    df = df.dropna()
    
    # Optional: Reset the index after dropping rows
    df = df.reset_index(drop=True)
    
    return df


