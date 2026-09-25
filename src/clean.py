from pydantic import annotated_handlers
import os
import pandas as pd
import numpy as np
import json
from ingest import load_datasets, tz_parsing, validate_sensor_data, validate_calendar_data, validate_weather_data


def process_data():
    print("Loading datasets...")
    calendar, weather, sensors = load_datasets()

    print("Validating raw data schemas...")
    calendar = validate_calendar_data(calendar)
    weather = validate_weather_data(weather)
    sensors = validate_sensor_data(sensors)

    print("Standardizing temporal fields...")
    # Parse timestamps
    calendar = tz_parsing(calendar, 'date', None)
    weather = tz_parsing(weather, 'date', 'time')
    sensors = tz_parsing(sensors, 'date', 'time')
    
    # --- CRITICAL FIX: Force all timestamp columns to actual datetime type ---
    sensors['timestamp'] = pd.to_datetime(sensors['timestamp'], errors='coerce')
    weather['timestamp'] = pd.to_datetime(weather['timestamp'], errors='coerce')
    calendar['timestamp'] = pd.to_datetime(calendar['timestamp'], errors='coerce')

    # Ensure sensor timestamps are strictly 30-minute intervals
    sensors = normalised_timestamp(sensors)

    print("Cleaning sensor data...")
    # Duplicate removal
    sensors = sensors.drop_duplicates(subset=['road_id', 'timestamp'], keep='first')

    # Outlier Detection (Rules)
    # Cap negative volumes to 0, speeds > 200 to NaN, occupancy > 100 to 100
    sensors['traffic_volume'] = sensors['traffic_volume'].clip(lower=0)
    sensors.loc[sensors['avg_speed'] > 200, 'avg_speed'] = np.nan
    sensors['occupancy'] = sensors['occupancy'].clip(upper=100)
    
    # Statistical Outlier Detection (IQR)
    def iqr_cap(s):
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        return s.clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
        
    for col in ['traffic_volume', 'avg_speed']:
        if col in sensors.columns:
            sensors[col] = sensors.groupby('road_id')[col].transform(iqr_cap)

    # Missing value handling
    # Sort for interpolation
    sensors = sensors.sort_values(by=['road_id', 'timestamp'])
    # Using apply instead of transform with lambda often hits faster Pandas code paths
    sensors['traffic_volume'] = sensors.groupby('road_id')['traffic_volume'].apply(lambda x: x.interpolate(method='linear', limit=2)).reset_index(level=0, drop=True)
    sensors['avg_speed'] = sensors.groupby('road_id')['avg_speed'].apply(lambda x: x.interpolate(method='linear', limit=2)).reset_index(level=0, drop=True)
    
    # Fill remaining with medians if necessary
    sensors['traffic_volume'] = sensors['traffic_volume'].fillna(sensors['traffic_volume'].median())
    
    # Missing Congestion Level
    sensors['congestion_level'] = sensors['congestion_level'].fillna('Unknown')

    print("Cleaning weather data...")
    # Categorical harmonization
    if 'weather_condition' in weather.columns:
        weather['weather_condition'] = weather['weather_condition'].str.lower().str.strip()

    print("Aligning and Merging datasets...")
    # Align weather (hourly) to sensors (30-min)
    sensors['hour_timestamp'] = sensors['timestamp'].dt.floor('h')
    
    if 'station_id' in weather.columns and 'weather_station_id' in sensors.columns:
        merged = pd.merge(sensors, weather, left_on=['weather_station_id', 'hour_timestamp'], right_on=['station_id', 'timestamp'], how='left', suffixes=('', '_weather'))
    else:
        merged = pd.merge(sensors, weather, left_on='hour_timestamp', right_on='timestamp', how='left', suffixes=('', '_weather'))
    
    # Drop intermediate columns
    if 'timestamp_weather' in merged.columns:
        merged.drop(columns=['timestamp_weather', 'hour_timestamp'], inplace=True)

    # Merge calendar (daily)
    merged['day_timestamp'] = merged['timestamp'].dt.floor('D')
    merged = pd.merge(merged, calendar, left_on='day_timestamp', right_on='timestamp', how='left', suffixes=('', '_cal'))
    if 'timestamp_cal' in merged.columns:
        merged.drop(columns=['timestamp_cal', 'day_timestamp'], inplace=True)

    # Feature Engineering: JSON expansion
    print("Expanding vehicle_type_dist JSON...")
    if 'vehicle_type_dist' in merged.columns:
        parsed_dicts = [
            json.loads(x) if isinstance(x, str) else {} 
            for x in merged['vehicle_type_dist']
        ]
        json_df = pd.DataFrame(parsed_dicts, index=merged.index)
        merged = pd.concat([merged, json_df], axis=1)
        merged.drop(columns=['vehicle_type_dist'], inplace=True)

    print("Handling comprehensive missing values and harmonizing categories...")
    merged = handle_missing_values(merged)

    # Data Validation & Quality Checks
    print("Running Data Quality Validation...")
    validate_data_quality(merged)

    # Export
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'interim')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'processed_dataset.csv')
    
    print(f"Saving processed dataset to {out_path}...")
    merged.to_csv(out_path, index=False)
    print(f"Data processing complete. Final shape: {merged.shape}")

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    numerical_columns = ["temperature", "rainfall", "visibility", "occupancy", "avg_speed"]
    
    categorical_fill_values = {
        "holiday_name": "No Holiday",
        "event_name": "No Event"
    }
    for col in numerical_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())

    for col, default_val in categorical_fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val)
    
    if 'weather_condition' not in df.columns:
        return df

    df['weather_condition'] = df['weather_condition'].astype(str).str.strip().str.lower()

    mapping = {
        'rainy': 'rain',
        'foggy': 'fog',
        'nan': 'unknown',
        'none': 'unknown'
    }
    df['weather_condition'] = df['weather_condition'].replace(mapping)
    df['weather_condition'] = df['weather_condition'].fillna('unknown')

    if "station_id" in df.columns:
        df.drop(columns=["station_id"], inplace=True)
        
    return df

def normalised_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure column is datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Assert that no rows failed datetime conversion
    if df['timestamp'].isna().any():
        n_invalid = df['timestamp'].isna().sum()
        raise ValueError(f"Datetime conversion failed: Found {n_invalid} invalid or unparseable timestamps.")

    # Ensure timestamp interval is rounded to 30 minutes
    df['timestamp'] = df['timestamp'].dt.round('30min')
    return df

def validate_data_quality(df: pd.DataFrame):
    print("Running Data Quality Checks...")
    
    # 1. Verify row counts
    if len(df) == 0:
        raise ValueError("Data validation failed: The processed dataframe is empty.")
        
    # 1b. Strictly verify that timestamp column exists and is of datetime type
    if 'timestamp' not in df.columns:
        raise ValueError("Data validation failed: 'timestamp' column is missing.")
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        raise TypeError(f"Data validation failed: 'timestamp' is of type {df['timestamp'].dtype}, expected datetime.")

    # 2. Assert no nulls in critical modeling columns
    critical_cols = ['road_id', 'timestamp', 'traffic_volume', 'avg_speed', 'occupancy']
    missing_critical = df[critical_cols].isnull().sum()
    if missing_critical.sum() > 0:
        print("WARNING: Null values found in critical columns:")
        print(missing_critical[missing_critical > 0])
        
    # 3. Confirm value ranges
    if (df['traffic_volume'] < 0).any():
        raise ValueError("Data validation failed: Negative traffic volume detected.")
    if (df['occupancy'] < 0).any() or (df['occupancy'] > 100).any():
        raise ValueError("Data validation failed: Occupancy out of bounds (0-100).")
        
    print("Data quality checks passed successfully!")


if __name__ == "__main__":
    process_data()