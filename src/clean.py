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

    print("Cleaning sensor data...")
    # Duplicate removal
    sensors = sensors.drop_duplicates(subset=['road_id', 'timestamp'], keep='first')

    # Outlier Detection (Rules)
    # Cap negative volumes to 0, speeds > 200 to NaN, occupancy > 100 to 100
    sensors['traffic_volume'] = sensors['traffic_volume'].clip(lower=0)
    sensors.loc[sensors['avg_speed'] > 200, 'avg_speed'] = np.nan
    sensors['occupancy'] = sensors['occupancy'].clip(upper=100)

    # Missing value handling
    # Sort for interpolation
    sensors = sensors.sort_values(by=['road_id', 'timestamp'])
    # Using apply instead of transform with lambda often hits faster Pandas code paths
    sensors['traffic_volume'] = sensors.groupby('road_id')['traffic_volume'].apply(lambda x: x.interpolate(method='linear', limit=2)).reset_index(level=0, drop=True)
    sensors['avg_speed'] = sensors.groupby('road_id')['avg_speed'].apply(lambda x: x.interpolate(method='linear', limit=2)).reset_index(level=0, drop=True)
    
    # Fill remaining with medians if necessary
    sensors['traffic_volume'] = sensors['traffic_volume'].fillna(sensors['traffic_volume'].median())
    
    # Missing Congestion Level
    # Derive from occupancy/volume/capacity logic. Just placeholder if missing:
    sensors['congestion_level'] = sensors['congestion_level'].fillna('Unknown')

    print("Cleaning weather data...")
    # Categorical harmonization
    if 'weather_condition' in weather.columns:
        weather['weather_condition'] = weather['weather_condition'].str.lower().str.strip()

    print("Aligning and Merging datasets...")
    # Align weather (hourly) to sensors (30-min)
    # Forward fill hourly data onto 30 min intervals by joining on time
    # Round sensor timestamp to floor hour to join with weather
    sensors['hour_timestamp'] = sensors['timestamp'].dt.floor('h')
    
    # Weather might not have road_id, just timestamp (or station_id).
    # Assuming weather maps globally if station_id not joined, or left merge on timestamp and station_id
    if 'station_id' in weather.columns and 'weather_station_id' in sensors.columns:
        merged = pd.merge(sensors, weather, left_on=['weather_station_id', 'hour_timestamp'], right_on=['station_id', 'timestamp'], how='left', suffixes=('', '_weather'))
    else:
        # Fallback to just time if no stations
        merged = pd.merge(sensors, weather, left_on='hour_timestamp', right_on='timestamp', how='left', suffixes=('', '_weather'))
    
    # Drop intermediate columns
    if 'timestamp_weather' in merged.columns:
        merged.drop(columns=['timestamp_weather', 'hour_timestamp'], inplace=True)

    # Merge calendar (daily)
    # Floor timestamp to day
    merged['day_timestamp'] = merged['timestamp'].dt.floor('D')
    merged = pd.merge(merged, calendar, left_on='day_timestamp', right_on='timestamp', how='left', suffixes=('', '_cal'))
    if 'timestamp_cal' in merged.columns:
        merged.drop(columns=['timestamp_cal', 'day_timestamp'], inplace=True)

    # Feature Engineering: JSON expansion
    print("Expanding vehicle_type_dist JSON...")
    if 'vehicle_type_dist' in merged.columns:
        # Optimized JSON parsing using list comprehension
        parsed_dicts = [
            json.loads(x) if isinstance(x, str) else {} 
            for x in merged['vehicle_type_dist']
        ]
        json_df = pd.DataFrame(parsed_dicts, index=merged.index)
        
        merged = pd.concat([merged, json_df], axis=1)
        merged.drop(columns=['vehicle_type_dist'], inplace=True)

    # Export
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'interim')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'processed_dataset.csv')
    
    print(f"Saving processed dataset to {out_path}...")
    merged.to_csv(out_path, index=False)
    print(f"Data processing complete. Final shape: {merged.shape}")

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Corrected spelling for temperature
    numerical_columns = ["temperature", "rainfall", "visibility", "occupancy","avg_speed"]
    
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
    
    #Handling weather condition column
    if 'weather_condition' not in df.columns:
        return df

    # 1. Clean whitespace and casing
    df['weather_condition'] = df['weather_condition'].astype(str).str.strip().str.lower()

    # 2. Harmonize synonymous categories
    mapping = {
        'rainy': 'rain',
        'foggy': 'fog',
        'nan': 'unknown',
        'none': 'unknown'
    }
    df['weather_condition'] = df['weather_condition'].replace(mapping)

    # 3. Fill remaining nulls
    df['weather_condition'] = df['weather_condition'].fillna('unknown')

    if "station_id" in df.columns:
        df.drop(columns=["station_id"],inplace=True)
    return df

def normalised_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    # Convert timestamp to datetime, handling any formatting issues
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Ensure timestamp interval is 30 minutes
    df['timestamp'] = df['timestamp'].dt.round('30min')
    return df

if __name__ == "__main__":
    process_data()