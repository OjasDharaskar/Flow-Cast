import pandas as pd
import pandera.pandas as pa
import os

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_datasets():
    root = get_project_root()
    calendar = pd.read_csv(os.path.join(root, "data", "Raw", "calendar_events.csv"))
    weather = pd.read_csv(os.path.join(root, "data", "Raw", "weather_observations.csv"))
    sensors = pd.read_csv(os.path.join(root, "data", "Raw", "traffic_sensor_log.csv"))

    return calendar, weather, sensors

def tz_parsing(df: pd.DataFrame, date_col: str, time_col: str = None, target_tz: str = 'Asia/Kolkata') -> pd.DataFrame:
    # Safely combine date and time strings, filling missing values to prevent concatenation errors
    if time_col and time_col in df.columns:
        raw_series = df[date_col].astype(str).str.strip() + ' ' + df[time_col].astype(str).str.strip()
        # Clean up cases where time or date was 'nan' or 'NaT'
        raw_series = raw_series.replace(['nan nan', 'NaT NaT', 'nan', 'NaT'], pd.NA)
        df.drop(columns=[date_col, time_col], inplace=True)
    else:
        raw_series = df[date_col].astype(str).str.strip()
        raw_series = raw_series.replace(['nan', 'NaT'], pd.NA)
        df.drop(columns=[date_col], inplace=True)
        
    # Parse to datetime safely using errors='coerce' so bad rows become NaT instead of crashing
    try:
        combined_date_time = pd.to_datetime(raw_series, format='mixed', errors='coerce')
    except (TypeError, ValueError):
        # Fallback for older Pandas versions (< 2.0) that don't support format='mixed'
        combined_date_time = pd.to_datetime(raw_series, errors='coerce')
        
    # Localize (if it's not already tz-aware)
    if getattr(combined_date_time.dt, 'tz', None) is None:
        # Localize only non-NaT values to prevent timezone localization errors on missing data
        localized = combined_date_time.dt.tz_localize(target_tz, ambiguous='NaT', nonexistent='NaT')
        df['timestamp'] = localized
    else:
        df['timestamp'] = combined_date_time.dt.tz_convert(target_tz)
        
    return df

def validate_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    schema = pa.DataFrameSchema(
        columns={
            'road_id': pa.Column(str),
            'road_name': pa.Column(str, nullable=True),
            'latitude': pa.Column(float, nullable=True),
            'longitude': pa.Column(float, nullable=True),
            'weather_station_id': pa.Column(str, nullable=True),
            'date': pa.Column(str),
            'time': pa.Column(str),
            'traffic_volume': pa.Column(float, nullable=True),
            'vehicle_count': pa.Column(int, nullable=True),
            'vehicle_type_dist': pa.Column(str, nullable=True),
            'avg_speed': pa.Column(float, nullable=True),
            'occupancy': pa.Column(float, nullable=True),
            'congestion_level': pa.Column(str, nullable=True),
            'travel_time': pa.Column(float, nullable=True),
            'accident_count': pa.Column(int, nullable=True),
            'signal_timing': pa.Column(int, nullable=True),
            'road_capacity': pa.Column(int, nullable=True),
        },
        coerce=True
    )
    return schema.validate(df)

def validate_calendar_data(df: pd.DataFrame) -> pd.DataFrame:
    schema = pa.DataFrameSchema(
        columns={
            'date': pa.Column(str),
            'public_holiday': pa.Column(int, nullable=True),
            'event_flag': pa.Column(int, nullable=True),
            'event_name': pa.Column(str, nullable=True),
            'roadwork_flag': pa.Column(int, nullable=True)
        }, 
        coerce=True
    )
    return schema.validate(df)

def validate_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    schema = pa.DataFrameSchema(
        columns={
            'station_id': pa.Column(str, nullable=True),
            'date': pa.Column(str),
            'time': pa.Column(str),
            'weather_condition': pa.Column(str, nullable=True),
            'temperature': pa.Column(float, nullable=True),
            'rainfall': pa.Column(float, nullable=True),
            'visibility': pa.Column(float, nullable=True)
        },
        coerce=True
    )
    return schema.validate(df)