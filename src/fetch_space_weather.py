import pandas as pd
import requests
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
SW_FILE = os.path.join(DATA_DIR, 'space_weather.csv')
SW_URL = "https://celestrak.org/SpaceData/SW-All.csv"

def fetch_space_weather():
    print(f"Downloading Space Weather data from {SW_URL}...")
    try:
        df = pd.read_csv(SW_URL)
        
        # Standardize column names if needed
        # CelesTrak columns are usually: DATE, RADIO_FLUX_10_7, AP_AVG, KP_SUM, etc.
        # Let's inspect/clean basic stuff
        
        # Save raw data
        df.to_csv(SW_FILE, index=False)
        print(f"Successfully saved {len(df)} records to {SW_FILE}")
        
        print("\nFirst 5 rows:")
        print(df.head())
        
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    fetch_space_weather()
