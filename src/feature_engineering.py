import pandas as pd
import numpy as np
import os
from sgp4.api import Satrec
from sgp4.earth_gravity import wgs84

# Constants
MU = 3.986004418e14  # Earth gravitational parameter (m^3/s^2)
RE = 6378.137        # Earth radius (km) for altitude calc
DAY_SEC = 86400.0

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
TLE_DIR = os.path.join(DATA_DIR, 'tles')
SW_FILE = os.path.join(DATA_DIR, 'space_weather.csv')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

def load_space_weather():
    if not os.path.exists(SW_FILE):
        raise FileNotFoundError(f"Space weather file not found at {SW_FILE}")
    
    df = pd.read_csv(SW_FILE)
    # Parse dates
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Select relevant columns. CelesTrak SW-All.csv usually has:
    # DATE, RADIO_FLUX_10_7, AP_AVG, KP_SUM (or similar mean)
    # We'll need to check the exact columns from the downloaded file.
    # For now, assuming some standard names or renaming them.
    # Let's clean up column names to be safe (strip spaces)
    df.columns = df.columns.str.strip()
    
    # Create a simplified dataframe
    # We want: F10.7_OBS, AP_AVG, KP_SUM
    cols_map = {
        'F10.7_OBS': 'F10_7',
        'AP_AVG': 'Ap',
        'KP_SUM': 'Kp' # Kp is usually sum of 8 3-hour indices. Divide by 8 or use as is?
                       # Kp indices are often 0-9. Sum is 0-72. Let's use AP_AVG which is linear.
                       # Actually, let's keep AP and F10.7 as primary.
    }
    
    # Fallback if specific columns missing (CelesTrak changes sometimes)
    available_cols = [c for c in cols_map.keys() if c in df.columns]
    sw_df = df[['DATE'] + available_cols].rename(columns=cols_map)
    return sw_df

def process_tle_file(filepath):
    df = pd.read_csv(filepath)
    
    # Expect columns from spacetrack JSON/CSV: EPOCH, MEAN_MOTION, ECCENTRICITY, INCLINATION, RA_OF_ASC_NODE, ARG_OF_PERICENTER, MEAN_ANOMALY, BSTAR, NORAD_CAT_ID
    # Standardize column names
    df.columns = df.columns.str.upper().str.strip()
    
    df['EPOCH'] = pd.to_datetime(df['EPOCH'])
    df = df.sort_values('EPOCH')
    
    features = []
    
    for idx, row in df.iterrows():
        # Physics calculations
        n_rev_per_day = row['MEAN_MOTION']
        ecc = row['ECCENTRICITY']
        inc = row['INCLINATION']
        bstar = row['BSTAR']
        
        # Calculate Semi-Major Axis (a)
        # n is in rev/day. Convert to rad/s
        n_rad_s = n_rev_per_day * (2 * np.pi) / DAY_SEC
        if n_rad_s <= 0: continue
        
        a_m = (MU / (n_rad_s ** 2)) ** (1/3) # in meters
        a_km = a_m / 1000.0
        
        # Calculate Perigee/Apogee Altitude
        # r_p = a(1-e), r_a = a(1+e)
        r_p = a_km * (1 - ecc)
        r_a = a_km * (1 + ecc)
        
        alt_p = r_p - RE
        alt_a = r_a - RE
        alt_mean = a_km - RE
        
        features.append({
            'EPOCH': row['EPOCH'],
            'NORAD_ID': row['NORAD_CAT_ID'],
            'a_km': a_km,
            'e': ecc,
            'i': inc,
            'mean_motion': n_rev_per_day,
            'bstar': bstar,
            'alt_perigee': alt_p,
            'alt_apogee': alt_a,
            'alt_mean': alt_mean
        })
        
    feat_df = pd.DataFrame(features)
    
    # Calculate derivatives (Decay Rate)
    # altitude decay rate (km/day)
    # diff() gives change between rows. We need to divide by time diff in days.
    
    feat_df['dt_days'] = feat_df['EPOCH'].diff().dt.total_seconds() / DAY_SEC
    feat_df['d_alt_mean'] = feat_df['alt_mean'].diff()
    
    feat_df['decay_rate'] = feat_df['d_alt_mean'] / feat_df['dt_days']
    
    # Filter extrema (initial noise)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return feat_df

def create_dataset():
    sw_df = load_space_weather()
    
    all_objects_data = []
    
    tle_files = [f for f in os.listdir(TLE_DIR) if f.endswith('.csv')]
    print(f"Processing {len(tle_files)} TLE files...")
    
    for t_file in tle_files:
        try:
            f_path = os.path.join(TLE_DIR, t_file)
            obj_df = process_tle_file(f_path)
            
            # Merge Space Weather
            # Match EPOCH to nearest DATE in sw_df
            # sw_df is daily. We can use merge_asof if sorted.
            
            obj_df = obj_df.sort_values('EPOCH')
            sw_df = sw_df.sort_values('DATE')
            
            merged = pd.merge_asof(obj_df, sw_df, left_on='EPOCH', right_on='DATE', direction='backward')
            
            # Add Lags (can be done here or in training loop, usually better here)
            # But here we have rows as TLE epochs, which are irregular.
            # Ideally, we should interpolate SW to TLE times, using standard 1-day lag logic.
            # Using 'backward' search effectively gives us the SW data from the day of (or before) the TLE.
            
            # Calculate Time-to-Reentry
            # The last TLE is roughly reentry (or close to it).
            reentry_epoch = obj_df['EPOCH'].iloc[-1]
            merged['reentry_epoch'] = reentry_epoch
            merged['days_to_reentry'] = (merged['reentry_epoch'] - merged['EPOCH']).dt.total_seconds() / DAY_SEC
            
            all_objects_data.append(merged)
            print(f"Processed {t_file}: {len(merged)} records")
            
        except Exception as e:
            print(f"Error processing {t_file}: {e}")
            
    if all_objects_data:
        final_df = pd.concat(all_objects_data, ignore_index=True)
        out_path = os.path.join(PROCESSED_DIR, 'dataset_v1.csv')
        final_df.to_csv(out_path, index=False)
        print(f"Saved merged dataset to {out_path} with {len(final_df)} rows.")
    else:
        print("No data processed.")

if __name__ == "__main__":
    create_dataset()
