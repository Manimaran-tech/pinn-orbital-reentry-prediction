import os
import sys
import pandas as pd
from spacetrack import SpaceTrackClient
# from spacetrack import operators # Not used directly
import datetime
import json

# Attempt to load credentials from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

USER = os.getenv('SPACETRACK_USER')
PASSWORD = os.getenv('SPACETRACK_PASSWORD')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
TLE_OUTPUT_DIR = os.path.join(DATA_DIR, 'tles')

if not os.path.exists(TLE_OUTPUT_DIR):
    os.makedirs(TLE_OUTPUT_DIR)

def get_client():
    if not USER or not PASSWORD:
        print("Error: SPACETRACK_USER and SPACETRACK_PASSWORD environment variables are not set.")
        print("Please create a .env file or export these variables.")
        sys.exit(1)
    return SpaceTrackClient(identity=USER, password=PASSWORD)

def fetch_decayed_rocket_bodies(limit=5):
    client = get_client()
    
    # DEBUG: Check if we can fetch ANYTHING (e.g. valid auth)
    print("DEBUG: Testing connection with ISS (ID 25544)...")
    try:
        logging_test = client.tle(norad_cat_id=25544, orderby='epoch desc', limit=1, format='json')
        res = list(logging_test)
        if res:
             # Just checking if we got data.
             pass
    except Exception as e:
        print(f"DEBUG: Auth/Connection failed: {e}")
        return

    print("Fetching list of decayed rocket bodies...")
    
    # Try getting ANY decayed object (top `limit` most recent)
    try:
        satcat_data = client.satcat(
            orderby='decay desc',
            limit=limit,
            decay='>2024-01-01', 
            object_type='2' # Rocket Body
        )
        print(f"Query returned {len(satcat_data)} objects.")
    except Exception as e:
        print(f"Query failed: {e}")
        satcat_data = []

    if not satcat_data:
        print("Detailed debug: trying without object_type filter...")
        try:
            satcat_data = client.satcat(orderby='decay desc', limit=limit, decay='>2024-01-01')
            print(f"Broad query returned {len(satcat_data)} objects.")
        except Exception as e:
            print(f"Broad query failed: {e}")

    # For each rocket body, fetch full TLE history
    for sat in satcat_data:
        norad_id = sat['NORAD_CAT_ID']
        decay_date = sat['DECAY']
        print(f"Fetching TLEs for Object {norad_id} (Decayed: {decay_date})...")
        
        try:
            tles_gen = client.tle(
                norad_cat_id=norad_id,
                orderby='EPOCH asc',
                format='json'
            )
            
            # Collapse generator to string then parse
            tles_str = "".join(list(tles_gen))
            if not tles_str:
                print(f"No TLE data found for {norad_id}")
                continue
                
            data = json.loads(tles_str)
            
            # Save to CSV
            df = pd.DataFrame(data)
            filename = os.path.join(TLE_OUTPUT_DIR, f"{norad_id}.csv")
            df.to_csv(filename, index=False)
            print(f"Saved {len(df)} TLEs to {filename}")
        except Exception as e:
            print(f"Failed to fetch TLEs for {norad_id}: {e}")

if __name__ == "__main__":
    fetch_decayed_rocket_bodies(limit=5)
