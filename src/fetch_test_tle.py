import os
import sys
import pandas as pd
import json
from spacetrack import SpaceTrackClient

# Load env for credentials
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

USER = os.getenv('SPACETRACK_USER')
PASSWORD = os.getenv('SPACETRACK_PASSWORD')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def fetch_fresh_object(target_id):
    if not target_id:
        print("Error: No Object ID provided.")
        return None

    if not USER or not PASSWORD:
        print("Credentials missing.")
        return None

    client = SpaceTrackClient(identity=USER, password=PASSWORD)
    print(f"Fetching full TLE history for {target_id}...")
    
    try:
        tles_gen = client.tle(
            norad_cat_id=target_id,
            orderby='EPOCH asc',
            format='json'
        )
        
        tles_list = list(tles_gen)
        
        if not tles_list:
            print(f"Error: No TLEs found for object {target_id}. Check if ID is correct or object is not tracked.")
            return None

        # Handle string response vs list of dicts response
        if isinstance(tles_list[0], str):
            tles_str = "".join(tles_list)
            data = json.loads(tles_str)
        else:
            data = tles_list
            
        if not data:
             print(f"Error: Parsed data is empty for {target_id}.")
             return None

        # 1. Save TLEs
        df = pd.DataFrame(data)
        out_path = os.path.join(DATA_DIR, 'test_tles', f'{target_id}.csv') 
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} TLEs to {out_path}")

        # 2. Fetch & Save Metadata (Decay Epoch)
        print(f"Fetching metadata (SATCAT) for {target_id}...")
        try:
            satcat_gen = client.satcat(norad_cat_id=target_id)
            satcat_list = list(satcat_gen)
            
            if satcat_list:
                # spacetrack returns list of dicts or strings usually. 
                # satcat entries are usually dicts in the library wrapper
                obj_data = satcat_list[0]
                decay_epoch = obj_data.get('DECAY')
                
                if decay_epoch:
                    meta_path = os.path.join(DATA_DIR, 'test_tles', f'{target_id}_meta.json')
                    with open(meta_path, 'w') as f:
                        json.dump({"decay_epoch": decay_epoch}, f, indent=2)
                    print(f"Saved metadata (Decay: {decay_epoch}) to {meta_path}")
                else:
                    print("Object has not officially decayed (DECAY field empty).")
        except Exception as e:
            print(f"Warning: Could not fetch metadata: {e}")

        return out_path
        
    except Exception as e:
        print(f"Error fetching TLEs: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        tid = sys.argv[1]
        fetch_fresh_object(tid)
    else:
        print("Usage: python src/fetch_test_tle.py <NORAD_ID>")
