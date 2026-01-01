import os
import sys
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

def find_candidates():
    if not USER or not PASSWORD:
        print("Error: Credentials missing (SPACETRACK_USER/PASSWORD).")
        return

    client = SpaceTrackClient(identity=USER, password=PASSWORD)
    print("Searching for TRULY ACTIVE objects (Decay=Null, Epoch > 2025-12-20) with MM > 16.0...")
    
    try:
        # Pass filters as DIRECT keyword arguments
        result = client.gp(
            epoch='>2025-12-20',
            mean_motion='>16.0',
            decay_date='null-val', # Critical: Must not be decayed
            orderby='MEAN_MOTION desc',
            limit=20,
            format='json'
        )
        
        candidates = json.loads(result)
        
        if not candidates:
             print("No active candidates found with Perigee < 250km.")
             return
             
        print(f"\nFound {len(candidates)} active candidates likely to decay soon:\n")
        print(f"{'NORAD ID':<10} | {'Name':<20} | {'Perigee (km)':<12} | {'Launch Date'}")
        print("-" * 65)
        
        for sat in candidates:
            name = sat.get('OBJECT_NAME', 'Unknown')
            norad = sat.get('NORAD_CAT_ID')
            mm = sat.get('MEAN_MOTION')
            epoch = sat.get('EPOCH')
            
            print(f"{norad:<10} | {name:<20} | {mm:<12} | {epoch}")
            
    except Exception as e:
        print(f"Error querying SpaceTrack: {e}")
            
    except Exception as e:
        print(f"Error querying SpaceTrack: {e}")

if __name__ == "__main__":
    find_candidates()
