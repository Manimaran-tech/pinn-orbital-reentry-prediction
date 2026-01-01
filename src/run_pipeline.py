import os
import sys
import subprocess

def run_pipeline():
    print("=== Space Re-entry Prediction Pipeline ===")
    
    # 1. Get Input
    target_id = input("Enter NORAD ID (e.g., 37820): ").strip()
    if not target_id:
        print("Error: No ID provided. Exiting.")
        return

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(project_root, 'data', 'test_tles', f'{target_id}.csv')
    fetch_script = os.path.join(project_root, 'src', 'fetch_test_tle.py')
    predict_script = os.path.join(project_root, 'src', 'predict.py')

    # 2. Always Fetch Data to ensure we have the latest info
    print(f"\n[INFO] Fetching latest TLE data for {target_id}...")
    try:
        result = subprocess.run(
            [sys.executable, fetch_script, target_id], 
            capture_output=True, 
            text=True
        )
        print(result.stdout)
        if result.returncode != 0 or ("Error" in result.stdout and "[ERROR]" in result.stdout): 
            # Note: fetch script might print some errors but still succeed partly, 
            # but we'll assume non-zero return code or explicit Error is bad.
            # adjusting check to be safe
            if result.returncode != 0:
                 print(f"[ERROR] Fetch failed: {result.stderr}")
                 return

        # Double check if file was actually created
        if not os.path.exists(data_file):
            print(f"[ERROR] Fetch script ran but file {data_file} was not created.")
            return
            
    except Exception as e:
        print(f"[ERROR] Failed to run fetch script: {e}")
        return

    # 3. Run Prediction
    print(f"\n[INFO] Running Prediction for {target_id}...")
    try:
        subprocess.run([sys.executable, predict_script, data_file], check=True)
        print("\n[SUCCESS] Pipeline Complete.")
        print(f"Check results/inference_{target_id}.png")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Prediction failed: {e}")
        return

    # 4. Optional Cleanup
    choice = input(f"\nDo you want to delete the downloaded data (csv & meta) for {target_id}? (y/n): ").strip().lower()
    if choice == 'y':
        try:
            if os.path.exists(data_file):
                os.remove(data_file)
                print(f"Deleted: {os.path.basename(data_file)}")
            
            meta_file = data_file.replace('.csv', '_meta.json')
            if os.path.exists(meta_file):
                os.remove(meta_file)
                print(f"Deleted: {os.path.basename(meta_file)}")
                
        except Exception as e:
            print(f"[ERROR] Failed to delete files: {e}")
    else:
        print("[INFO] Data files kept.")

if __name__ == "__main__":
    run_pipeline()
