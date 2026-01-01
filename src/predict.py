import pandas as pd
import numpy as np
import torch
import os
import json
import sys
# Make sure we can import from src
sys.path.append(os.path.dirname(__file__))

from models import ReentryLSTM, ReentryPINN
from feature_engineering import process_tle_file, load_space_weather

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
SCALER_FILE = os.path.join(RESULTS_DIR, 'scaler_stats.json')
LSTM_PATH = os.path.join(RESULTS_DIR, 'lstm.pth')
PINN_PATH = os.path.join(RESULTS_DIR, 'pinn.pth')
SEQ_LEN = 10

# --- Survivability / Burn-up Estimation (Heuristic) ---
def estimate_survivability(obj_type, bstar):
    """
    Estimate if object burns up.
    Physics: High BSTAR = High Drag/Mass Ratio = Light Object => Burns Up.
    Object Type: Payload (Satellite) usually survives less than Rocket Body (Tank).
    """
    # Very rough heuristics
    risk = "UNCERTAIN"
    reason = ""
    
    # BSTAR > 0.001 suggests very light structure (or huge drag)
    if bstar > 0.001: 
        risk = "HIGH PROBABILITY OF BURN-UP"
        reason = "High Drag/Mass Ratio (Light structure)"
    elif "ROCKET" in obj_type.upper():
        risk = "MODERATE SURVIVAL RISK"
        reason = "Dense components (Rocket Body materials like Titanium/Steel)"
    elif "PAYLOAD" in obj_type.upper():
        risk = "LIKELY BURN-UP" 
        reason = "Modern satellites designed for demise (Aluminum mostly)"
    else:
        risk = "POSSIBLE SURVIVAL"
        reason = "Unknown composition"
        
    return risk, reason

def predict_satellite(tle_file_path):
    print(f"--- Running Inference on {tle_file_path} ---")
    
    # 1. Process TLEs into Raw Features
    if not os.path.exists(tle_file_path):
        print(f"Error: File not found: {tle_file_path}")
        return
        
    print("Parsing TLEs and calculating orbital parameters...")
    # reuse the logic from feature_engineering (it returns a DF with orbital elements)
    obj_df = process_tle_file(tle_file_path)
    
    if len(obj_df) < SEQ_LEN + 1:
        print(f"Error: Not enough TLEs. Need at least {SEQ_LEN + 1} for a prediction (10 context + 1 target).")
        return

    # 2. Merge Space Weather
    print("Merging Space Weather data...")
    sw_df = load_space_weather()
    obj_df = obj_df.sort_values('EPOCH')
    sw_df = sw_df.sort_values('DATE')
    
    # merge_asof requires sorted
    merged_df = pd.merge_asof(obj_df, sw_df, left_on='EPOCH', right_on='DATE', direction='backward')
    
    # 3. Normalize
    print("Normalizing features...")
    cols_to_norm = ['alt_mean', 'mean_motion', 'e', 'i', 'bstar', 'F10_7', 'Kp', 'Ap', 'decay_rate']
    
    with open(SCALER_FILE, 'r') as f:
        stats = json.load(f)
    
    means = pd.Series(stats['mean'])
    stds = pd.Series(stats['std'])
    
    # Create working copy for normalized data
    norm_df = merged_df.copy()
    for c in cols_to_norm:
        norm_df[c] = (norm_df[c] - means[c]) / stds[c]
        
    # 4. Prepare Sequences
    features = norm_df[cols_to_norm].values.astype(np.float32)
    timestamps = merged_df['EPOCH'].values
    true_alts = merged_df['alt_mean'].values
    
    # Load Models
    input_dim = len(cols_to_norm)
    lstm = ReentryLSTM(input_dim, 64, 1)
    lstm.load_state_dict(torch.load(LSTM_PATH))
    lstm.eval()
    
    pinn = ReentryPINN(input_dim, 64, 1)
    pinn.load_state_dict(torch.load(PINN_PATH))
    pinn.eval()
    
    import matplotlib.pyplot as plt
    
    lstm_preds = []
    pinn_preds = []
    true_vals = []
    times = []

    print("\n--- Predictions (Next Step) ---")
    print(f"{'Epoch':<20} | {'True Alt (km)':<15} | {'LSTM Pred':<15} | {'PINN Pred':<15} | {'LSTM Err':<10}")
    print("-" * 85)
    
    with torch.no_grad():
        # Predict for every possible window
        for i in range(len(features) - SEQ_LEN):
            # Input: i to i+SEQ_LEN
            x_seq = features[i : i+SEQ_LEN]
            x_tensor = torch.tensor(x_seq).unsqueeze(0) # (1, 10, feat)
            
            # Target: i+SEQ_LEN (The next step after the sequence)
            target_idx = i + SEQ_LEN
            target_alt = true_alts[target_idx]
            target_time = timestamps[target_idx]
            
            # Predict
            l_pred_norm = lstm(x_tensor).item()
            p_pred_norm = pinn(x_tensor).item()
            
            # Denormalize
            l_pred = l_pred_norm * stds['alt_mean'] + means['alt_mean']
            p_pred = p_pred_norm * stds['alt_mean'] + means['alt_mean']
            
            # Store
            lstm_preds.append(l_pred)
            pinn_preds.append(p_pred)
            true_vals.append(target_alt)
            times.append(pd.to_datetime(target_time))
            
            # Only print every 10th or last few to avoid spam
            if i % 20 == 0 or i == (len(features) - SEQ_LEN - 1):
                err = abs(l_pred - target_alt)
                time_str = str(target_time)[:16]
                print(f"{time_str:<20} | {target_alt:>13.2f}   | {l_pred:>13.2f}   | {p_pred:>13.2f}   | {err:>8.2f}")

    # --- 5. Post-Processing & Analysis ---
    
    # Define Re-entry Threshold (Critical Interface)
    # Satellites typically survive only 30-90 mins after reaching ~140km
    REENTRY_ALT_KM = 140.0
    
    def get_time_offset(obj_type):
        """
        Returns a Timedelta offset for time from 140km to Impact.
        Based on heuristics:
        - Light debris/panels: 20 mins
        - Payload (Satellites): 60 mins
        - Rocket Body: 120 mins
        """
        ot = obj_type.upper()
        minutes = 60 # Default
        
        if "PAYLOAD" in ot:
            minutes = 60 # Average 30-90 mins
        elif "ROCKET" in ot:
            minutes = 120 # Dense, 1-3 hours
        elif "DEBRIS" in ot:
            minutes = 20 # 10-30 mins
            
        return pd.Timedelta(minutes=minutes)

    def get_reentry_time(times_arr, alts_arr, label):
        """Finds first timestamp where altitude < 140km."""
        below = np.where(alts_arr < REENTRY_ALT_KM)[0]
        if len(below) > 0:
            idx = below[0]
            return times_arr[idx], alts_arr[idx]
        return None, None

    true_reentry_t, true_reentry_h = get_reentry_time(times, np.array(true_vals), "True")
    lstm_reentry_t, lstm_reentry_h = get_reentry_time(times, np.array(lstm_preds), "LSTM")
    pinn_reentry_t, pinn_reentry_h = get_reentry_time(times, np.array(pinn_preds), "PINN")
    
    # Extrapolation Utility (if data ends before 140km)
    def extrapolate_reentry(times_arr, alts_arr):
        """
        Extrapolates altitude decay to 140km threshold.
        Attempt 1: Quadratic (Acceleration) - Fits y = at^2 + bt + c
                   Catches the exponential drag increase.
        Attempt 2: Linear (Velocity) - Fallback if quadratic fit is poor.
        """
        if alts_arr[-1] > REENTRY_ALT_KM:
            # Use last 50 points or 20% of data for trend
            lookback = min(50, len(alts_arr))
            
            # Standardize inputs
            last_times_pd = pd.to_datetime(times_arr[-lookback:])
            last_alts = alts_arr[-lookback:]
            
            # Time axis: Days relative to window start
            t0 = last_times_pd[0]
            x_days = (last_times_pd - t0).total_seconds().to_numpy() / 86400.0
            y_alt = last_alts
            
            # --- Try Quadratic Fit (Account for Acceleration) ---
            try:
                # h(t) = at^2 + bt + c
                coeffs = np.polyfit(x_days, y_alt, 2) 
                a, b, c = coeffs
                
                # Check convexity: a should be negative (accelerating down) 
                # or small positive if it's just starting to turn. 
                # Ideally for drag it curves down.
                
                # Solve: at^2 + bt + (c - 140) = 0
                # Roots finding
                roots = np.roots([a, b, c - REENTRY_ALT_KM])
                
                # We need a root that is in the future (greater than current last time)
                current_t_max = x_days[-1]
                future_roots = [r.real for r in roots if np.isreal(r) and r.real > current_t_max]
                
                if future_roots:
                    t_target = min(future_roots)
                    # Check if reasonable (e.g. not 100 years away)
                    if t_target < 36500: # 100 years
                        pred_time = t0 + pd.Timedelta(days=t_target)
                        return pred_time, "Quadratic"
            except Exception:
                pass # Fail silently to linear

            # --- Fallback: Linear Fit (Constant Velocity) ---
            # h(t) = mt + c
            coeffs_lin = np.polyfit(x_days, y_alt, 1)
            m, c_lin = coeffs_lin
            
            if m < 0: # Must be decaying
                # 140 = m*t + c  => t = (140 - c)/m
                t_target = (REENTRY_ALT_KM - c_lin) / m
                if t_target > x_days[-1]:
                    pred_time = t0 + pd.Timedelta(days=t_target)
                    return pred_time, "Linear"

        return None, False

    # Post-processing: check for extrapolation if needed
    extrap_msg = ""
    if not pinn_reentry_t:
        pred_time, is_extrap = extrapolate_reentry(np.array(times), np.array(pinn_preds))
        if pred_time:
            pinn_reentry_t = pred_time
            extrap_msg = f"({is_extrap})"
            
    # --- Check for Metadata (Official Decay) ---
    meta_path = tle_file_path.replace('.csv', '_meta.json')
    actual_decay_epoch = None
    if os.path.exists(meta_path):
        try:
             with open(meta_path, 'r') as f:
                 meta = json.load(f)
                 if meta.get('decay_epoch'):
                     actual_decay_epoch = pd.to_datetime(meta['decay_epoch'])
        except Exception:
             pass

    # --- Console Output ---
    print("\n" + "="*60)
    print(f"RE-ENTRY ANALYSIS: {os.path.basename(tle_file_path)}")
    print("="*60)
    
    # Extract meta-info
    try:
        raw_df = pd.read_csv(tle_file_path)
        first_row = raw_df.iloc[0]
        obj_name = first_row.get('OBJECT_NAME', 'Unknown')
        obj_id = first_row.get('NORAD_CAT_ID', 'Unknown')
        obj_type = first_row.get('OBJECT_TYPE', 'Unknown')
        avg_bstar = raw_df['BSTAR'].mean()
    except Exception:
        obj_type = "UNKNOWN"
        avg_bstar = 0.0

    risk, risk_reason = estimate_survivability(obj_type, avg_bstar)
    offset = get_time_offset(obj_type)
    
    print(f"\n[Survivability & Timing]")
    print(f"Object Type:   {obj_type}")
    print(f"Decay Quality: {risk}")
    print(f"Time Offset:   +{offset} (from 140km to Impact)")
    
    print(f"\n[Prediction Table]")
    
    headers = ["Source", "Time to 140km", "Est. Impact Time (+Offset)", "Status"]
    row_fmt = "{:<8} | {:<20} | {:<25} | {:<15}"
    print(row_fmt.format(*headers))
    print("-" * 80)
    
    def fmt_time(t): return str(t)[:19] if t else "Not Reached"
    
    # True
    t_impact = true_reentry_t + offset if true_reentry_t else None
    t_status = "Observed" if true_reentry_t else "Data Ends > 140km"
    print(row_fmt.format("TRUE", fmt_time(true_reentry_t), fmt_time(t_impact), t_status))
    
    # Official (SATCAT)
    if actual_decay_epoch:
        print(row_fmt.format("SATCAT", "N/A", fmt_time(actual_decay_epoch), "Official Decay"))

    # LSTM
    l_impact = lstm_reentry_t + offset if lstm_reentry_t else None
    l_status = "Predicted" if lstm_reentry_t else "Stalled/High"
    print(row_fmt.format("LSTM", fmt_time(lstm_reentry_t), fmt_time(l_impact), l_status))
    
    # PINN
    p_impact = pinn_reentry_t + offset if pinn_reentry_t else None
    p_status = "Predicted " + extrap_msg
    print(row_fmt.format("PINN", fmt_time(pinn_reentry_t), fmt_time(p_impact), p_status))
    
    print("\n" + "="*60)

    # --- Plotting Updates ---
    plt.figure(figsize=(12, 7))
    
    # Main Curves
    plt.plot(times, true_vals, label='True Trajectory', color='black', linewidth=2, alpha=0.7)
    plt.plot(times, lstm_preds, label='LSTM Prediction', color='blue', linestyle='--', alpha=0.8)
    plt.plot(times, pinn_preds, label='PINN Prediction', color='orange', linestyle='-', linewidth=2)
    
    # Threshold Line
    plt.axhline(REENTRY_ALT_KM, color='red', linestyle=':', label='Crit. Boundary (140 km)')
    
    # Vertical Markers
    
    # Official Decay (Green)
    if actual_decay_epoch:
         plt.axvline(actual_decay_epoch, color='green', linestyle='-', linewidth=2, label='Official Decay')
         plt.text(actual_decay_epoch, REENTRY_ALT_KM + 15, " Official Decay", color='green', rotation=90, verticalalignment='bottom')

    if p_impact:
        plt.axvline(p_impact, color='red', linestyle='--', alpha=0.8, label="Predicted Impact")
        plt.text(p_impact, REENTRY_ALT_KM + 50, f" PINN Impact\n ({extrap_msg})", rotation=90, color='red')

    if pinn_reentry_t:
        plt.axvline(pinn_reentry_t, color='orange', linestyle=':', alpha=0.8)
        # Handle extrapolation plotting
        if extrap_msg:
             plt.text(pinn_reentry_t, REENTRY_ALT_KM + 10, f" 140km Reached", rotation=90, color='darkorange')
        else:
             plt.text(pinn_reentry_t, REENTRY_ALT_KM + 10, " 140km Reached", rotation=90, color='darkorange')

    plt.xlabel('Date')
    plt.ylabel('Mean Altitude (km)')
    plt.title(f'Re-entry Prediction & Survivability: {os.path.basename(tle_file_path)}\n({risk})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    base_name = os.path.splitext(os.path.basename(tle_file_path))[0]
    out_img = os.path.join(RESULTS_DIR, f'inference_{base_name}.png')
    plt.savefig(out_img)
    print(f"\nAnalysis Plot saved to: {out_img}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fpath = sys.argv[1]
    else:
        # Default test
        print("No file provided. Using default test object (48450.csv)...")
        fpath = os.path.join(DATA_DIR, 'tles', '48450.csv')
        
    predict_satellite(fpath)
