import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from models import ReentryLSTM, ReentryPINN
from train import ReentryDataset

DATA_FILE = r'C:\Users\K\Desktop\my python programs\New folder\Space_Reentry_Project\data\processed\dataset_v1.csv'
OUT_DIR = r'C:\Users\K\Desktop\my python programs\New folder\Space_Reentry_Project\results'
SEQ_LEN = 10

def evaluate():
    print("Loading data for evaluation...")
    df = pd.read_csv(DATA_FILE)
    
    # Define columns to normalize (must match training)
    cols_to_norm = ['alt_mean', 'mean_motion', 'e', 'i', 'bstar', 'F10_7', 'Kp', 'Ap', 'decay_rate']

    # Load Scaler Stats
    import json
    stats_file = os.path.join(OUT_DIR, 'scaler_stats.json')
    with open(stats_file, 'r') as f:
        stats = json.load(f)
        
    means = pd.Series(stats['mean'])
    stds = pd.Series(stats['std'])
    
    # Apply to Test DF (Normalize using TRAIN stats)
    for c in cols_to_norm:
        df[c] = (df[c] - means[c]) / stds[c]
        
    # Pick the SAME test (hardcoded logic assumption same as train.py)
    all_ids = df['NORAD_ID'].unique()
    test_id = all_ids[1] 
    print(f"Evaluating on Test Object: {test_id}")
    
    test_df = df[df['NORAD_ID'] == test_id].sort_values('EPOCH')
    
    # Prepare dataset
    test_ds = ReentryDataset(test_df, SEQ_LEN)
    
    # Load Models
    input_dim = len(test_ds.feature_cols)
    
    lstm = ReentryLSTM(input_dim, 64, 1)
    lstm.load_state_dict(torch.load(os.path.join(OUT_DIR, "lstm.pth")))
    lstm.eval()
    
    pinn = ReentryPINN(input_dim, 64, 1)
    pinn.load_state_dict(torch.load(os.path.join(OUT_DIR, "pinn.pth")))
    pinn.eval()
    
    # Predict
    true_alts = []
    lstm_preds = []
    pinn_preds = []
    timestamps = []
    
    # We iterate through the dataset (sliding windows)
    # This gives us "One-Step-Ahead" predictions
    print("Generating predictions...")
    with torch.no_grad():
        for i in range(len(test_ds)):
            X, y = test_ds[i] # X is (seq_len, feat), y is scalar
            X = X.unsqueeze(0) # Batch dim
            
            # LSTM
            l_out = lstm(X)
            lstm_preds.append(l_out.item())
            
            # PINN
            p_out = pinn(X)
            pinn_preds.append(p_out.item())
            
            true_alts.append(y.item())
            # For timestamp, we need to map back to original df
            # The target index in df corresponds to i + seq_len
            # But grouping adds complexity.
            # Simplified: just plot vs index
            
    # ... (prediction loop same as before) ...
    # We need to make sure we collect predictions first.
    # (Since I'm replacing the whole file content effectively via big chunk, I'll rewrite the plotting part)

    # Denormalize
    alt_mean = means['alt_mean']
    alt_std = stds['alt_mean']
    
    true_alts = np.array(true_alts) * alt_std + alt_mean
    lstm_preds = np.array(lstm_preds) * alt_std + alt_mean
    pinn_preds = np.array(pinn_preds) * alt_std + alt_mean
    
    # --- Scientific Metrics ---
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    def print_metrics(name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"--- {name} Metrics ---")
        print(f"RMSE: {rmse:.4f} km")
        print(f"MAE:  {mae:.4f} km")
        print(f"R^2:  {r2:.4f}")
        return rmse, mae
        
    print("\nEvaluation Results:")
    lstm_rmse, lstm_mae = print_metrics("LSTM", true_alts, lstm_preds)
    pinn_rmse, pinn_mae = print_metrics("PINN", true_alts, pinn_preds)
    
    # --- Visualizations ---
    
    # 1. Decay Trajectory Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(true_alts, label='Ground Truth', color='black', linewidth=2)
    plt.plot(lstm_preds, label=f'LSTM (RMSE={lstm_rmse:.2f})', linestyle='--', alpha=0.8)
    plt.plot(pinn_preds, label=f'PINN (RMSE={pinn_rmse:.2f})', linestyle='-.', alpha=0.8)
    plt.xlabel('Time Step')
    plt.ylabel('Altitude (km)')
    plt.title(f'Altitude Decay Prediction (Object {test_id})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, 'trajectory_comparison.png'))
    plt.close()
    
    # 2. Error Distribution (Histogram)
    lstm_err = lstm_preds - true_alts
    pinn_err = pinn_preds - true_alts
    
    plt.figure(figsize=(10, 6))
    plt.hist(lstm_err, bins=30, alpha=0.5, label='LSTM Errors', color='blue')
    plt.hist(pinn_err, bins=30, alpha=0.5, label='PINN Errors', color='orange')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Prediction Error (km)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution Histogram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, 'error_histogram.png'))
    plt.close()
    
    # 3. Scatter Plot (Pred vs True)
    plt.figure(figsize=(8, 8))
    plt.scatter(true_alts, lstm_preds, alpha=0.3, label='LSTM', s=10)
    plt.scatter(true_alts, pinn_preds, alpha=0.3, label='PINN', s=10)
    plt.plot([min(true_alts), max(true_alts)], [min(true_alts), max(true_alts)], 'k--', label='Perfect Fit')
    plt.xlabel('True Altitude (km)')
    plt.ylabel('Predicted Altitude (km)')
    plt.title('Predicted vs True Altitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, 'scatter_pred_vs_true.png'))
    plt.close()

    print(f"All plots saved to {OUT_DIR}")


if __name__ == "__main__":
    evaluate()
