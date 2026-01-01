import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from models import ReentryLSTM, ReentryPINN

# Params
SEQ_LEN = 10     # Past 10 steps to predict next
PRED_HORIZON = 1 # Predict 1 step ahead (or we can predict sequence)
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001

DATA_FILE = r'C:\Users\K\Desktop\my python programs\New folder\Space_Reentry_Project\data\processed\dataset_v1.csv'
OUT_DIR = r'C:\Users\K\Desktop\my python programs\New folder\Space_Reentry_Project\results'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

class ReentryDataset(Dataset):
    def __init__(self, df, seq_len=10):
        self.seq_len = seq_len
        self.samples = []
        self.targets = []
        
        # Group by Object to avoid window crossing objects
        grouped = df.groupby('NORAD_ID')
        
        # Features to use
        # X: alt_mean, mean_motion, e, i, bstar, F10_7, Kp, Ap, decay_rate (lagged)
        # Y: alt_mean (next step)
        
        # Note: We should normalize features!
        self.feature_cols = ['alt_mean', 'mean_motion', 'e', 'i', 'bstar', 'F10_7', 'Kp', 'Ap', 'decay_rate']
        self.target_col = 'alt_mean'
        
        # Normalization (simple min-max or std for now)
        # We need to fit scaler on TRAIN only. Ideally pass scaler in.
        # For simplicity in this script, we'll assume df is already scaled or we scale locally.
        # But wait, we need to return UN-scaled targets for evaluation?
        # Let's stick to raw for logic, implement scaling outside.
        
        for _, group in grouped:
            group = group.sort_values('EPOCH')
            values = group[self.feature_cols].values.astype(np.float32)
            y_values = group[self.target_col].values.astype(np.float32)
            
            # Create sliding windows
            count = len(group)
            if count <= seq_len:
                continue
                
            for i in range(count - seq_len):
                x_window = values[i : i+seq_len]
                y_target = y_values[i+seq_len] # Next step altitude
                
                self.samples.append(x_window)
                self.targets.append(y_target)
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]), torch.tensor(self.targets[idx])

def train_model(model, train_loader, test_loader, model_name="model"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    train_losses = []
    test_losses = []
    
    print(f"Starting training for {model_name}...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward
            if model_name == "PINN":
                pred_alt = model(X) # MLP output -> Alt
                
                # Physics Loss 1: Monotonicity (Data is noisy, but truth is monotonic)
                # predicted alt should be <= current alt (X_last)
                x_last_alt = X[:, -1, 0] # 0 is alt_mean index
                
                # Penalty if Prediction > Last_Observed (means altitude went UP)
                # We relax this slightly with a margin or softplus, but relu is standard.
                diff_violation = torch.relu(pred_alt.flatten() - x_last_alt)
                phy_loss_monotonicity = torch.mean(diff_violation ** 2) # Square it to match MSE scale
                
                # Physics Loss 2: Decay Rate Consistency (Optional/Advanced)
                # We can enforce that (pred - last)/dt is close to observed decay_rate feature?
                # For now, let's stick to fixing the "Weighting is Killing You" issue.
                
                loss_data = criterion(pred_alt.flatten(), y)
                
                # Curriculum Learning:
                # Epoch 0 -> weight 0 (Learn data first)
                # Epoch 50 -> weight MAX
                progress = epoch / EPOCHS
                # Cap weight at a reasonable value (e.g., 0.1 to 1.0, not 10.0)
                # If Data Loss is ~0.001 (normalized), Physics Loss should be comparable.
                # If we weight it 10.0, we force flat-line (monotonicity dominates data).
                
                lambda_phy = 0.5 * progress # Ramp up to 0.5
                
                loss = loss_data + lambda_phy * phy_loss_monotonicity
                
                # Log components occasionally
                if i == 0 and (epoch+1) % 10 == 0:
                    print(f" [PINN Debug] Data Loss: {loss_data.item():.6f} | Phy Loss: {phy_loss_monotonicity.item():.6f} | Lambda: {lambda_phy:.2f}")

            else:
                pred = model(X)
                loss = criterion(pred.flatten(), y)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train = running_loss / len(train_loader)
        train_losses.append(avg_train)
        
        # Test
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                out = model(X)
                if model_name == "PINN":
                     test_loss += criterion(out.flatten(), y).item() # Just check MSE for test
                else:
                     test_loss += criterion(out.flatten(), y).item()
        
        avg_test = test_loss / len(test_loader)
        test_losses.append(avg_test)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.4f} | Test Loss: {avg_test:.4f}")
            
    # Plot
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, f'{model_name}_loss.png'))
    plt.close()
    
    return model

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    # Split Train/Test by Object
    # We have 5 objects.
    all_ids = df['NORAD_ID'].unique()
    test_id = all_ids[1] # Pick second one
    
    train_df = df[df['NORAD_ID'] != test_id].copy()
    test_df = df[df['NORAD_ID'] == test_id].copy()
    
    print(f"Train on objects: {train_df['NORAD_ID'].unique()}")
    print(f"Test on object: {test_id}")
    
    # Normalize features
    cols_to_norm = ['alt_mean', 'mean_motion', 'e', 'i', 'bstar', 'F10_7', 'Kp', 'Ap', 'decay_rate']
    
    # Fit on TRAIN
    train_means = train_df[cols_to_norm].mean()
    train_stds = train_df[cols_to_norm].std() + 1e-6
    
    # Apply to Train
    for c in cols_to_norm:
        train_df[c] = (train_df[c] - train_means[c]) / train_stds[c]
        
    # Apply to Test (using TRAIN stats)
    for c in cols_to_norm:
        test_df[c] = (test_df[c] - train_means[c]) / train_stds[c]
        
    # Save statistics for evaluation script
    import json
    stats = {
        'mean': train_means.to_dict(),
        'std': train_stds.to_dict()
    }
    with open(os.path.join(OUT_DIR, 'scaler_stats.json'), 'w') as f:
        json.dump(stats, f)
    
    train_ds = ReentryDataset(train_df, SEQ_LEN)
    test_ds = ReentryDataset(test_df, SEQ_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train LSTM
    input_dim = len(train_ds.feature_cols)
    lstm = ReentryLSTM(input_dim, 64, 1)
    train_model(lstm, train_loader, test_loader, "LSTM")
    torch.save(lstm.state_dict(), os.path.join(OUT_DIR, "lstm.pth"))
    
    # Train PINN
    pinn = ReentryPINN(input_dim, 64, 1)
    train_model(pinn, train_loader, test_loader, "PINN")
    torch.save(pinn.state_dict(), os.path.join(OUT_DIR, "pinn.pth"))
    
    print("Done!")

if __name__ == "__main__":
    main()

