import torch
import torch.nn as nn

class ReentryLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        """
        Baseline LSTM model.
        Predicts future Altitude given sequence of past states.
        """
        super(ReentryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        # We usually want the prediction for the next step, based on the last step of the input
        # Or we might want to predict a sequence.
        # For this setup: Many-to-One (predict next altitude).
        last_step_out = out[:, -1, :]
        pred = self.head(last_step_out)
        return pred

class ReentryPINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        """
        Physics-Informed Neural Network.
        Instead of a black-box LSTM, we use an MLP that predicts the decay rate locally,
        constrained by physics loss during training.
        """
        super(ReentryPINN, self).__init__()
        
        # Simplified MLP structure
        # Input: [Current State (Alt, n, e, ...), Space Weather]
        # Output: Predicted Decay Rate (dh/dt)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), # Tanh is often used in PINNs (smoother derivatives)
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # x is (batch, input_dim) -> representing state at ONE time point (or last of window)
        # If x is a sequence (LSTM like input), we might flatten or take last.
        # Let's assume we feed it the current state inputs to predict the instantaneous rate.
        
        # If input is (batch, seq, feat), take last
        if x.dim() == 3:
            x = x[:, -1, :]
            
        rate = self.net(x)
        return rate
