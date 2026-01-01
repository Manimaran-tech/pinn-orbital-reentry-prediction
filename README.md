# Physics-Informed Neural Networks (PINN) for Orbital Reentry Prediction

> **A hybrid deep learning framework combining SGP4 orbital mechanics with data-driven LSTM networks to predict space debris reentry trajectories with physical consistency.**

---

## 🔬 Scientific Objective

The rapid proliferation of space debris poses a critical threat to orbital sustainability and ground safety. Traditional propagators (SGP4) struggle with the highly non-linear atmospheric drag dynamics during the final decay phase, while pure data-driven models (LSTMs) often violate physical laws when extrapolating to unseen regimes.

**This project introduces a Physics-Informed Neural Network (PINN) approach that bridges this gap.** By embedding orbital mechanics constraints directly into the loss landscape, we force the neural network to respect fundamental physical laws (e.g., monotonic altitude decay, energy conservation) while allowing it to learn complex atmospheric density variations from data.

## 🚀 Key Innovation: The PINN Architecture

Our model optimizes a dual-objective loss function essential for scientific constraints:

$$ \mathcal{L}_{total} = \mathcal{L}_{Data} + \lambda \cdot \mathcal{L}_{Physics} $$

1.  **$\mathcal{L}_{Data}$ (MSE)**: Minimizes the error between predicted and observed orbital elements (Altitude, Inclination, Eccentricity).
2.  **$\mathcal{L}_{Physics}$ (Constraint)**: Enforces **monotonic decay**. Since a passive object cannot gain potential energy without propulsion, any prediction $\frac{dh}{dt} > 0$ incurs a heavy penalty.
    *   *Implementation*: A **Curriculum Learning** strategy gradually increases the weight ($\lambda$) of the physics loss, guiding the model from data-fitting to physical compliance.

## 📊 Comparative Analysis

We evaluate two distinct architectures against real-world decaying rocket bodies (e.g., CZ-2D, Falcon 9 R/B):

### 1. ReentryLSTM (Pure Data-Driven)
*   **Mechanism**: Uses Long Short-Term Memory (LSTM) cells to capture temporal dependencies in TLE time-series.
*   **Pros**: Excellent at interpolating dense historical data.
*   **Cons**: Prone to "hallucinations" (e.g., predicting orbit raising) when data is sparse or noisy.

### 2. ReentryPINN (Physics-Informed)
*   **Mechanism**: A feed-forward network regularized by physical differential constraints.
*   **Pros**: accurately predicts decay trends even with sparse data; guarantees physical validity (no altitude spikes).
*   **Result**: Reduced prediction error on unseen test objects from **>100km (LSTM)** to **~10km (PINN)**.

---

## 🛠️ System Architecture

The pipeline consists of four modular stages designed for research reproducibility:

### 1. Data Acquisition & Processing
*   **`fetch_tles.py`**: Automated retrieval of TLEs (Two-Line Elements) from Space-Track.org.
*   **`fetch_space_weather.py`**: Integration of solar flux indices (F10.7, Kp, Ap) which critically affect atmospheric density.
*   **`feature_engineering.py`**: SGP4 propagation to align asynchronous TLE epochs and extract orbital state vectors.

### 2. Model Training (`train.py`)
*   Implements **Object-Based Splitting** (Train on Objects A/B/C, Test on D) to prevent data leakage.
*   Uses **Adaptive Normalization** to handle the distinct scales of Altitude (km) vs. Eccentricity (0-1).

### 3. Inference & Extrapolation (`predict.py`)
*   **Quadratic Extrapolation**: Fits a physics-based polynomial ($h(t) = at^2 + bt + c$) to model the exponential acceleration of drag in the final <140km phase.
*   **Survivability Heuristic**: Estimates burn-up probability based on BSTAR (drag-to-mass ratio) and object material composition (Titanium vs. Aluminum).

---

## 💻 Installation & Usage

### Prerequisites
*   Python 3.8+
*   PyTorch
*   SGP4
*   Pandas / NumPy / Matplotlib

### Setup
```bash
git clone https://github.com/Manimaran-tech/pinn-orbital-reentry-prediction.git
cd pinn-orbital-reentry-prediction
pip install -r requirements.txt
```

### Running the Pipeline
**1. Inference on New Data:**
To predict the reentry of a specific object using its TLE file:
```bash
python src/predict.py data/tles/your_satellite_file.csv
```

**2. Training the Models:**
To retrain the LSTM and PINN models from scratch:
```bash
python src/train.py
```

---

## 📈 Results & Visualization

The system generates detailed comparisons showing the "Critical Interface" (140km altitude) and predicted impact windows.

*   **Black Line**: True Trajectory (Ground Truth)
*   **Blue Dashed**: LSTM Prediction (Data-Driven)
*   **Orange Line**: PINN Prediction (Physics-Constrained)

*Note: The PINN consistently avoids the "altitude drift" seen in pure LSTMs during data gaps.*

---

## 🔮 Future Ideology

This project serves as a foundational step towards **Autonomous Space Traffic Management**. By embedding physical laws into AI, we aim to create "Safe AI" systems that are robust enough to trust with critical orbital maneuvering decisions.

1.  **Higher-Fidelity Physics**: Incorporating NRLMSISE-00 atmospheric density models directly into the loss function.
2.  **Uncertainty Quantification**: Using Bayesian Neural Networks to output confidence intervals for reentry windows.
3.  **Real-Time Edge Deployment**: Optimizing models for on-orbit inference.

---
**Author**: [Manimaran](https://github.com/Manimaran-tech)
