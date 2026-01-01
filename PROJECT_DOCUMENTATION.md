# Space Debris Re-entry Prediction Project: Technical Documentation

## 1. Project Overview
This project targets the prediction of **re-entry trajectories** for Space Debris (specifically spent Rocket Bodies) in Low Earth Orbit (LEO). It compares two approaches:
1.  **Pure Data-Driven (LSTM)**: A deep learning model that learns patterns from historical data alone.
2.  **Physics-Informed Neural Network (PINN)**: A hybrid model that combines neural networks with physical constraints (orbital mechanics) to ensure scientific limitations are met (e.g., altitude must decrease over time).

## 2. Directory Structure & Contents

```text
Space_Reentry_Project/
├── data/                       # Stores all datasets
│   ├── tles/                   # Raw Two-Line Element (TLE) files for individual objects
│   ├── processed/              # Merged and cleaned datasets ready for training
│   ├── space_weather.csv       # Solar activity data (F10.7, Kp, Ap indices)
│   └── new_test_object.csv     # Unseen test data (CZ-2D R/B) for validation
├── src/                        # Source code for the pipeline
│   ├── fetch_tles.py           # Script to download TLEs from Space-Track
│   ├── fetch_space_weather.py  # Script to download solar data from CelesTrak
│   ├── feature_engineering.py  # core logic for processing TLEs and SGP4 propagation
│   ├── models.py               # PyTorch definitions of LSTM and PINN
│   ├── train.py                # Main training loop (Curriculum Learning)
│   ├── evaluate.py             # Script to generate metrics and plots
│   └── predict.py              # Inference script for new satellite files
├── results/                    # Output directory for models and plots
│   ├── lstm.pth, pinn.pth      # Saved trained model weights
│   ├── scaler_stats.json       # Normalization statistics (Mean/Std) to prevent leakage
│   └── *.png                   # Generated analysis plots
└── requirements.txt            # Python dependencies
```

---

## 3. Python Script Functions

### `src/fetch_tles.py`
*   **Purpose**: Connects to the Space-Track.org API.
*   **Key Function**: `fetch_decayed_objects()` - Queries LEO Rocket Bodies that decayed recently. It handles authentication and saves raw TLE text files.

### `src/feature_engineering.py`
*   **Purpose**: The mathematical core. Converts "text TLEs" into "numerical features".
*   **Key Functions**:
    *   `process_tle_file()`: Uses **SGP4** (Simplified General Perturbations 4) library to propagate the TLE to its epoch and extract:
        *   `alt_mean`: Mean Altitude (km)
        *   `e`, `inc`: Eccentricity, Inclination
        *   `mean_motion`: Revolutions per day (speed)
        *   `decay_rate`: Derived feature ($n_{t} - n_{t-1}$) showing how fast it's falling.
    *   `merge_data()`: Aligns the Satellite data with Space Weather features (F10.7, Kp) based on timestamps, using `pd.merge_asof` to match the closest readings.

### `src/train.py`
*   **Purpose**: The machine learning engine.
*   **Key Logic**:
    *   **Data Splitting**: Uses **Object-Based Splitting** (train on Satellites A, B, C, D; test on Satellite E). This prevents "Data Leakage" where the model sees the future of the test object.
    *   **Normalization**: Fits a Scaler ONLY on the training sets and applies it to the test set.
    *   **Curriculum Learning**: For the PINN, it slowly increases the `physics_weight` (lambda) from 0.0 to 1.0 over epochs. This prevents the physics loss from overpowering the data loss early in training.

### `src/models.py`
Defines the Neural Network Architectures.
*   **ReentryLSTM**: A standard Long Short-Term Memory network.
    *   *Input*: Sequence of last 10 days of orbital parameters.
    *   *Hidden Layer*: Captures time-dependent patterns (decay acceleration).
    *   *Output*: Predicted altitude at the next time step.
*   **ReentryPINN**: A Physics-Informed Feed-Forward Network.
    *   *Structure*: Simple Multi-Layer Perceptron (MLP).
    *   *Output*: Predicted altitude.
    *   *Physics*: The loss function (in `train.py`) enforces strict rules on this output.

### `src/evaluate.py`
*   **Purpose**: Validates the model on the held-out "Test Object" (ID 48450).
*   **Outputs**:
    *   **RMSE**: Root Mean Square Error (Accuracy).
    *   **R²**: Coefficient of Determination (Fit Quality).
    *   **Plots**: Generates the comparison images.

### `src/predict.py`
*   **Purpose**: **Production Inference**.
*   **Usage**: Takes ANY new TLE file, processes it exactly like training data (using saved `scaler_stats.json`), and outputs predictions.
*   **New Features**:
    *   **Re-entry Time Extraction**: Identifies exact date when altitude < 80km.
    *   **Quadratic Extrapolation**: Uses a physics-based $at^2+bt+c$ fit to predict re-entry for incomplete datasets (accounting for drag acceleration).
    *   **Survivability Estimation**: Heuristic analysis based on BSTAR (drag/mass) and Object Type.

---

## 4. Models & Algorithms Explained

### A. LSTM (Long Short-Term Memory)
**Why?** Space debris decay is a "Time-Series" problem. The altitude today depends heavily on the altitude yesterday and the drag accumulated over the last week. LSTMs are perfect for "remembering" this history.

**Pipeline**:
1.  **Input Window**: Takes a sequence of 10 data points (e.g., $t_{-9}$ to $t_{0}$).
2.  **Processing**: The LSTM cell passes a "hidden state" forward, updating its memory of the decay rate.
3.  **Prediction**: Outputs $Alt_{t+1}$ (Next Step).
4.  **Strength**: Excellent at interpolating known patterns.
5.  **Weakness**: **Overfitting**. If it sees a new orbit it hasn't trained on, it might predict nonsense because it has no "understanding" of gravity or drag.

### B. PINN (Physics-Informed Neural Network)
**Why?** To fix the LSTM's weakness. We want a model that "knows" physics, so even if it sees data it has never seen before, it won't break the laws of nature.

**The PINN Pipeline**:
The PINN is not just a network; it's a **Loss Function Strategy**.
$$ Loss_{Total} = Loss_{Data} + \lambda \cdot Loss_{Physics} $$

1.  **$Loss_{Data}$ (MSE)**: Using the training data, minimize $(Pred - True)^2$. This teaches the model the general correlation.
2.  **$Loss_{Physics}$ (Constraint)**: reliable physics knowledge.
    *   *Implemented Factor*: **Monotonic Altitude Decay**.
    *   *Rule*: A satellite WITHOUT propulsion **cannot** climb higher. Gravity and Drag only pull it down.
    *   *Equation*: $ReLu(Pred_{Alt} - Last_{Alt})$. If the model predicts an altitude increase, this term becomes huge, punishing the network severely.
3.  **Optimization**: By minimizing both, we get a model that fits the data AND obeys gravity.

**Why Compare?**
In our final test (CZ-2D object), the LSTM failed (~120km error) because the new object was different from the training set. The PINN succeeded (~10km error) because the "Physics Loss" forced it to generate a realistic, continuously falling trajectory, effectively "regularizing" the model against wild guesses.

---

## 5. Understanding the Images

### `trajectory_comparison.png`
*   **X-Axis**: Time (Date).
*   **Y-Axis**: Altitude (km).
*   **Visual**:
    *   **Black Line**: Truth. The satellite falling.
    *   **Blue Line**: LSTM. Fits perfectly on training data, but might drift on new data.
    *   **Orange Line**: PINN. Should follow the general trend smoothly. If it's a straight line, physics weight is too high. If it matches truth, it's balanced.

### `error_histogram.png`
*   **What it shows**: The distribution of mistakes.
*   **Ideal**: A tall, thin bell curve centered at 0.
*   **Interpretation**: If the curve is shifted (e.g., centered at +50), the model is "Biased" (consistently overestimating altitude).

### `inference_plot.png` (The Breakthrough)
*   **Context**: Results on the **Unseen CZ-2D R/B** object.
*   **Details**:
    *   The **PINN (Orange)** is very close to the Black line. Its internal constraints prevented it from hallucinating a rapid decay that didn't happen.
    *   **Enhancements**: Now shows the **80km Interface Line**, calculated **Re-entry Date** (with extrapolation status), and a **Survivability Risk** assessment in the title.

---

## 6. Re-entry Analysis Module
*Added in Phase 2*

### A. Quadratic Extrapolation
Standard TLE data often stops at ~130-150km because sensors lose track of the object in the final "plunge".
*   **Problem**: Linear extension incorrectly assumes constant speed, predicting re-entry weeks too late.
*   **Solution**: We fit a **Quadratic Polynomial** ($h(t) = at^2 + bt + c$) to the final data points. This captures the **Acceleration** ($a$) caused by exponentially increasing atmospheric density.
*   **Result**: Reduced prediction error from ~26 days to ~6 days on historical tests (Tiangong-1).

### B. Survivability Heuristic
Estimates if the object will burn up or hit the ground.
*   **Logic**:
    *   High BSTAR (> 0.001) IMPLIES Light Object $\rightarrow$ **High Burn-up Probability**.
    *   "ROCKET BODY" IMPLIES Dense Materials (Titanium/Steel) $\rightarrow$ **Moderate Survival Risk**.
    *   "PAYLOAD" IMPLIES Aluminum Structure $\rightarrow$ **Likely Burn-up**.
