import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# --- 1. DEFINE MODEL (Must match train.py) ---
class OrbitClassifier(nn.Module):
    def __init__(self):
        super(OrbitClassifier, self).__init__()
        self.layer1 = nn.Linear(3, 64)  # Input size 3!
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# --- 2. RELOAD SCALER ---
csv_path = os.path.join(os.path.dirname(__file__), '../data/orbits.csv')
df = pd.read_csv(csv_path)
X_raw = df[['r', 'v', 'm_bh']].values # Load r, v, m
scaler = StandardScaler()
scaler.fit(X_raw)

# --- 3. LOAD MODEL ---
model = OrbitClassifier()
model.load_state_dict(torch.load("orbit_model.pth"))
model.eval()

# --- 4. GENERATE GRID ---
r_vals = np.linspace(50, 150, 100)
v_vals = np.linspace(0, 10, 100)
R, V = np.meshgrid(r_vals, v_vals)

# Create inputs: [r, v, m]
grid_inputs = np.zeros((R.size, 3))
grid_inputs[:, 0] = R.ravel() # r
grid_inputs[:, 1] = V.ravel() # v
grid_inputs[:, 2] = 1000.0    # m

# Scale and Predict
grid_tensor = torch.FloatTensor(scaler.transform(grid_inputs))
with torch.no_grad():
    predictions = model(grid_tensor).numpy().reshape(R.shape)

# --- 5. PLOT ---
plt.figure(figsize=(10, 6))
plt.contourf(R, V, predictions, levels=50, cmap='coolwarm', alpha=0.8)
plt.colorbar(label='AI Unstable Probability')

# Theoretical Line
v_theory = np.sqrt(2 * 1.0 * 1000.0 / r_vals)
plt.plot(r_vals, v_theory, 'k--', linewidth=3, label='Theoretical Escape Velocity')

plt.xlabel('Radius')
plt.ylabel('Velocity')
plt.title('AI Prediction with Feature Engineering\n(Inputs: Radius, Velocity, Mass)')
plt.legend()
plt.savefig("phase_space_result.png", dpi=150)
plt.show()
