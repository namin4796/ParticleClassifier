import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# -- config -- 
# "Epochs": How many times the model sees the ENTIRE dataset.
# "Batch Size": How many examples it looks at before updating its brain.
# "Learning Rate": How big of a change it makes to its brain each time.
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# -- load data --
# find the csv file relative to this script
csv_path = os.path.join(os.path.dirname(__file__), '../data/orbits.csv')
print(f"Loading data from {csv_path}...")

# read csv into dataframe
df = pd.read_csv(csv_path)

# separation: X is known, y is required prediction
X = df[['r','v','m_bh']].values # 5 physics parameters
y = df['is_unstable'].values

# -- preprocessing --
# standardizing data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# convert to pytorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1) # shape becomes (N,1) instead of (N,)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")

# -- define the model --
class OrbitClassifier(nn.Module):
    def __init__(self):
        super(OrbitClassifier, self).__init__()
        # layer 1: input (3 inputs: r, v, m_bh) -> hidden layers (64 neurons)
        self.layer1 = nn.Linear(3, 64)
        # layer 2: Hidden (64 features) -> hidden layers (32 neurons)
        self.layer2 = nn.Linear(64, 32)
        # layer 3: hidden (32) -> output (1 probability)
        self.layer3 = nn.Linear(32, 1)

        # activation function: ReLU (Rectified Linear Unit)
        self.relu = nn.ReLU() # turn -ve nos. to 0
        # sigmoid for 0 <= output <= 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # This describes how data flows through the brain
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# Initialize the model
model = OrbitClassifier()

# --setup training tools ---
# Loss Function: Measures how wrong the model is.
# BCELoss = Binary Cross Entropy (Standard for Yes/No classification)
criterion = nn.BCELoss()

# Optimizer: The math formula that updates the weights. Adam is the best default.
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --the training loop--
print("\nStarting Training...")
for epoch in range(EPOCHS):
    model.train() # Set to training mode
    
    # A. Forward Pass: Make a guess
    y_pred = model(X_train_tensor)
    
    # B. Calculate Loss: How bad was the guess?
    loss = criterion(y_pred, y_train_tensor)
    
    # C. Backward Pass: Calculate gradients (Who is to blame for the error?)
    optimizer.zero_grad() # Clear old gradients
    loss.backward()       # Backpropagation
    
    # D. Update Weights: Fix the errors
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

# -- evaluation --
print("\nEvaluating on Test Set...")
model.eval() # Set to evaluation mode (turns off learning features)
with torch.no_grad(): # Don't track gradients (saves memory)
    test_predictions = model(X_test_tensor)
    # Convert probabilities (0.1, 0.9) to labels (0, 1)
    predicted_labels = (test_predictions > 0.5).float()
    
    # Calculate Accuracy
    accuracy = (predicted_labels == y_test_tensor).sum() / len(y_test_tensor)
    print(f"Final Accuracy: {accuracy.item() * 100:.2f}%")

# -- save the logic  --
torch.save(model.state_dict(), "orbit_model.pth")
print("Model saved to orbit_model.pth")
