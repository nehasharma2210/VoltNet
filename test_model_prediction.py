"""
Test model prediction to see what's happening
"""

import torch
import numpy as np
import pickle

print("=" * 70)
print("🔍 Testing Model Prediction")
print("=" * 70)

# Load model
print("\n1️⃣ Loading model...")
checkpoint = torch.load('backend/artifacts/best_model.pth', weights_only=False)
from backend.model import SurrogateGNN

model = SurrogateGNN(in_dim=4, edge_dim=2, hidden_dim=128, n_layers=3, dropout=0.1, out_node_dim=4)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("   ✅ Model loaded")

# Load scalers
print("\n2️⃣ Loading scalers...")
with open('backend/artifacts/scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

scaler_x = scalers['scaler_x']
scaler_y = scalers['scaler_y']
print(f"   ✅ Scaler X: {type(scaler_x)}")
print(f"   ✅ Scaler Y: {type(scaler_y)}")

# Load graph
print("\n3️⃣ Loading graph...")
import scipy.io as sio
edge_index = sio.loadmat('backend/artifacts/edge_index.mat')['edge_index'] - 1  # 0-indexed
edge_attr = sio.loadmat('backend/artifacts/edge_attr.mat')['edge_attr']
print(f"   ✅ Edges: {edge_index.shape}")
print(f"   ✅ Edge attr: {edge_attr.shape}")

# Create test input
print("\n4️⃣ Creating test input...")
X = np.ones((39, 4))  # Baseline
X[:, 0] = X[:, 0] * 1.2  # Load factor
X[:, -1] = X[:, -1] * 0.6  # Renewable

print(f"   Input X shape: {X.shape}")
print(f"   Input X sample:\n{X[:3]}")

# Scale input
print("\n5️⃣ Scaling input...")
X_flat = X.reshape(-1, 4)  # (39, 4)
X_scaled = scaler_x.transform(X_flat)
X_scaled = X_scaled.reshape(39, 4)

print(f"   Scaled X shape: {X_scaled.shape}")
print(f"   Scaled X sample:\n{X_scaled[:3]}")
print(f"   Scaled X stats: min={X_scaled.min():.4f}, max={X_scaled.max():.4f}, mean={X_scaled.mean():.4f}")

# Predict
print("\n6️⃣ Running prediction...")
x_t = torch.tensor(X_scaled, dtype=torch.float)
ei_t = torch.tensor(edge_index.T, dtype=torch.long)
ea_t = torch.tensor(edge_attr, dtype=torch.float)

with torch.no_grad():
    pred_scaled = model(x_t, ei_t, ea_t).numpy()

print(f"   Prediction shape: {pred_scaled.shape}")
print(f"   Prediction (scaled) sample:\n{pred_scaled[:3]}")
print(f"   Prediction (scaled) stats: min={pred_scaled.min():.4f}, max={pred_scaled.max():.4f}, mean={pred_scaled.mean():.4f}")

# Check if all predictions are same
print(f"\n   🔍 Are all predictions same?")
print(f"   Unique values in column 0: {len(np.unique(pred_scaled[:, 0]))}")
print(f"   Std dev of column 0: {pred_scaled[:, 0].std():.6f}")

# Inverse scale
print("\n7️⃣ Inverse scaling...")
pred_flat = pred_scaled.reshape(-1, 4)
pred_real = scaler_y.inverse_transform(pred_flat)
pred_real = pred_real.reshape(39, 4)

print(f"   Real prediction shape: {pred_real.shape}")
print(f"   Real prediction sample:\n{pred_real[:3]}")
print(f"   Real prediction stats: min={pred_real.min():.4f}, max={pred_real.max():.4f}, mean={pred_real.mean():.4f}")

# Voltage
voltage = pred_real[:, 0]
print(f"\n8️⃣ Voltage values:")
print(f"   Min: {voltage.min():.4f}")
print(f"   Max: {voltage.max():.4f}")
print(f"   Mean: {voltage.mean():.4f}")
print(f"   Std: {voltage.std():.6f}")
print(f"   Unique values: {len(np.unique(voltage))}")
print(f"   First 10: {voltage[:10]}")

print("\n" + "=" * 70)
if voltage.std() < 0.001:
    print("❌ PROBLEM: All voltages are same! Model not learning properly!")
else:
    print("✅ SUCCESS: Voltages are different!")
print("=" * 70)
