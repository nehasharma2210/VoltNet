"""
Deploy trained model artifacts to backend
"""

import torch
import pickle
import numpy as np
import shutil
import os

print("=" * 70)
print("📦 Deploying Trained Model Artifacts")
print("=" * 70)

# Load trained model checkpoint
print("\n1️⃣ Loading trained model checkpoint...")
checkpoint = torch.load('models/best_model.pth', weights_only=False)
metadata = checkpoint['metadata']

print(f"   ✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"   ✅ Val Loss: {checkpoint['val_loss']:.6f}")
print(f"   ✅ Metadata: {list(metadata.keys())}")

# Create artifacts directory
artifact_dir = 'backend/artifacts'
os.makedirs(artifact_dir, exist_ok=True)

# 1. Copy model
print("\n2️⃣ Copying model...")
shutil.copy('models/best_model.pth', os.path.join(artifact_dir, 'best_model.pth'))
print(f"   ✅ Copied best_model.pth")

# 2. Save scalers (REAL StandardScalers from training)
print("\n3️⃣ Saving scalers...")
scalers_data = {
    'scaler_x': metadata['scaler_x'],
    'scaler_y': metadata['scaler_y'],
    'scaler_e': metadata['scaler_e'],
    'num_nodes': metadata['num_nodes'],
    'in_dim': metadata['node_features'],
    'out_dim': metadata['output_dim'],
    'edge_dim': metadata['edge_features']
}

with open(os.path.join(artifact_dir, 'scalers.pkl'), 'wb') as f:
    pickle.dump(scalers_data, f)

print(f"   ✅ Saved scalers.pkl")
print(f"   ✅ Scaler X type: {type(scalers_data['scaler_x'])}")
print(f"   ✅ Scaler Y type: {type(scalers_data['scaler_y'])}")

# 3. Create baseline_X
print("\n4️⃣ Creating baseline_X...")
baseline_X = np.ones((metadata['num_nodes'], metadata['node_features']))
np.save(os.path.join(artifact_dir, 'baseline_X.npy'), baseline_X)
print(f"   ✅ Created baseline_X.npy: shape {baseline_X.shape}")

# Verify artifacts
print("\n" + "=" * 70)
print("✅ Artifacts Deployed Successfully!")
print("=" * 70)

print(f"\n📁 Artifacts in {artifact_dir}:")
for f in os.listdir(artifact_dir):
    filepath = os.path.join(artifact_dir, f)
    size = os.path.getsize(filepath) / 1024
    print(f"   - {f} ({size:.1f} KB)")

print("\n🚀 Next: Restart backend")
print("   cd backend && python app.py")
