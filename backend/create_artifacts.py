# backend/create_artifacts.py
# Script to create artifacts that match your training script structure
import numpy as np
import torch
import pickle
from sklearn.preprocessing import MinMaxScaler
import os

def create_artifacts():
    """Create synthetic artifacts for demo purposes that match your training script structure."""
    
    # Create artifacts directory
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Parameters matching your training script
    num_nodes = 39  # from your baseline_X.npy
    in_dim = 4      # from your baseline_X.npy
    out_dim = 3     # typical for power grid (voltage, angle, power)
    edge_dim = 2    # edge attributes
    
    print(f"Creating artifacts for {num_nodes} nodes, {in_dim} input dims, {out_dim} output dims")
    
    # 1. Create baseline_X.npy (if it doesn't exist)
    baseline_path = os.path.join(artifacts_dir, "baseline_X.npy")
    if not os.path.exists(baseline_path):
        # Create synthetic baseline data
        baseline_X = np.random.randn(num_nodes, in_dim) * 0.1 + 1.0  # centered around 1.0
        np.save(baseline_path, baseline_X)
        print(f"✅ Created {baseline_path}")
    else:
        baseline_X = np.load(baseline_path)
        print(f"✅ Using existing {baseline_path}")
    
    # 2. Create scalers.pkl (matching your training script)
    scalers_path = os.path.join(artifacts_dir, "scalers.pkl")
    
    # Create synthetic data to fit scalers (matching your training approach)
    n_samples = 1000
    
    # Create synthetic X data (n_samples, num_nodes * in_dim)
    X_flat = np.random.randn(n_samples, num_nodes * in_dim) * 0.5 + 1.0
    
    # Create synthetic Y data (n_samples, num_nodes * out_dim)
    Y_flat = np.random.randn(n_samples, num_nodes * out_dim) * 0.2 + 1.0
    
    # Create synthetic edge attributes
    edge_attr_flat = np.random.randn(100, edge_dim) * 0.1 + 1.0
    
    # Fit scalers
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_e = MinMaxScaler()
    
    scaler_x.fit(X_flat)
    scaler_y.fit(Y_flat)
    scaler_e.fit(edge_attr_flat)
    
    # Save scalers with metadata (matching your training script)
    scalers_data = {
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'scaler_e': scaler_e,
        'num_nodes': num_nodes,
        'in_dim': in_dim,
        'out_dim': out_dim,
        'edge_dim': edge_dim
    }
    
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers_data, f)
    print(f"✅ Created {scalers_path}")
    
    # 3. Create a dummy model (since we don't have the actual trained model)
    model_path = os.path.join(artifacts_dir, "best_model.pth")
    if not os.path.exists(model_path):
        # Import the model class
        import sys
        sys.path.append('.')
        from model import SurrogateGNN
        
        # Create and save a dummy model
        model = SurrogateGNN(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=128,
            n_layers=3,
            dropout=0.1,
            out_node_dim=out_dim
        )
        
        # Initialize with some weights
        for param in model.parameters():
            torch.nn.init.normal_(param, 0, 0.1)
        
        torch.save(model.state_dict(), model_path)
        print(f"✅ Created {model_path}")
    else:
        print(f"✅ Using existing {model_path}")
    
    print("\n🎉 All artifacts created successfully!")
    print("You can now run the backend with your trained model by replacing best_model.pth")

if __name__ == "__main__":
    create_artifacts()
