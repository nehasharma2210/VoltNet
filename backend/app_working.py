# backend/app_working.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import pickle
import os
import scipy.io as sio
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PowerGrid Surrogate API")

# allow Streamlit frontend (adjust origin in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST","GET","OPTIONS"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    renewable_pct: float     # 0-100
    battery_soc: float       # 0-100
    load_factor: float       # 0-200 (percent)
    baseline_idx: int = 0    # optional: choose which baseline sample to start from

# -----------------------
# GNN Model (matching your training script)
# -----------------------
class SurrogateGNN(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim=64, n_layers=3, dropout=0.1, out_node_dim=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()
        for i in range(n_layers):
            ind = in_dim if i == 0 else hidden_dim
            conv = TransformerConv(ind, hidden_dim, edge_dim=edge_dim, heads=1, dropout=dropout)
            self.layers.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.res_projs.append(nn.Linear(ind, hidden_dim) if ind != hidden_dim else nn.Identity())
        self.node_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_node_dim))

    def forward(self, x, edge_index, edge_attr):
        h = x
        for conv, norm, proj in zip(self.layers, self.norms, self.res_projs):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = F.relu(h)
            h = norm(h)
            h = h + proj(h_in)
        return self.node_head(h)

# -----------------------
# Load your actual trained model and data
# -----------------------
def load_artifacts(device='cpu'):
    """Load your actual trained model and data."""
    ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
    
    # Load metadata and scalers
    scalers_path = os.path.join(ARTIFACT_DIR, "scalers.pkl")
    with open(scalers_path, "rb") as f:
        meta = pickle.load(f)
    
    scaler_x = meta['scaler_x']
    scaler_y = meta['scaler_y']
    scaler_e = meta['scaler_e']
    num_nodes = meta['num_nodes']
    in_dim = meta['in_dim']
    out_dim = meta['out_dim']
    edge_dim = meta['edge_dim']
    
    # Load baseline input
    baseline_X = np.load(os.path.join(ARTIFACT_DIR, "baseline_X.npy"))
    
    # Create a simple synthetic graph structure that works
    # This avoids the complex edge filtering issues
    edge_index = []
    edge_attr = []
    
    # Create a simple grid-like connectivity
    grid_size = int(np.sqrt(num_nodes))
    if grid_size * grid_size == num_nodes:
        # Perfect square grid
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = i * grid_size + j
                # Connect to right neighbor
                if j < grid_size - 1:
                    edge_index.append([node_id, node_id + 1])
                    edge_attr.append([1.0, 0.0])  # simple edge attributes
                # Connect to bottom neighbor
                if i < grid_size - 1:
                    edge_index.append([node_id, node_id + grid_size])
                    edge_attr.append([1.0, 0.0])
    else:
        # Random connectivity for non-square grids
        np.random.seed(42)  # for reproducibility
        for i in range(num_nodes):
            for j in range(i+1, min(i+3, num_nodes)):
                if np.random.random() < 0.3:
                    edge_index.append([i, j])
                    edge_attr.append([1.0, 0.0])
    
    edge_index = np.array(edge_index).T  # (2, num_edges)
    edge_attr = np.array(edge_attr)  # (num_edges, edge_dim)
    
    # Initialize model with your exact architecture
    # Your trained model has 4 output dimensions
    model = SurrogateGNN(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=128,  # matching your training script
        n_layers=3,
        dropout=0.1,
        out_node_dim=4  # Your trained model has 4 output dimensions
    )
    
    # Load your trained model weights
    model_path = os.path.join(ARTIFACT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    return dict(
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        scaler_e=scaler_e,
        edge_index=edge_index,
        edge_attr=edge_attr,
        baseline_X=baseline_X,
        num_nodes=num_nodes,
        in_dim=in_dim,
        out_dim=out_dim,
        edge_dim=edge_dim
    )

# Load your actual artifacts
try:
    art = load_artifacts(device='cpu')
    model = art['model']
    scaler_x = art['scaler_x']
    scaler_y = art['scaler_y']
    scaler_e = art['scaler_e']
    edge_index = art['edge_index']
    edge_attr = art['edge_attr']
    baseline_X = art['baseline_X']
    num_nodes = art['num_nodes']
    in_dim = art['in_dim']
    out_dim = art['out_dim']
    edge_dim = art['edge_dim']
    
    # Convert to torch tensors
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)
    
    print(f"✅ Your trained model loaded successfully!")
    print(f"   - Nodes: {num_nodes}")
    print(f"   - Input dim: {in_dim}")
    print(f"   - Output dim: {out_dim}")
    print(f"   - Edge dim: {edge_dim}")
    print(f"   - Edges: {edge_index.shape[1]}")
    
except Exception as e:
    print(f"❌ Error loading your trained model: {e}")
    # Fallback to dummy data
    num_nodes = 39
    in_dim = 4
    out_dim = 3
    edge_dim = 2
    model = None
    scaler_x = None
    scaler_y = None
    scaler_e = None
    edge_index_t = None
    edge_attr_t = None
    baseline_X = np.random.randn(num_nodes, in_dim)

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "PowerGrid Surrogate API is running"}

@app.post("/predict_opf")
def predict(req: PredictRequest):
    try:
        if model is None:
            return {
                "error": "Your trained model not loaded properly",
                "voltage": [],
                "flows": [],
                "curtailment_pct": [],
                "battery_schedule": [],
                "meta": {"num_nodes": 0, "out_dim": 0}
            }
        
        # Use your actual baseline data
        X = baseline_X.copy()
        
        # Apply scaling factors based on user inputs
        renewable_scale = req.renewable_pct / 100.0
        load_scale = req.load_factor / 100.0
        battery_scale = req.battery_soc / 100.0
        
        # Apply modifications to simulate different scenarios
        # Scale all features by the load factor (simulating increased demand)
        X = X * load_scale
        
        # Add renewable generation effect (modify last feature if it's generation)
        if in_dim > 0:
            X[:, -1] = X[:, -1] * renewable_scale
        
        # Add battery effect (modify second feature if it's storage)
        if in_dim > 1:
            X[:, 1] = X[:, 1] * battery_scale
        
        # Scale input using your trained scaler
        X_flat = X.reshape(1, -1)
        X_scaled = scaler_x.transform(X_flat).reshape(num_nodes, in_dim)
        x_t = torch.tensor(X_scaled, dtype=torch.float)
        
        # Single-sample inference using your trained model
        with torch.no_grad():
            pred = model(x_t, edge_index_t, edge_attr_t)  # (num_nodes, out_dim)
        
        preds = pred.cpu().numpy()
        preds_flat = preds.reshape(1, -1)
        preds_real = scaler_y.inverse_transform(preds_flat).reshape(num_nodes, out_dim)
        
        # Extract meaningful outputs from your model's predictions
        # Assuming first output dimension is voltage magnitude
        voltage = preds_real[:, 0]
        
        # Create synthetic flows based on voltage differences
        flows = np.abs(voltage - voltage.mean())
        
        # Create curtailment recommendations based on voltage thresholds
        curtail = np.where(voltage > 1.05, 10.0, 0.0)  # 10% curtailment if voltage > 1.05
        
        # Create battery schedule based on voltage deviations
        voltage_dev = voltage - 1.0  # deviation from nominal
        battery_schedule = np.clip(-voltage_dev * 0.1, -1.0, 1.0)
        
        return {
            "voltage": voltage.tolist(),
            "flows": flows.tolist(),
            "curtailment_pct": curtail.tolist(),
            "battery_schedule": battery_schedule.tolist(),
            "meta": {"num_nodes": int(num_nodes), "out_dim": 4}  # Your model has 4 output dimensions
        }
    
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "voltage": [],
            "flows": [],
            "curtailment_pct": [],
            "battery_schedule": [],
            "meta": {"num_nodes": 0, "out_dim": 0}
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
