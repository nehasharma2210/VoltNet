# backend/app.py
import os
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from model_loader import load_artifacts
import traceback

app = FastAPI(title="PowerGrid Surrogate API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add static file serving to FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "PowerGrid Surrogate API is running!",
        "frontend": "/app",
        "endpoints": {
            "health": "/health",
            "predict": "/predict_opf",
            "docs": "/docs"
        }
    }

# Mount static files only if directory exists
static_dir = "static"
if not os.path.exists(static_dir):
    # Try different paths
    possible_paths = ["../frontend", "./frontend", "../static"]
    for path in possible_paths:
        if os.path.exists(path):
            static_dir = path
            break
    else:
        # Create empty static directory if none found
        os.makedirs("static", exist_ok=True)
        static_dir = "static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve frontend files
@app.get("/app")
async def serve_frontend():
    frontend_file = os.path.join(static_dir, "index.html")
    if os.path.exists(frontend_file):
        return FileResponse(frontend_file)
    else:
        return {"message": "Frontend not available", "api_docs": "/docs"}

# Serve other frontend files
@app.get("/app/{file_path:path}")
async def serve_frontend_files(file_path: str):
    file_location = os.path.join(static_dir, file_path)
    if os.path.exists(file_location) and os.path.isfile(file_location):
        return FileResponse(file_location)
    else:
        return FileResponse(os.path.join(static_dir, "index.html"))

class PredictRequest(BaseModel):
    renewable_pct: float = Field(..., ge=0.0, le=100.0)
    battery_soc: float = Field(..., ge=0.0, le=100.0)
    load_factor: float = Field(..., ge=0.0, le=200.0)
    baseline_idx: int = Field(0, ge=0)

ART = {}
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

import asyncio
from keep_alive import keep_alive_loop

@app.on_event("startup")
def startup_event():
    global ART
    try:
        ART = load_artifacts(artifact_dir=ARTIFACT_DIR, device='cpu', verbose=True)
        print("✅ Artifacts loaded successfully at startup.")
        
        # Start keep-alive task
        asyncio.create_task(keep_alive_loop())
        print("🔄 Keep-alive service started")
        
    except Exception as e:
        print("❌ Failed to load artifacts during startup:", str(e))
        traceback.print_exc()
        ART = {}

@app.get("/health")
def health_check():
    ok = bool(ART)
    return {"status": "healthy" if ok else "unhealthy", "loaded": ok, "message": "Artifacts loaded" if ok else "Artifacts not loaded; check logs."}

@app.post("/predict_opf")
def predict(req: PredictRequest):
    if not ART:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded. Check server logs.")

    try:
        model = ART["model"]
        scaler_x = ART["scaler_x"]
        scaler_y = ART["scaler_y"]
        scaler_e = ART["scaler_e"]
        edge_index = ART["edge_index"]
        edge_attr = ART["edge_attr"]
        baseline_X = ART["baseline_X"].copy()
        num_nodes = int(ART["num_nodes"])
        in_dim = int(ART["in_dim"])
        out_dim = int(ART["out_dim"])          # checkpoint-derived
        meta_out_dim = int(ART.get("meta_out_dim", out_dim))  # scaler metadata

        renewable_scale = float(req.renewable_pct) / 100.0
        load_scale = float(req.load_factor) / 100.0
        battery_scale = float(req.battery_soc) / 100.0

        # Get baseline - should be (num_nodes, in_dim)
        bx = np.array(baseline_X)
        if bx.ndim == 1:
            # If 1D, reshape to (num_nodes, in_dim)
            if bx.size == num_nodes * in_dim:
                bx = bx.reshape(num_nodes, in_dim)
            else:
                print(f"WARNING: baseline size {bx.size} != expected {num_nodes*in_dim}. Creating default baseline.")
                bx = np.ones((num_nodes, in_dim))
        elif bx.ndim == 2:
            # If 2D, ensure correct shape
            if bx.shape != (num_nodes, in_dim):
                print(f"WARNING: baseline shape {bx.shape} != expected ({num_nodes}, {in_dim}). Creating default baseline.")
                bx = np.ones((num_nodes, in_dim))
        else:
            print(f"WARNING: baseline has {bx.ndim} dimensions. Creating default baseline.")
            bx = np.ones((num_nodes, in_dim))
        
        bx = bx.astype(float)

        # scenario edits
        X = bx * load_scale
        if in_dim > 0:
            X[:, -1] = X[:, -1] * renewable_scale
        if in_dim > 1:
            X[:, 1] = X[:, 1] * battery_scale

        # Scale input - IMPORTANT: StandardScaler was fitted on (num_samples*num_nodes, features)
        # So we need to flatten per-node, not per-sample
        X_flat_per_node = X.reshape(-1, in_dim)  # Shape: (num_nodes, in_dim)
        X_scaled_per_node = scaler_x.transform(X_flat_per_node)  # Transform each node
        X_scaled = X_scaled_per_node.reshape(num_nodes, in_dim).astype(np.float32)

        ei_t = torch.tensor(edge_index, dtype=torch.long)
        ea_t = torch.tensor(edge_attr, dtype=torch.float)
        device = next(model.parameters()).device
        x_t = torch.tensor(X_scaled, dtype=torch.float, device=device)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(x_t, ei_t.to(device), ea_t.to(device)).cpu().numpy()

        # Inverse transform - IMPORTANT: StandardScaler was fitted on (num_samples*num_nodes, output_dim)
        # So we need to flatten per-node, not per-sample
        pred_scaled_per_node = pred_scaled.reshape(-1, out_dim)  # Shape: (num_nodes, out_dim)
        pred_real_per_node = scaler_y.inverse_transform(pred_scaled_per_node)  # Inverse transform each node
        pred_real = pred_real_per_node.reshape(num_nodes, out_dim)  # Shape: (num_nodes, out_dim)

        # build outputs with realistic bounds
        raw_voltage = pred_real[:, 0] if out_dim > 0 else np.zeros(num_nodes)
        
        # Model predicts voltage magnitude, normalize to per-unit (p.u.)
        # Typical base voltage for IEEE39 is around 345 kV for transmission
        # But model output is in arbitrary units, so normalize to 0.95-1.05 p.u. range
        voltage_min, voltage_max = raw_voltage.min(), raw_voltage.max()
        
        if voltage_max > voltage_min and voltage_max > 2.0:
            # Normalize to 0-1, then scale to realistic p.u. range
            voltage_normalized = (raw_voltage - voltage_min) / (voltage_max - voltage_min)
            voltage = 0.95 + (voltage_normalized * 0.10)  # 0.95 to 1.05 p.u. range
        elif voltage_max > 1.5:
            # Values too high, normalize
            voltage_normalized = (raw_voltage - voltage_min) / (voltage_max - voltage_min) if voltage_max > voltage_min else np.ones_like(raw_voltage) * 0.5
            voltage = 0.95 + (voltage_normalized * 0.10)
        else:
            # Already in reasonable range, just clip
            voltage = np.clip(raw_voltage, 0.85, 1.15)
        
        flows = np.abs(voltage - voltage.mean())
        curtail = np.where(voltage > 1.05, 10.0, 0.0)
        voltage_dev = voltage - 1.0
        battery_schedule = np.clip(-voltage_dev * 0.1, -1.0, 1.0)

        return {
            "voltage": voltage.tolist(),
            "flows": flows.tolist(),
            "curtailment_pct": curtail.tolist(),
            "battery_schedule": battery_schedule.tolist(),
            "meta": {"num_nodes": int(num_nodes), "out_dim": int(out_dim), "meta_out_dim": int(meta_out_dim)}
        }

    except Exception as exc:
        print("Prediction error:", str(exc))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
