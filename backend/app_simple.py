# backend/app_simple.py - No static files, just API
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

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "PowerGrid Surrogate API is running!",
        "endpoints": {
            "health": "/health",
            "predict": "/predict_opf",
            "docs": "/docs"
        }
    }

class PredictRequest(BaseModel):
    renewable_pct: float = Field(..., ge=0.0, le=100.0)
    battery_soc: float = Field(..., ge=0.0, le=100.0)
    load_factor: float = Field(..., ge=0.0, le=200.0)
    baseline_idx: int = Field(0, ge=0)

ART = {}
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

@app.on_event("startup")
def startup_event():
    global ART
    try:
        ART = load_artifacts(artifact_dir=ARTIFACT_DIR, device='cpu', verbose=True)
        print("✅ Artifacts loaded successfully at startup.")
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
        out_dim = int(ART["out_dim"])
        meta_out_dim = int(ART.get("meta_out_dim", out_dim))

        renewable_scale = float(req.renewable_pct) / 100.0
        load_scale = float(req.load_factor) / 100.0
        battery_scale = float(req.battery_soc) / 100.0

        bx = np.array(baseline_X)
        if bx.ndim == 2 and bx.shape[0] > 1:
            idx = min(req.baseline_idx, bx.shape[0]-1)
            bx = bx[idx]
        bx = bx.flatten()
        if bx.size < num_nodes * in_dim:
            print(f"WARNING: baseline size {bx.size} < expected {num_nodes*in_dim}. Using fallback baseline.")
            bx = ART["baseline_X"].flatten()
        bx = bx[: num_nodes*in_dim].reshape(num_nodes, in_dim).astype(float)

        X = bx * load_scale
        if in_dim > 0:
            X[:, -1] = X[:, -1] * renewable_scale
        if in_dim > 1:
            X[:, 1] = X[:, 1] * battery_scale

        X_flat = X.reshape(1, -1)
        X_scaled_flat = scaler_x.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(num_nodes, in_dim).astype(np.float32)

        ei_t = torch.tensor(edge_index, dtype=torch.long)
        ea_t = torch.tensor(edge_attr, dtype=torch.float)
        device = next(model.parameters()).device
        x_t = torch.tensor(X_scaled, dtype=torch.float, device=device)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(x_t, ei_t.to(device), ea_t.to(device)).cpu().numpy()

        pred_flat = pred_scaled.reshape(1, -1)
        required_len = num_nodes * meta_out_dim
        got_len = pred_flat.shape[1]

        if got_len != required_len:
            print(f"WARNING: Model produced flattened len {got_len} but scaler_y expects {required_len}. Padding/truncating.")
            if got_len < required_len:
                pad = np.zeros((1, required_len - got_len), dtype=pred_flat.dtype)
                pred_flat_for_scaler = np.concatenate([pred_flat, pad], axis=1)
            else:
                pred_flat_for_scaler = pred_flat[:, :required_len]
        else:
            pred_flat_for_scaler = pred_flat

        pred_real_flat = scaler_y.inverse_transform(pred_flat_for_scaler)
        
        if pred_real_flat.shape[1] % meta_out_dim == 0:
            pred_real = pred_real_flat.reshape(-1, meta_out_dim)
            if meta_out_dim == out_dim:
                pass
            elif meta_out_dim > out_dim:
                pred_real = pred_real[:, :out_dim]
            else:
                padding_cols = np.zeros((pred_real.shape[0], out_dim - meta_out_dim), dtype=pred_real.dtype)
                pred_real = np.concatenate([pred_real, padding_cols], axis=1)
            
            if pred_real.shape[0] < num_nodes:
                padding_rows = np.zeros((num_nodes - pred_real.shape[0], out_dim), dtype=pred_real.dtype)
                pred_real = np.concatenate([pred_real, padding_rows], axis=0)
            elif pred_real.shape[0] > num_nodes:
                pred_real = pred_real[:num_nodes]
        else:
            actual_elements = pred_real_flat.shape[1]
            if actual_elements % out_dim == 0:
                num_complete_nodes = actual_elements // out_dim
                pred_real = pred_real_flat.reshape(num_complete_nodes, out_dim)
                if num_complete_nodes < num_nodes:
                    padding = np.zeros((num_nodes - num_complete_nodes, out_dim), dtype=pred_real.dtype)
                    pred_real = np.concatenate([pred_real, padding], axis=0)
                elif num_complete_nodes > num_nodes:
                    pred_real = pred_real[:num_nodes]
            else:
                num_complete = actual_elements // out_dim
                remainder = actual_elements % out_dim
                if num_complete > 0:
                    pred_real_partial = pred_real_flat[:, :num_complete * out_dim].reshape(num_complete, out_dim)
                    if remainder > 0:
                        remainder_padded = np.pad(
                            pred_real_flat[:, num_complete * out_dim:].flatten(),
                            (0, out_dim - remainder),
                            mode='constant'
                        ).reshape(1, out_dim)
                        pred_real = np.concatenate([pred_real_partial, remainder_padded], axis=0)
                    else:
                        pred_real = pred_real_partial
                    if pred_real.shape[0] < num_nodes:
                        padding = np.zeros((num_nodes - pred_real.shape[0], out_dim), dtype=pred_real.dtype)
                        pred_real = np.concatenate([pred_real, padding], axis=0)
                    elif pred_real.shape[0] > num_nodes:
                        pred_real = pred_real[:num_nodes]
                else:
                    padded = np.pad(pred_real_flat.flatten(), (0, out_dim - remainder), mode='constant')
                    pred_real = padded[:out_dim].reshape(1, out_dim)
                    if num_nodes > 1:
                        padding = np.zeros((num_nodes - 1, out_dim), dtype=pred_real.dtype)
                        pred_real = np.concatenate([pred_real, padding], axis=0)

        voltage = pred_real[:, 0] if out_dim > 0 else np.zeros(num_nodes)
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
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_simple:app", host="0.0.0.0", port=port)