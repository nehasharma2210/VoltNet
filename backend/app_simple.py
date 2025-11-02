# backend/app_simple.py - Simplified version for demo
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PowerGrid Surrogate API - Simplified Demo")

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

# Simplified model parameters for demo
NUM_NODES = 39
IN_DIM = 4
OUT_DIM = 3

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "PowerGrid Surrogate API is running"}

@app.post("/predict_opf")
def predict(req: PredictRequest):
    try:
        # Generate synthetic data for demo purposes
        # This simulates what the GNN would output
        
        # Create synthetic voltage data based on parameters
        base_voltage = 1.0 + np.random.normal(0, 0.02, NUM_NODES)
        
        # Adjust voltage based on renewable percentage
        renewable_effect = (req.renewable_pct - 100) / 1000  # small effect
        voltage = base_voltage + renewable_effect
        
        # Adjust voltage based on load factor
        load_effect = (req.load_factor - 100) / 2000  # small effect
        voltage = voltage + load_effect
        
        # Add some noise
        voltage = voltage + np.random.normal(0, 0.01, NUM_NODES)
        
        # Generate synthetic flows (proxy for line flows)
        flows = np.abs(voltage - 1.0) * 2 + np.random.normal(0, 0.1, NUM_NODES)
        flows = np.maximum(flows, 0)  # ensure non-negative
        
        # Generate curtailment recommendations
        curtail = np.zeros(NUM_NODES)
        high_voltage_nodes = voltage > 1.05
        curtail[high_voltage_nodes] = (voltage[high_voltage_nodes] - 1.05) * 20
        
        # Generate battery schedule
        battery_schedule = np.random.normal(0, 0.1, NUM_NODES)
        battery_schedule = np.clip(battery_schedule, -1.0, 1.0)
        
        # Adjust battery schedule based on battery SOC
        battery_scale = (req.battery_soc - 50) / 50  # -1 to 1
        battery_schedule = battery_schedule * (1 + battery_scale * 0.5)
        
        return {
            "voltage": voltage.tolist(),
            "flows": flows.tolist(),
            "curtailment_pct": curtail.tolist(),
            "battery_schedule": battery_schedule.tolist(),
            "meta": {"num_nodes": NUM_NODES, "out_dim": OUT_DIM}
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
