# Minimal demo app for presentation (no ML dependencies)
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import random

app = FastAPI(title="VoltNet Demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "VoltNet Demo API is running!",
        "status": "ready",
        "endpoints": {
            "health": "/health",
            "predict": "/predict_opf",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "loaded": True, "message": "Demo mode - ready for presentation"}

class PredictRequest(BaseModel):
    renewable_pct: float = Field(..., ge=0.0, le=100.0)
    battery_soc: float = Field(..., ge=0.0, le=100.0)
    load_factor: float = Field(..., ge=0.0, le=200.0)
    baseline_idx: int = Field(0, ge=0)

@app.post("/predict_opf")
def predict_demo(req: PredictRequest):
    """Demo prediction with simulated data for presentation"""
    
    # Simulate realistic power grid data
    num_nodes = 30
    
    # Generate realistic voltage values based on inputs
    base_voltage = 1.0
    renewable_impact = (req.renewable_pct / 100.0) * 0.05
    load_impact = (req.load_factor / 100.0) * 0.03
    battery_impact = (req.battery_soc / 100.0) * 0.02
    
    voltage = []
    for i in range(num_nodes):
        node_voltage = base_voltage + random.uniform(-0.02, 0.02)
        node_voltage += renewable_impact * random.uniform(-1, 1)
        node_voltage -= load_impact * random.uniform(0, 1)
        node_voltage += battery_impact * random.uniform(-0.5, 0.5)
        voltage.append(round(node_voltage, 4))
    
    # Calculate flows (power flows between nodes)
    flows = []
    for i in range(num_nodes):
        flow = abs(voltage[i] - np.mean(voltage)) * random.uniform(0.5, 2.0)
        flows.append(round(flow, 4))
    
    # Calculate curtailment (renewable energy curtailment)
    curtailment_pct = []
    for i in range(num_nodes):
        if voltage[i] > 1.05:  # Over-voltage
            curtail = min(10.0, (voltage[i] - 1.05) * 100)
        else:
            curtail = 0.0
        curtailment_pct.append(round(curtail, 2))
    
    # Calculate battery schedule
    battery_schedule = []
    for i in range(num_nodes):
        voltage_dev = voltage[i] - 1.0
        schedule = max(-1.0, min(1.0, -voltage_dev * 0.1))
        battery_schedule.append(round(schedule, 4))
    
    return {
        "voltage": voltage,
        "flows": flows,
        "curtailment_pct": curtailment_pct,
        "battery_schedule": battery_schedule,
        "meta": {
            "num_nodes": num_nodes,
            "mode": "demo",
            "renewable_pct": req.renewable_pct,
            "battery_soc": req.battery_soc,
            "load_factor": req.load_factor
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app-demo:app", host="0.0.0.0", port=8000, reload=True)