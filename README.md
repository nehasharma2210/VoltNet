# VoltNet - Power Grid Optimization

A machine learning-powered web application for power grid optimization using Graph Neural Networks.

## 🚀 Quick Start with Docker

### Prerequisites
- Docker
- Docker Compose

### Deployment

**Windows:**
```cmd
deploy.bat
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

**Manual:**
```bash
docker-compose up --build -d
```

### Access
- **Frontend**: http://localhost
- **Backend API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
VoltNet/
├── backend/                 # FastAPI backend
│   ├── app.py              # Main API application
│   ├── model.py            # ML model definitions
│   ├── model_loader.py     # Model loading utilities
│   ├── artifacts/          # Pre-trained models and data
│   ├── Dockerfile          # Backend container config
│   └── requirements.txt    # Python dependencies
├── frontend/               # Web frontend
│   ├── index.html          # Main web page
│   ├── script.js           # Frontend logic
│   ├── style.css           # Styling
│   ├── api.js              # API communication
│   ├── config.js           # Configuration
│   ├── static/             # Static assets
│   ├── Dockerfile          # Frontend container config
│   └── nginx.conf          # Nginx configuration
├── docker-compose.yml      # Multi-container setup
├── deploy.sh              # Linux/Mac deployment script
└── deploy.bat             # Windows deployment script
```

## 🔧 Features

- **Graph Neural Network**: Advanced ML model for power grid optimization
- **Real-time Predictions**: Fast API responses for grid scenarios
- **Interactive UI**: Web-based interface for parameter adjustment
- **Docker Deployment**: One-command deployment
- **Health Monitoring**: Built-in health checks
- **Auto-scaling**: Container restart policies

## 🛠️ Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Development
Serve the frontend directory with any web server:
```bash
cd frontend
python -m http.server 8080
```

## 📊 API Endpoints

- `GET /` - API status
- `GET /health` - Health check
- `POST /predict_opf` - Power flow optimization prediction

### Prediction Request Format
```json
{
  "renewable_pct": 50.0,
  "battery_soc": 75.0,
  "load_factor": 100.0,
  "baseline_idx": 0
}
```

## 🐳 Docker Services

- **Backend**: FastAPI app on port 8000
- **Frontend**: Nginx server on port 80
- **Volumes**: ML artifacts mounted for persistence
- **Networks**: Internal Docker network for service communication

## 🔍 Troubleshooting

Check service logs:
```bash
docker-compose logs backend
docker-compose logs frontend
```

Restart services:
```bash
docker-compose restart
```

## 📝 License

This project is part of a hackathon submission for power grid optimization.