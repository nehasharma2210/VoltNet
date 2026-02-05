# Offline demo app - No external dependencies needed
import json
import random
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

class VoltNetHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                "message": "VoltNet API is running!",
                "status": "ready",
                "endpoints": {
                    "health": "/health",
                    "predict": "/predict_opf",
                    "docs": "/docs"
                }
            }
            self.wfile.write(json.dumps(response).encode())
            
        elif path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"status": "healthy", "loaded": True, "message": "Offline demo ready"}
            self.wfile.write(json.dumps(response).encode())
            
        elif path == '/docs':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            html = """
            <html><body>
            <h1>VoltNet API Documentation</h1>
            <h2>Endpoints:</h2>
            <ul>
                <li>GET / - API status</li>
                <li>GET /health - Health check</li>
                <li>POST /predict_opf - Power flow prediction</li>
            </ul>
            <h2>Prediction Request:</h2>
            <pre>{
  "renewable_pct": 50.0,
  "battery_soc": 75.0,
  "load_factor": 100.0,
  "baseline_idx": 0
}</pre>
            </body></html>
            """
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/predict_opf':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode())
                renewable_pct = data.get('renewable_pct', 50)
                battery_soc = data.get('battery_soc', 75)
                load_factor = data.get('load_factor', 100)
                
                # Generate realistic demo data
                num_nodes = 30
                voltage = []
                flows = []
                curtailment_pct = []
                battery_schedule = []
                
                for i in range(num_nodes):
                    # Voltage simulation
                    base_v = 1.0 + (renewable_pct/100 * 0.05) - (load_factor/100 * 0.03)
                    v = base_v + random.uniform(-0.02, 0.02)
                    voltage.append(round(v, 4))
                    
                    # Flow simulation
                    flow = abs(v - 1.0) * random.uniform(0.5, 2.0)
                    flows.append(round(flow, 4))
                    
                    # Curtailment
                    curtail = max(0, (v - 1.05) * 100) if v > 1.05 else 0
                    curtailment_pct.append(round(curtail, 2))
                    
                    # Battery schedule
                    schedule = max(-1, min(1, -(v - 1.0) * 0.1))
                    battery_schedule.append(round(schedule, 4))
                
                response = {
                    "voltage": voltage,
                    "flows": flows,
                    "curtailment_pct": curtailment_pct,
                    "battery_schedule": battery_schedule,
                    "meta": {
                        "num_nodes": num_nodes,
                        "mode": "offline_demo",
                        "renewable_pct": renewable_pct,
                        "battery_soc": battery_soc,
                        "load_factor": load_factor
                    }
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                error_response = {"error": str(e)}
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server():
    server = HTTPServer(('localhost', 8000), VoltNetHandler)
    print("🚀 VoltNet Offline Server running on http://localhost:8000")
    print("📊 API Docs: http://localhost:8000/docs")
    print("🏥 Health: http://localhost:8000/health")
    server.serve_forever()

if __name__ == "__main__":
    run_server()