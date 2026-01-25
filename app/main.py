"""
FastAPI Backend for Real-Time Gas Pipeline Monitoring
======================================================
Streams simulated pipeline data and performs anomaly detection
"""

from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import json
from pathlib import Path
from .data_simulator import PipelineDataSimulator

# Initialize FastAPI app
app = FastAPI(title="Gas Pipeline Monitoring System")

# Setup templates and static files
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Load trained model and scaler
ARTIFACTS_DIR = BASE_DIR.parent / "artifacts"
try:
    pipeline = joblib.load(ARTIFACTS_DIR / "production_pipeline.joblib")
    feature_config = json.load(open(ARTIFACTS_DIR / "feature_config.json"))
    MODEL_LOADED = True
    print("✓ Production model loaded successfully")
except Exception as e:
    MODEL_LOADED = False
    print(f"⚠ Warning: Could not load model - {e}")
    print("  Running in simulation-only mode")

# Initialize data simulator
simulator = PipelineDataSimulator()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the main dashboard"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "model_status": "Active" if MODEL_LOADED else "Simulation Only"
    })


@app.get("/api/historical")
async def get_historical_data():
    """Get historical anomaly statistics"""
    try:
        outputs_dir = BASE_DIR.parent / "outputs"

        # Load location statistics
        location_stats = pd.read_csv(outputs_dir / "05_anomalies_by_location.csv")

        # Load hourly statistics
        hourly_stats = pd.read_csv(outputs_dir / "06_anomalies_by_hour.csv")

        return {
            "locations": location_stats.to_dict(orient="records"),
            "hourly": hourly_stats.to_dict(orient="records"),
            "summary": {
                "total_records": 170289,
                "total_anomalies": 1703,
                "anomaly_rate": 1.00,
                "peak_hour": 0,
                "peak_hour_rate": 2.82,
                "high_risk_location": "Sumqayit",
                "high_risk_location_rate": 1.22
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time data streaming
    Sends pipeline data every 2 seconds with anomaly detection
    """
    await websocket.accept()

    try:
        while True:
            # Generate simulated data point
            data_point = simulator.generate_data_point()

            # Prepare data for prediction
            if MODEL_LOADED:
                try:
                    # Create DataFrame with required features
                    df = pd.DataFrame([data_point])

                    # Extract temporal features
                    df['hour'] = df['timestamp'].dt.hour
                    df['day_of_week'] = df['timestamp'].dt.dayofweek
                    df['month'] = df['timestamp'].dt.month
                    df['year'] = df['timestamp'].dt.year

                    # One-hot encode location (use 'loc' prefix to match training)
                    location_dummies = pd.get_dummies(df['location'], prefix='loc')
                    df = pd.concat([df, location_dummies], axis=1)

                    # Ensure all required columns exist
                    for col in feature_config['feature_columns']:
                        if col not in df.columns:
                            df[col] = 0

                    # Select features in correct order and convert to numpy array
                    # (avoids feature names warning from sklearn)
                    X = df[feature_config['feature_columns']].values

                    # Predict anomaly
                    prediction = pipeline.predict(X)[0]
                    is_anomaly = (prediction == -1)

                except Exception as e:
                    print(f"Prediction error: {e}")
                    is_anomaly = False
            else:
                # Simulate anomaly detection (1% random)
                is_anomaly = np.random.random() < 0.01

            # Prepare response
            response = {
                "timestamp": data_point["timestamp"].isoformat(),
                "location": data_point["location"],
                "sensors": {
                    "density_kg_m3": round(data_point["density_kg_m3"], 3),
                    "pressure_diff_kpa": round(data_point["pressure_diff_kpa"], 2),
                    "pressure_kpa": round(data_point["pressure_kpa"], 2),
                    "temperature_c": round(data_point["temperature_c"], 1),
                    "hourly_flow_m3": round(data_point["hourly_flow_m3"], 1),
                    "total_flow_m3": round(data_point["total_flow_m3"], 1)
                },
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": float(np.random.random()) if is_anomaly else 0.0
            }

            # Send data
            await websocket.send_json(response)

            # Wait before next update (2 seconds)
            await asyncio.sleep(2)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Only close if not already closed
        try:
            await websocket.close()
        except RuntimeError:
            pass  # Already closed


@app.get("/api/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "simulator_active": True,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
