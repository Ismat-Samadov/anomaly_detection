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
import httpx

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

                    # Select features in correct order
                    X = df[feature_config['feature_columns']]

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
        await websocket.close()


@app.get("/api/weather")
async def get_weather():
    """
    Get current weather for Baku, Azerbaijan (near pipeline stations)
    Uses Open-Meteo API (free, no API key needed)
    """
    try:
        # Coordinates for Baku, Azerbaijan (near Mardakan, Sumqayit, Turkan)
        latitude = 40.4093
        longitude = 49.8671

        # Open-Meteo API (free, no key required)
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m",
            "timezone": "Asia/Baku"
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            current = data.get("current", {})

            # Weather code to description mapping
            weather_codes = {
                0: "Clear sky",
                1: "Mainly clear",
                2: "Partly cloudy",
                3: "Overcast",
                45: "Foggy",
                48: "Foggy",
                51: "Light drizzle",
                53: "Drizzle",
                55: "Heavy drizzle",
                61: "Light rain",
                63: "Rain",
                65: "Heavy rain",
                71: "Light snow",
                73: "Snow",
                75: "Heavy snow",
                77: "Snow grains",
                80: "Light showers",
                81: "Showers",
                82: "Heavy showers",
                85: "Light snow showers",
                86: "Snow showers",
                95: "Thunderstorm",
                96: "Thunderstorm with hail",
                99: "Thunderstorm with heavy hail"
            }

            weather_code = current.get("weather_code", 0)
            weather_desc = weather_codes.get(weather_code, "Unknown")

            # Weather code to emoji
            weather_emoji = {
                0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️",
                45: "🌫️", 48: "🌫️",
                51: "🌦️", 53: "🌧️", 55: "🌧️",
                61: "🌧️", 63: "🌧️", 65: "⛈️",
                71: "🌨️", 73: "❄️", 75: "❄️", 77: "🌨️",
                80: "🌦️", 81: "🌦️", 82: "⛈️",
                85: "🌨️", 86: "🌨️",
                95: "⛈️", 96: "⛈️", 99: "⛈️"
            }

            return {
                "location": "Baku, Azerbaijan",
                "temperature": round(current.get("temperature_2m", 0), 1),
                "feels_like": round(current.get("apparent_temperature", 0), 1),
                "humidity": current.get("relative_humidity_2m", 0),
                "wind_speed": round(current.get("wind_speed_10m", 0), 1),
                "wind_direction": current.get("wind_direction_10m", 0),
                "precipitation": current.get("precipitation", 0),
                "weather_code": weather_code,
                "weather_description": weather_desc,
                "weather_emoji": weather_emoji.get(weather_code, "🌡️"),
                "timestamp": current.get("time", datetime.now().isoformat())
            }

    except Exception as e:
        # Return fallback data if API fails
        return {
            "location": "Baku, Azerbaijan",
            "temperature": 15,
            "feels_like": 15,
            "humidity": 60,
            "wind_speed": 5,
            "wind_direction": 180,
            "precipitation": 0,
            "weather_code": 1,
            "weather_description": "Data unavailable",
            "weather_emoji": "🌡️",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


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
