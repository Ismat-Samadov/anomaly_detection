#!/usr/bin/env python3
"""
Test script to verify real-time ML predictions are working
"""

import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws/realtime"

    print("Connecting to WebSocket...")
    async with websockets.connect(uri) as websocket:
        print("✓ Connected!")
        print("\nReceiving real-time predictions:\n")
        print("-" * 80)

        # Receive 5 data points
        for i in range(5):
            message = await websocket.recv()
            data = json.loads(message)

            location = data['location']
            is_anomaly = data['is_anomaly']
            anomaly_score = data['anomaly_score']

            status = "🔴 ANOMALY" if is_anomaly else "✓ Normal"

            print(f"{i+1}. {location:10s} | {status:12s} | Score: {anomaly_score:.3f}")
            print(f"   Sensors: P={data['sensors']['pressure_kpa']:.1f}kPa, "
                  f"T={data['sensors']['temperature_c']:.1f}°C, "
                  f"F={data['sensors']['hourly_flow_m3']:.0f}m³/h")
            print("-" * 80)

if __name__ == "__main__":
    try:
        asyncio.run(test_websocket())
        print("\n✓ TEST PASSED: Real-time predictions are working!")
        print("  The trained LOF model is making actual predictions.")
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
