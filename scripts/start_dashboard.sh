#!/bin/bash

# Gas Pipeline Monitoring Dashboard - Startup Script
# ===================================================

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================================"
echo "  Gas Pipeline Monitoring System - Starting Dashboard"
echo "============================================================"
echo -e "${NC}"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not found at .venv${NC}"
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if dependencies are installed
echo "Checking dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Dependencies OK${NC}"
fi

# Check if ML model exists
if [ ! -f "artifacts/production_pipeline.joblib" ]; then
    echo -e "${YELLOW}Warning: ML model not found at artifacts/production_pipeline.joblib${NC}"
    echo "Please run the Jupyter notebook to generate the model first:"
    echo "  jupyter notebook notebooks/anomaly_detection_pipeline.ipynb"
    echo ""
    echo "Dashboard will run in simulation-only mode."
    sleep 2
else
    echo -e "${GREEN}✓ ML Model loaded${NC}"
fi

# Display startup information
echo ""
echo -e "${BLUE}Starting FastAPI server...${NC}"
echo ""
echo "Dashboard will be available at:"
echo -e "  ${GREEN}http://localhost:8000${NC}"
echo ""
echo "API Documentation:"
echo -e "  ${GREEN}http://localhost:8000/docs${NC}"
echo ""
echo "Press CTRL+C to stop the server"
echo ""
echo "============================================================"
echo ""

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
