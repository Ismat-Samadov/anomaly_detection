#!/bin/bash

# Gas Pipeline Monitoring System - Docker Deployment Script
# ===========================================================

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================================"
echo "  Gas Pipeline Monitoring - Docker Deployment"
echo "============================================================"
echo -e "${NC}"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running${NC}"
        echo ""
        echo "Please start Docker Desktop and try again:"
        echo "  - macOS: Open Docker Desktop application"
        echo "  - Linux: sudo systemctl start docker"
        echo ""
        exit 1
    fi
}

# Function to check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${YELLOW}Warning: docker-compose command not found${NC}"
        echo "Using 'docker compose' instead..."
        DOCKER_COMPOSE="docker compose"
    else
        DOCKER_COMPOSE="docker-compose"
    fi
}

# Function to stop existing containers
stop_existing() {
    echo -e "${YELLOW}Stopping existing containers...${NC}"
    $DOCKER_COMPOSE down 2>/dev/null || true
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

# Function to build and start
deploy() {
    echo ""
    echo -e "${BLUE}Building Docker image...${NC}"
    $DOCKER_COMPOSE build

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Build successful${NC}"
    else
        echo -e "${RED}✗ Build failed${NC}"
        exit 1
    fi

    echo ""
    echo -e "${BLUE}Starting containers...${NC}"
    $DOCKER_COMPOSE up -d

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Containers started${NC}"
    else
        echo -e "${RED}✗ Failed to start containers${NC}"
        exit 1
    fi
}

# Function to show status
show_status() {
    echo ""
    echo -e "${BLUE}Container Status:${NC}"
    $DOCKER_COMPOSE ps

    echo ""
    echo -e "${BLUE}Checking health...${NC}"
    sleep 3

    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Application is healthy${NC}"
    else
        echo -e "${YELLOW}⚠ Application may still be starting...${NC}"
    fi
}

# Function to show access information
show_info() {
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}  Deployment Successful!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo "Dashboard:        http://localhost:8000"
    echo "API Documentation: http://localhost:8000/docs"
    echo "Health Check:     http://localhost:8000/api/health"
    echo ""
    echo "View logs:        ${DOCKER_COMPOSE} logs -f"
    echo "Stop containers:  ${DOCKER_COMPOSE} down"
    echo "Restart:          ${DOCKER_COMPOSE} restart"
    echo ""
    echo -e "${BLUE}Opening dashboard in browser...${NC}"

    # Open browser (cross-platform)
    if command -v open &> /dev/null; then
        sleep 2 && open http://localhost:8000 &
    elif command -v xdg-open &> /dev/null; then
        sleep 2 && xdg-open http://localhost:8000 &
    elif command -v start &> /dev/null; then
        sleep 2 && start http://localhost:8000 &
    fi

    echo ""
}

# Main execution
main() {
    # Parse arguments
    PRODUCTION=false
    LOGS=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --production|-p)
                PRODUCTION=true
                shift
                ;;
            --logs|-l)
                LOGS=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  -p, --production    Deploy with Nginx reverse proxy"
                echo "  -l, --logs          Show logs after deployment"
                echo "  -h, --help          Show this help message"
                echo ""
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Check prerequisites
    check_docker
    check_docker_compose

    # Check for ML model
    if [ ! -f "artifacts/production_pipeline.joblib" ]; then
        echo -e "${YELLOW}Warning: ML model not found${NC}"
        echo "Dashboard will run in simulation-only mode"
        echo ""
    fi

    # Stop existing containers
    stop_existing

    # Deploy
    if [ "$PRODUCTION" = true ]; then
        echo -e "${BLUE}Deploying with Nginx reverse proxy...${NC}"
        DOCKER_COMPOSE="$DOCKER_COMPOSE --profile production"
    fi

    deploy
    show_status
    show_info

    # Show logs if requested
    if [ "$LOGS" = true ]; then
        echo "Press Ctrl+C to stop viewing logs (containers will keep running)"
        echo ""
        sleep 2
        $DOCKER_COMPOSE logs -f
    fi
}

# Run main function
main "$@"
