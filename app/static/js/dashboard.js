/**
 * Real-Time Dashboard Controller
 * Handles WebSocket streaming, chart updates, and anomaly alerts
 */

// ============================================================
// Configuration
// ============================================================

const CONFIG = {
    // Automatically use wss:// for HTTPS and ws:// for HTTP
    wsUrl: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/realtime`,
    maxDataPoints: 20,
    reconnectDelay: 3000,
    chartUpdateInterval: 100
};

// ============================================================
// State Management
// ============================================================

const state = {
    ws: null,
    reconnectAttempts: 0,
    anomalyCount: 0,
    charts: {},
    chartData: {
        labels: [],
        pressure: { mardakan: [], sumqayit: [], turkan: [] },
        flow: { mardakan: [], sumqayit: [], turkan: [] }
    }
};

// ============================================================
// System Time Update
// ============================================================

function updateSystemTime() {
    const now = new Date();
    const timeString = now.toLocaleString('en-US', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
    });
    document.getElementById('system-time').textContent = timeString;
}

setInterval(updateSystemTime, 1000);
updateSystemTime();

// ============================================================
// Chart Initialization
// ============================================================

function initCharts() {
    const chartConfig = {
        type: 'line',
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 300
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(58, 74, 90, 0.5)'
                    },
                    ticks: {
                        color: '#8899aa'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(58, 74, 90, 0.5)'
                    },
                    ticks: {
                        color: '#8899aa',
                        maxTicksLimit: 10
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#e0e6ed'
                    }
                }
            }
        }
    };

    // Pressure Chart
    const pressureCtx = document.getElementById('pressureChart').getContext('2d');
    state.charts.pressure = new Chart(pressureCtx, {
        ...chartConfig,
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Mardakan',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Sumqayit',
                    data: [],
                    borderColor: '#ffa500',
                    backgroundColor: 'rgba(255, 165, 0, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Turkan',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    tension: 0.4
                }
            ]
        }
    });

    // Flow Chart
    const flowCtx = document.getElementById('flowChart').getContext('2d');
    state.charts.flow = new Chart(flowCtx, {
        ...chartConfig,
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Mardakan',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Sumqayit',
                    data: [],
                    borderColor: '#ffa500',
                    backgroundColor: 'rgba(255, 165, 0, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Turkan',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    tension: 0.4
                }
            ]
        }
    });
}

// ============================================================
// WebSocket Connection
// ============================================================

function connectWebSocket() {
    showConnectionStatus('Connecting to real-time data stream...');

    try {
        state.ws = new WebSocket(CONFIG.wsUrl);

        state.ws.onopen = () => {
            console.log('✓ WebSocket connected');
            hideConnectionStatus();
            state.reconnectAttempts = 0;
        };

        state.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleRealtimeData(data);
        };

        state.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            showConnectionStatus('Connection error. Retrying...');
        };

        state.ws.onclose = () => {
            console.log('WebSocket closed. Reconnecting...');
            showConnectionStatus('Connection lost. Reconnecting...');
            setTimeout(() => {
                state.reconnectAttempts++;
                connectWebSocket();
            }, CONFIG.reconnectDelay);
        };

    } catch (error) {
        console.error('Failed to create WebSocket:', error);
        showConnectionStatus('Failed to connect. Retrying...');
        setTimeout(connectWebSocket, CONFIG.reconnectDelay);
    }
}

// ============================================================
// Real-Time Data Handler
// ============================================================

function handleRealtimeData(data) {
    const location = data.location.toLowerCase();
    const timestamp = new Date(data.timestamp);
    const timeLabel = timestamp.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });

    // Update station metrics
    updateStationMetrics(location, data.sensors);

    // Update charts
    updateCharts(location, timeLabel, data.sensors);

    // Handle anomalies
    if (data.is_anomaly) {
        handleAnomaly(location, data);
    } else {
        clearAnomaly(location);
    }
}

// ============================================================
// Station Metrics Update
// ============================================================

function updateStationMetrics(location, sensors) {
    const flowEl = document.getElementById(`flow-${location}`);
    const pressureEl = document.getElementById(`pressure-${location}`);
    const tempEl = document.getElementById(`temp-${location}`);

    if (flowEl) flowEl.textContent = `${sensors.hourly_flow_m3.toFixed(0)} m³/h`;
    if (pressureEl) pressureEl.textContent = `${sensors.pressure_kpa.toFixed(1)} kPa`;
    if (tempEl) tempEl.textContent = `${sensors.temperature_c.toFixed(1)} °C`;

    // Add pulse animation
    [flowEl, pressureEl, tempEl].forEach(el => {
        if (el) {
            el.style.transition = 'color 0.3s ease';
            el.style.color = '#00ff88';
            setTimeout(() => {
                el.style.color = '#00d4ff';
            }, 300);
        }
    });
}

// ============================================================
// Chart Updates
// ============================================================

function updateCharts(location, timeLabel, sensors) {
    // Add new data point
    if (!state.chartData.labels.includes(timeLabel)) {
        state.chartData.labels.push(timeLabel);

        // Maintain max data points
        if (state.chartData.labels.length > CONFIG.maxDataPoints) {
            state.chartData.labels.shift();
        }
    }

    // Update pressure data
    state.chartData.pressure[location].push(sensors.pressure_kpa);
    if (state.chartData.pressure[location].length > CONFIG.maxDataPoints) {
        state.chartData.pressure[location].shift();
    }

    // Update flow data
    state.chartData.flow[location].push(sensors.hourly_flow_m3);
    if (state.chartData.flow[location].length > CONFIG.maxDataPoints) {
        state.chartData.flow[location].shift();
    }

    // Update chart objects
    updateChartData();
}

function updateChartData() {
    // Update Pressure Chart
    state.charts.pressure.data.labels = state.chartData.labels;
    state.charts.pressure.data.datasets[0].data = state.chartData.pressure.mardakan;
    state.charts.pressure.data.datasets[1].data = state.chartData.pressure.sumqayit;
    state.charts.pressure.data.datasets[2].data = state.chartData.pressure.turkan;
    state.charts.pressure.update('none');

    // Update Flow Chart
    state.charts.flow.data.labels = state.chartData.labels;
    state.charts.flow.data.datasets[0].data = state.chartData.flow.mardakan;
    state.charts.flow.data.datasets[1].data = state.chartData.flow.sumqayit;
    state.charts.flow.data.datasets[2].data = state.chartData.flow.turkan;
    state.charts.flow.update('none');
}

// ============================================================
// Anomaly Handling
// ============================================================

function handleAnomaly(location, data) {
    state.anomalyCount++;

    // Update anomaly count
    document.getElementById('anomaly-count').textContent = state.anomalyCount;

    // Update station indicator
    const indicator = document.getElementById(`anomaly-${location}`);
    if (indicator) {
        indicator.className = 'anomaly-indicator alert';
        indicator.innerHTML = `⚠️ ANOMALY DETECTED - Score: ${data.anomaly_score.toFixed(3)}`;
    }

    // Update station status
    const stationCard = document.getElementById(`station-${location}`);
    if (stationCard) {
        const statusEl = stationCard.querySelector('.station-status');
        if (statusEl) {
            statusEl.className = 'station-status status-alert';
            statusEl.textContent = 'ANOMALY';
        }
    }

    // Add to alert log
    addAlertToLog(location, data);

    // Highlight SVG station
    highlightStation(location, true);
}

function clearAnomaly(location) {
    const indicator = document.getElementById(`anomaly-${location}`);
    if (indicator && indicator.classList.contains('alert')) {
        indicator.className = 'anomaly-indicator';
        indicator.innerHTML = '✓ Operating normally';

        // Reset station status (except Sumqayit which is elevated)
        const stationCard = document.getElementById(`station-${location}`);
        if (stationCard) {
            const statusEl = stationCard.querySelector('.station-status');
            if (statusEl && location !== 'sumqayit') {
                statusEl.className = 'station-status status-normal';
                statusEl.textContent = 'NORMAL';
            } else if (statusEl && location === 'sumqayit') {
                statusEl.className = 'station-status status-warning';
                statusEl.textContent = 'ELEVATED RISK';
            }
        }

        highlightStation(location, false);
    }
}

function addAlertToLog(location, data) {
    const alertList = document.getElementById('alert-list');

    // Remove empty message if present
    const emptyMsg = alertList.querySelector('.alert-empty');
    if (emptyMsg) emptyMsg.remove();

    // Create alert item
    const alertItem = document.createElement('div');
    alertItem.className = 'alert-item';

    const timestamp = new Date(data.timestamp).toLocaleString('en-US', {
        hour12: false,
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });

    alertItem.innerHTML = `
        <div class="alert-time">${timestamp}</div>
        <div class="alert-location">${location.toUpperCase()} STATION</div>
        <div class="alert-message">Anomalous operation detected - ML Score: ${data.anomaly_score.toFixed(3)}</div>
    `;

    // Add to top of list
    alertList.insertBefore(alertItem, alertList.firstChild);

    // Limit to 10 alerts
    while (alertList.children.length > 10) {
        alertList.removeChild(alertList.lastChild);
    }
}

function highlightStation(location, isAlert) {
    const svgStation = document.getElementById(`svg-${location}`);
    if (svgStation) {
        const rect = svgStation.querySelector('rect');
        const circle = svgStation.querySelector('circle');

        if (isAlert) {
            rect.setAttribute('stroke', '#ff3232');
            circle.setAttribute('fill', '#ff3232');
        } else {
            const normalColor = location === 'sumqayit' ? '#ffa500' : '#00ff88';
            rect.setAttribute('stroke', normalColor);
            circle.setAttribute('fill', normalColor);
        }
    }
}

// ============================================================
// Connection Status
// ============================================================

function showConnectionStatus(message) {
    const statusEl = document.getElementById('connection-status');
    const textEl = document.getElementById('status-text');

    if (statusEl && textEl) {
        textEl.textContent = message;
        statusEl.classList.remove('hidden');
    }
}

function hideConnectionStatus() {
    const statusEl = document.getElementById('connection-status');
    if (statusEl) {
        statusEl.classList.add('hidden');
    }
}

// ============================================================
// Initialization
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Gas Pipeline SCADA Dashboard...');

    // Initialize charts
    initCharts();

    // Connect to WebSocket
    connectWebSocket();

    console.log('✓ Dashboard initialized');
});

// ============================================================
// Cleanup on page unload
// ============================================================

window.addEventListener('beforeunload', () => {
    if (state.ws) {
        state.ws.close();
    }
});
