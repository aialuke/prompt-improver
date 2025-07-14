/**
 * Real-time A/B Testing Dashboard JavaScript
 * Handles WebSocket connections, chart updates, and user interactions
 */

class ABTestingDashboard {
    constructor() {
        this.websocket = null;
        this.currentExperimentId = null;
        this.isConnected = false;
        this.charts = {};
        this.metrics = {};
        this.alerts = [];
        
        // Chart data storage
        this.effectSizeData = [];
        this.timestamps = [];
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.loadExperiments();
    }
    
    setupEventListeners() {
        // Experiment selector
        document.getElementById('experiment-selector').addEventListener('change', (e) => {
            const experimentId = e.target.value;
            if (experimentId) {
                this.connectToExperiment(experimentId);
            } else {
                this.disconnect();
            }
        });
        
        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.refreshMetrics();
        });
        
        // Handle page unload
        window.addEventListener('beforeunload', () => {
            this.disconnect();
        });
    }
    
    async loadExperiments() {
        try {
            const response = await fetch('/api/v1/experiments/real-time/monitoring/active');
            const data = await response.json();
            
            if (data.status === 'success') {
                const selector = document.getElementById('experiment-selector');
                selector.innerHTML = '<option value="">Select Experiment...</option>';
                
                data.experiments.forEach(exp => {
                    const option = document.createElement('option');
                    option.value = exp.experiment_id;
                    option.textContent = `Experiment ${exp.experiment_id} (${exp.active_connections} connections)`;
                    selector.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Failed to load experiments:', error);
            this.showAlert('error', 'Failed to load experiments');
        }
    }
    
    connectToExperiment(experimentId) {
        this.disconnect(); // Close existing connection
        
        this.currentExperimentId = experimentId;
        
        // Determine WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v1/experiments/real-time/live/${experimentId}`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            this.isConnected = true;
            this.updateConnectionStatus('Connected', 'connected');
            console.log(`Connected to experiment ${experimentId}`);
            
            // Start monitoring
            this.startMonitoring(experimentId);
            
            // Request initial metrics
            this.requestMetrics();
        };
        
        this.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        this.websocket.onclose = () => {
            this.isConnected = false;
            this.updateConnectionStatus('Disconnected', 'disconnected');
            console.log('WebSocket connection closed');
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus('Error', 'error');
        };
    }
    
    disconnect() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        this.isConnected = false;
        this.currentExperimentId = null;
        this.updateConnectionStatus('Disconnected', 'disconnected');
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'welcome':
                console.log('Welcome message:', data.message);
                break;
                
            case 'metrics_update':
                this.updateMetrics(data.metrics);
                if (data.alerts && data.alerts.length > 0) {
                    this.handleAlerts(data.alerts);
                }
                break;
                
            case 'pong':
                console.log('Pong received');
                break;
                
            case 'error':
                console.error('WebSocket error:', data.message);
                this.showAlert('error', data.message);
                break;
                
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    async startMonitoring(experimentId) {
        try {
            const response = await fetch(`/api/v1/experiments/real-time/experiments/${experimentId}/monitoring/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            if (data.status === 'success') {
                console.log('Monitoring started successfully');
            }
        } catch (error) {
            console.error('Failed to start monitoring:', error);
        }
    }
    
    requestMetrics() {
        if (this.websocket && this.isConnected) {
            this.websocket.send(JSON.stringify({
                type: 'request_metrics'
            }));
        }
    }
    
    async refreshMetrics() {
        if (!this.currentExperimentId) return;
        
        try {
            const response = await fetch(`/api/v1/experiments/real-time/experiments/${this.currentExperimentId}/metrics`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.updateMetrics(data.metrics);
            }
        } catch (error) {
            console.error('Failed to refresh metrics:', error);
            this.showAlert('error', 'Failed to refresh metrics');
        }
    }
    
    updateMetrics(metrics) {
        this.metrics = metrics;
        
        // Update overview cards
        this.updateOverviewCards(metrics);
        
        // Update charts
        this.updateCharts(metrics);
        
        // Update detailed metrics table
        this.updateMetricsTable(metrics);
        
        // Update early stopping section
        this.updateEarlyStoppingSection(metrics);
    }
    
    updateOverviewCards(metrics) {
        // Status card
        document.getElementById('experiment-status').textContent = 'Running'; // Could come from metrics
        
        const progress = metrics.progress?.completion_percentage || 0;
        document.getElementById('progress-bar').style.width = `${progress}%`;
        document.getElementById('progress-text').textContent = `${progress.toFixed(1)}% Complete`;
        
        // Sample sizes
        const sampleSizes = metrics.sample_sizes || {};
        document.getElementById('total-sample-size').textContent = sampleSizes.total || 0;
        document.getElementById('control-size').textContent = sampleSizes.control || 0;
        document.getElementById('treatment-size').textContent = sampleSizes.treatment || 0;
        
        // Statistical significance
        const statistical = metrics.statistical_analysis || {};
        const pValue = statistical.p_value;
        const isSignificant = statistical.statistical_significance;
        
        document.getElementById('p-value').textContent = pValue ? pValue.toFixed(4) : '-';
        
        const significanceIcon = document.getElementById('significance-icon');
        const significanceStatus = document.getElementById('significance-status');
        
        if (isSignificant) {
            significanceIcon.className = 'p-3 bg-green-100 rounded-full';
            significanceIcon.querySelector('svg').className = 'h-6 w-6 text-green-600';
            significanceStatus.className = 'inline-flex px-2 py-1 text-xs font-medium rounded-full bg-green-100 text-green-800';
            significanceStatus.textContent = 'Significant';
        } else {
            significanceIcon.className = 'p-3 bg-gray-100 rounded-full';
            significanceIcon.querySelector('svg').className = 'h-6 w-6 text-gray-600';
            significanceStatus.className = 'inline-flex px-2 py-1 text-xs font-medium rounded-full bg-gray-100 text-gray-800';
            significanceStatus.textContent = 'Not Significant';
        }
    }
    
    updateCharts(metrics) {
        const timestamp = new Date();
        const statistical = metrics.statistical_analysis || {};
        
        // Update effect size chart
        this.effectSizeData.push(statistical.effect_size || 0);
        this.timestamps.push(timestamp);
        
        // Keep only last 50 data points
        if (this.effectSizeData.length > 50) {
            this.effectSizeData.shift();
            this.timestamps.shift();
        }
        
        if (this.charts.effectSize) {
            this.charts.effectSize.data.labels = this.timestamps.map(t => t.toLocaleTimeString());
            this.charts.effectSize.data.datasets[0].data = this.effectSizeData;
            this.charts.effectSize.update('none');
        }
        
        // Update confidence interval chart
        if (this.charts.confidenceInterval && statistical.confidence_interval) {
            const [lower, upper] = statistical.confidence_interval;
            const effectSize = statistical.effect_size || 0;
            
            this.charts.confidenceInterval.data.datasets[0].data = [lower, effectSize, upper];
            this.charts.confidenceInterval.update('none');
        }
    }
    
    updateMetricsTable(metrics) {
        const tbody = document.getElementById('metrics-table-body');
        const means = metrics.means || {};
        const statistical = metrics.statistical_analysis || {};
        
        const rows = [
            {
                metric: 'Conversion Rate',
                control: (means.control || 0).toFixed(4),
                treatment: (means.treatment || 0).toFixed(4),
                difference: ((means.treatment || 0) - (means.control || 0)).toFixed(4),
                ci: statistical.confidence_interval ? 
                    `[${statistical.confidence_interval[0].toFixed(4)}, ${statistical.confidence_interval[1].toFixed(4)}]` : '-'
            },
            {
                metric: 'Effect Size (Cohen\'s d)',
                control: '-',
                treatment: '-',
                difference: (statistical.effect_size || 0).toFixed(4),
                ci: '-'
            },
            {
                metric: 'Statistical Power',
                control: '-',
                treatment: '-',
                difference: (statistical.statistical_power || 0).toFixed(4),
                ci: '-'
            }
        ];
        
        tbody.innerHTML = rows.map(row => `
            <tr>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${row.metric}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${row.control}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${row.treatment}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${row.difference}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${row.ci}</td>
            </tr>
        `).join('');
    }
    
    updateEarlyStoppingSection(metrics) {
        const earlyStoppig = metrics.early_stopping || {};
        const section = document.getElementById('early-stopping-section');
        const content = document.getElementById('early-stopping-content');
        
        if (earlyStoppig.recommendation) {
            section.classList.remove('hidden');
            content.innerHTML = `
                <div class="bg-yellow-50 border border-yellow-200 rounded-md p-4">
                    <div class="flex">
                        <div class="flex-shrink-0">
                            <svg class="h-5 w-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path>
                            </svg>
                        </div>
                        <div class="ml-3">
                            <h3 class="text-sm font-medium text-yellow-800">Early Stopping Recommendation</h3>
                            <div class="mt-2 text-sm text-yellow-700">
                                <p>${earlyStoppig.recommendation}</p>
                                ${earlyStoppig.confidence ? `<p class="mt-1">Confidence: ${(earlyStoppig.confidence * 100).toFixed(1)}%</p>` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            section.classList.add('hidden');
        }
    }
    
    handleAlerts(alerts) {
        this.alerts = alerts;
        this.updateAlertsDisplay();
    }
    
    updateAlertsDisplay() {
        const container = document.getElementById('alerts-container');
        const list = document.getElementById('alerts-list');
        
        if (this.alerts.length === 0) {
            container.classList.add('hidden');
            return;
        }
        
        container.classList.remove('hidden');
        
        list.innerHTML = this.alerts.map(alert => {
            const alertClass = `alert-${alert.severity}`;
            return `
                <div class="${alertClass} border rounded-md p-4">
                    <div class="flex">
                        <div class="flex-shrink-0">
                            ${this.getAlertIcon(alert.severity)}
                        </div>
                        <div class="ml-3">
                            <h3 class="text-sm font-medium">${alert.title}</h3>
                            <p class="text-sm mt-1">${alert.message}</p>
                            <p class="text-xs mt-2 opacity-75">${new Date(alert.timestamp).toLocaleString()}</p>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }
    
    getAlertIcon(severity) {
        const icons = {
            critical: '<svg class="h-5 w-5 text-red-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path></svg>',
            warning: '<svg class="h-5 w-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path></svg>',
            info: '<svg class="h-5 w-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"></path></svg>'
        };
        return icons[severity] || icons.info;
    }
    
    initializeCharts() {
        // Effect Size Chart
        const effectSizeCtx = document.getElementById('effect-size-chart').getContext('2d');
        this.charts.effectSize = new Chart(effectSizeCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Effect Size',
                    data: [],
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Effect Size (Cohen\'s d)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
        
        // Confidence Interval Chart
        const ciCtx = document.getElementById('confidence-interval-chart').getContext('2d');
        this.charts.confidenceInterval = new Chart(ciCtx, {
            type: 'bar',
            data: {
                labels: ['Lower CI', 'Effect Size', 'Upper CI'],
                datasets: [{
                    label: 'Confidence Interval',
                    data: [0, 0, 0],
                    backgroundColor: ['rgba(239, 68, 68, 0.5)', 'rgba(59, 130, 246, 0.8)', 'rgba(34, 197, 94, 0.5)'],
                    borderColor: ['rgb(239, 68, 68)', 'rgb(59, 130, 246)', 'rgb(34, 197, 94)'],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Effect Size'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    updateConnectionStatus(status, type) {
        const statusElement = document.getElementById('connection-status');
        const indicator = statusElement.previousElementSibling;
        
        statusElement.textContent = status;
        
        // Update indicator color
        indicator.className = 'status-indicator';
        switch (type) {
            case 'connected':
                indicator.classList.add('status-running');
                break;
            case 'disconnected':
            case 'error':
                indicator.classList.add('status-stopped');
                break;
            default:
                indicator.classList.add('status-pending');
        }
    }
    
    showAlert(type, message) {
        // Simple alert implementation - could be enhanced with a proper notification system
        console.log(`${type.toUpperCase()}: ${message}`);
        
        // Could implement toast notifications here
        const alertDiv = document.createElement('div');
        alertDiv.className = `fixed top-4 right-4 p-4 rounded-md border z-50 ${type === 'error' ? 'bg-red-100 border-red-400 text-red-700' : 'bg-blue-100 border-blue-400 text-blue-700'}`;
        alertDiv.textContent = message;
        
        document.body.appendChild(alertDiv);
        
        setTimeout(() => {
            document.body.removeChild(alertDiv);
        }, 5000);
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new ABTestingDashboard();
});