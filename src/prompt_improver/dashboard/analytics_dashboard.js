/**
 * APES Analytics Dashboard JavaScript
 * Handles real-time data updates, chart management, and user interactions
 * Implements 2025 best practices for dashboard performance and UX
 */

function analyticsDashboard() {
    return {
        // State management
        dashboardData: {},
        recentSessions: [],
        timeRange: '24',
        loading: false,
        connectionStatus: 'status-connecting',
        connectionText: 'Connecting...',
        
        // WebSocket connection
        websocket: null,
        reconnectAttempts: 0,
        maxReconnectAttempts: 5,
        
        // Charts
        charts: {},
        
        // Configuration
        config: {},
        
        // Initialization
        init() {
            this.loadConfiguration();
            this.initializeCharts();
            this.connectWebSocket();
            this.loadInitialData();
            
            // Set up periodic refresh as fallback
            setInterval(() => {
                if (this.connectionStatus === 'status-disconnected') {
                    this.refreshData();
                }
            }, 30000); // 30 seconds
        },
        
        async loadConfiguration() {
            try {
                const response = await fetch('/api/v1/analytics/dashboard/config');
                const data = await response.json();
                this.config = data.config;
            } catch (error) {
                console.error('Failed to load dashboard configuration:', error);
            }
        },
        
        async loadInitialData() {
            this.loading = true;
            try {
                await Promise.all([
                    this.loadDashboardMetrics(),
                    this.loadRecentSessions(),
                    this.loadPerformanceTrends(),
                    this.loadPerformanceDistribution()
                ]);
            } catch (error) {
                console.error('Failed to load initial data:', error);
                this.showNotification('error', 'Failed to load dashboard data');
            } finally {
                this.loading = false;
            }
        },
        
        async loadDashboardMetrics() {
            try {
                const response = await fetch(`/api/v1/analytics/dashboard/metrics?time_range_hours=${this.timeRange}&include_comparisons=true`);
                const data = await response.json();
                
                this.dashboardData = {
                    ...data.current_period.session_summary,
                    ...data.current_period.performance,
                    ...data.current_period.efficiency,
                    ...data.current_period.errors,
                    ...data.current_period.resources,
                    
                    // Add trend information from changes
                    performance_trend: this.calculateTrend(data.changes?.performance_change),
                    velocity_trend: this.calculateTrend(data.changes?.improvement_change),
                    efficiency_trend: this.calculateTrend(data.changes?.efficiency_change),
                    success_trend: this.calculateTrend(data.changes?.success_rate_change),
                    
                    // Format change values
                    performance_change: this.formatChange(data.changes?.performance_change_pct),
                    velocity_change: this.formatChange(data.changes?.improvement_change_pct),
                    efficiency_change: this.formatChange(data.changes?.efficiency_change_pct),
                    success_change: this.formatChange(data.changes?.success_rate_change_pct)
                };
                
            } catch (error) {
                console.error('Failed to load dashboard metrics:', error);
                throw error;
            }
        },
        
        async loadRecentSessions() {
            try {
                // This would need a sessions endpoint - for now use mock data
                this.recentSessions = [
                    {
                        session_id: 'session_001',
                        status: 'completed',
                        current_performance: 0.85,
                        improvement: 0.15,
                        duration_hours: 2.5
                    },
                    {
                        session_id: 'session_002',
                        status: 'running',
                        current_performance: 0.78,
                        improvement: 0.08,
                        duration_hours: 1.2
                    }
                ];
            } catch (error) {
                console.error('Failed to load recent sessions:', error);
                throw error;
            }
        },
        
        async loadPerformanceTrends() {
            try {
                const response = await fetch('/api/v1/analytics/trends/analysis', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        time_range: { hours: parseInt(this.timeRange) },
                        granularity: this.timeRange <= 24 ? 'hour' : 'day',
                        metric_type: 'performance'
                    })
                });
                
                const data = await response.json();
                this.updatePerformanceTrendChart(data.time_series);
                
            } catch (error) {
                console.error('Failed to load performance trends:', error);
                throw error;
            }
        },
        
        async loadPerformanceDistribution() {
            try {
                const response = await fetch('/api/v1/analytics/distribution/performance');
                const data = await response.json();
                this.updatePerformanceDistributionChart(data.histogram);
                
            } catch (error) {
                console.error('Failed to load performance distribution:', error);
                throw error;
            }
        },
        
        // WebSocket Management
        connectWebSocket() {
            try {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/api/v1/analytics/live/dashboard`;
                
                this.websocket = new WebSocket(wsUrl);
                
                this.websocket.onopen = () => {
                    this.connectionStatus = 'status-connected';
                    this.connectionText = 'Connected';
                    this.reconnectAttempts = 0;
                    console.log('WebSocket connected');
                };
                
                this.websocket.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    this.handleWebSocketMessage(message);
                };
                
                this.websocket.onclose = () => {
                    this.connectionStatus = 'status-disconnected';
                    this.connectionText = 'Disconnected';
                    this.attemptReconnect();
                };
                
                this.websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.connectionStatus = 'status-disconnected';
                    this.connectionText = 'Connection Error';
                };
                
            } catch (error) {
                console.error('Failed to connect WebSocket:', error);
                this.connectionStatus = 'status-disconnected';
                this.connectionText = 'Connection Failed';
            }
        },
        
        attemptReconnect() {
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                this.connectionStatus = 'status-connecting';
                this.connectionText = `Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`;
                
                setTimeout(() => {
                    this.connectWebSocket();
                }, 2000 * this.reconnectAttempts); // Exponential backoff
            } else {
                this.connectionText = 'Connection Failed';
                this.showNotification('error', 'Lost connection to server. Please refresh the page.');
            }
        },
        
        handleWebSocketMessage(message) {
            switch (message.type) {
                case 'dashboard_data':
                case 'dashboard_update':
                    this.processDashboardUpdate(message.data);
                    break;
                    
                case 'session_update':
                    this.processSessionUpdate(message);
                    break;
                    
                case 'error':
                    console.error('WebSocket error:', message.message);
                    this.showNotification('error', message.message);
                    break;
                    
                default:
                    console.log('Unknown message type:', message.type);
            }
        },
        
        processDashboardUpdate(data) {
            // Update dashboard data
            this.dashboardData = { ...this.dashboardData, ...data.current_period };
            
            // Update charts if needed
            if (data.trend_data) {
                this.updatePerformanceTrendChart(data.trend_data);
            }
        },
        
        processSessionUpdate(message) {
            // Update session in recent sessions list
            const sessionIndex = this.recentSessions.findIndex(s => s.session_id === message.session_id);
            if (sessionIndex >= 0) {
                this.recentSessions[sessionIndex] = { ...this.recentSessions[sessionIndex], ...message.data };
            }
        },
        
        // Chart Management
        initializeCharts() {
            // Performance Trend Chart
            const perfCtx = document.getElementById('performanceTrendChart');
            if (perfCtx) {
                this.charts.performanceTrend = new Chart(perfCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Performance Score',
                            data: [],
                            borderColor: 'rgb(59, 130, 246)',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1
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
            
            // Session Activity Chart
            const activityCtx = document.getElementById('sessionActivityChart');
            if (activityCtx) {
                this.charts.sessionActivity = new Chart(activityCtx, {
                    type: 'bar',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Sessions',
                            data: [],
                            backgroundColor: 'rgba(34, 197, 94, 0.8)',
                            borderColor: 'rgb(34, 197, 94)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                display: false
                            }
                        }
                    }
                });
            }
            
            // Performance Distribution Chart
            const distCtx = document.getElementById('performanceDistributionChart');
            if (distCtx) {
                this.charts.performanceDistribution = new Chart(distCtx, {
                    type: 'bar',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Frequency',
                            data: [],
                            backgroundColor: 'rgba(168, 85, 247, 0.8)',
                            borderColor: 'rgb(168, 85, 247)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                display: false
                            }
                        }
                    }
                });
            }
        },
        
        updatePerformanceTrendChart(timeSeriesData) {
            if (!this.charts.performanceTrend || !timeSeriesData) return;
            
            const labels = timeSeriesData.map(point => {
                const date = new Date(point.timestamp);
                return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            });
            
            const data = timeSeriesData.map(point => point.value);
            
            this.charts.performanceTrend.data.labels = labels;
            this.charts.performanceTrend.data.datasets[0].data = data;
            this.charts.performanceTrend.update();
        },
        
        updatePerformanceDistributionChart(histogramData) {
            if (!this.charts.performanceDistribution || !histogramData) return;
            
            const labels = histogramData.map(bucket => 
                `${bucket.min_value.toFixed(2)}-${bucket.max_value.toFixed(2)}`
            );
            const data = histogramData.map(bucket => bucket.frequency);
            
            this.charts.performanceDistribution.data.labels = labels;
            this.charts.performanceDistribution.data.datasets[0].data = data;
            this.charts.performanceDistribution.update();
        },
        
        // User Actions
        async refreshData() {
            this.loading = true;
            try {
                await this.loadInitialData();
                this.showNotification('success', 'Dashboard data refreshed');
            } catch (error) {
                this.showNotification('error', 'Failed to refresh data');
            } finally {
                this.loading = false;
            }
        },
        
        async updateTimeRange() {
            await this.loadInitialData();
        },
        
        viewSession(sessionId) {
            // Navigate to session detail view
            window.location.href = `/dashboard/session/${sessionId}`;
        },
        
        // Utility Functions
        formatKPI(value) {
            if (value === null || value === undefined) return '--';
            return parseFloat(value).toFixed(3);
        },
        
        formatPercentage(value) {
            if (value === null || value === undefined) return '--';
            return (parseFloat(value) * 100).toFixed(1) + '%';
        },
        
        formatMemory(value) {
            if (value === null || value === undefined) return '--';
            return parseFloat(value).toFixed(0) + ' MB';
        },
        
        formatHours(value) {
            if (value === null || value === undefined) return '--';
            return parseFloat(value).toFixed(1) + 'h';
        },
        
        formatChange(value) {
            if (value === null || value === undefined) return 'No change';
            const sign = value >= 0 ? '+' : '';
            return `${sign}${parseFloat(value).toFixed(1)}%`;
        },
        
        calculateTrend(changeValue) {
            if (!changeValue) return 'stable';
            return changeValue > 0.1 ? 'increasing' : changeValue < -0.1 ? 'decreasing' : 'stable';
        },
        
        getTrendClass(trend) {
            switch (trend) {
                case 'increasing': return 'trend-up';
                case 'decreasing': return 'trend-down';
                default: return 'trend-stable';
            }
        },
        
        getTrendIcon(trend) {
            switch (trend) {
                case 'increasing': return '↗';
                case 'decreasing': return '↘';
                default: return '→';
            }
        },
        
        getStatusClass(status) {
            switch (status) {
                case 'completed': return 'bg-green-100 text-green-800';
                case 'running': return 'bg-blue-100 text-blue-800';
                case 'failed': return 'bg-red-100 text-red-800';
                default: return 'bg-gray-100 text-gray-800';
            }
        },
        
        showNotification(type, message) {
            // Simple notification implementation
            const notification = document.createElement('div');
            notification.className = `fixed top-4 right-4 p-4 rounded-md border z-50 ${
                type === 'error' ? 'bg-red-100 border-red-400 text-red-700' : 
                type === 'success' ? 'bg-green-100 border-green-400 text-green-700' :
                'bg-blue-100 border-blue-400 text-blue-700'
            }`;
            notification.textContent = message;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 5000);
        }
    };
}
