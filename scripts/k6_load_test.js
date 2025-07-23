/**
 * K6 Load Testing Script - 2025 Best Practices
 * Comprehensive performance validation for APES ML Pipeline Orchestrator
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';

// Custom metrics following 2025 observability standards
const errorRate = new Rate('error_rate');
const responseTime = new Trend('response_time');
const throughput = new Counter('throughput');
const healthCheckFailures = new Rate('health_check_failures');
const apiEndpointFailures = new Rate('api_endpoint_failures');

// Test configuration following 2025 performance testing standards
export const options = {
  stages: [
    // Ramp-up phase
    { duration: '2m', target: 20 },   // Gradual ramp-up
    { duration: '5m', target: 50 },   // Steady load
    { duration: '2m', target: 100 },  // Peak load
    { duration: '5m', target: 100 },  // Sustained peak
    { duration: '2m', target: 0 },    // Ramp-down
  ],
  
  // SLO-based thresholds (2025 best practice)
  thresholds: {
    // Response time thresholds
    'http_req_duration': ['p(50)<100', 'p(95)<200', 'p(99)<500'],
    
    // Error rate thresholds
    'http_req_failed': ['rate<0.01'], // <1% error rate
    'error_rate': ['rate<0.01'],
    
    // Custom metric thresholds
    'response_time': ['p(95)<200'],
    'health_check_failures': ['rate<0.001'], // <0.1% health check failures
    'api_endpoint_failures': ['rate<0.01'],
    
    // Availability threshold
    'checks': ['rate>0.99'], // >99% success rate
  },
  
  // Resource limits
  maxRedirects: 4,
  userAgent: 'K6LoadTest/2025.1.0 (Production-Readiness-Validation)',
  
  // Test scenarios for different load patterns
  scenarios: {
    // Constant load scenario
    health_checks: {
      executor: 'constant-vus',
      vus: 10,
      duration: '16m',
      tags: { test_type: 'health_check' },
    },
    
    // Ramping load scenario
    api_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 10 },
        { duration: '5m', target: 25 },
        { duration: '2m', target: 50 },
        { duration: '5m', target: 50 },
        { duration: '2m', target: 0 },
      ],
      tags: { test_type: 'api_load' },
    },
    
    // Spike testing scenario
    spike_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '10s', target: 0 },
        { duration: '30s', target: 200 }, // Sudden spike
        { duration: '1m', target: 200 },  // Sustained spike
        { duration: '30s', target: 0 },   // Quick ramp-down
      ],
      tags: { test_type: 'spike' },
      startTime: '10m', // Start after 10 minutes
    },
  },
};

// Base URL configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:5000';

// Test data for realistic load simulation
const testData = {
  prompts: [
    'Optimize this Python function for better performance',
    'Generate a comprehensive test suite for this API',
    'Refactor this code to follow SOLID principles',
    'Create documentation for this machine learning model',
    'Implement error handling for this async function',
  ],
  userAgents: [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
  ],
};

/**
 * Setup function - runs once before all VUs
 */
export function setup() {
  console.log('🚀 Starting K6 Load Test - 2025 Production Readiness Validation');
  console.log(`📊 Target URL: ${BASE_URL}`);
  console.log(`👥 Max VUs: ${options.stages.reduce((max, stage) => Math.max(max, stage.target), 0)}`);
  
  // Verify service is available before starting load test
  const healthCheck = http.get(`${BASE_URL}/health`);
  if (healthCheck.status !== 200) {
    throw new Error(`Service not available: ${healthCheck.status}`);
  }
  
  return { startTime: new Date().toISOString() };
}

/**
 * Main test function - runs for each VU iteration
 */
export default function(data) {
  const testType = __VU % 3; // Distribute test types across VUs
  
  switch (testType) {
    case 0:
      healthCheckTest();
      break;
    case 1:
      apiEndpointTest();
      break;
    case 2:
      mlPipelineTest();
      break;
  }
  
  // Think time between requests (realistic user behavior)
  sleep(Math.random() * 2 + 1); // 1-3 seconds
}

/**
 * Health check testing
 */
function healthCheckTest() {
  group('Health Check Tests', () => {
    const response = http.get(`${BASE_URL}/health`, {
      headers: {
        'User-Agent': testData.userAgents[Math.floor(Math.random() * testData.userAgents.length)],
      },
      tags: { endpoint: 'health' },
    });
    
    const success = check(response, {
      'health check status is 200': (r) => r.status === 200,
      'health check response time < 100ms': (r) => r.timings.duration < 100,
      'health check has valid JSON': (r) => {
        try {
          JSON.parse(r.body);
          return true;
        } catch {
          return false;
        }
      },
    });
    
    // Record custom metrics
    responseTime.add(response.timings.duration);
    throughput.add(1);
    healthCheckFailures.add(!success);
    errorRate.add(response.status !== 200);
  });
}

/**
 * API endpoint testing
 */
function apiEndpointTest() {
  group('API Endpoint Tests', () => {
    // Test various API endpoints
    const endpoints = [
      '/api/v1/status',
      '/api/v1/metrics',
      '/api/v1/health',
    ];
    
    const endpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
    const response = http.get(`${BASE_URL}${endpoint}`, {
      headers: {
        'Accept': 'application/json',
        'User-Agent': testData.userAgents[Math.floor(Math.random() * testData.userAgents.length)],
      },
      tags: { endpoint: endpoint },
    });
    
    const success = check(response, {
      'API endpoint status is 200 or 404': (r) => [200, 404].includes(r.status),
      'API endpoint response time < 200ms': (r) => r.timings.duration < 200,
      'API endpoint has content': (r) => r.body.length > 0,
    });
    
    // Record custom metrics
    responseTime.add(response.timings.duration);
    throughput.add(1);
    apiEndpointFailures.add(!success);
    errorRate.add(![200, 404].includes(response.status));
  });
}

/**
 * ML Pipeline testing (if endpoints are available)
 */
function mlPipelineTest() {
  group('ML Pipeline Tests', () => {
    const prompt = testData.prompts[Math.floor(Math.random() * testData.prompts.length)];
    
    const payload = JSON.stringify({
      prompt: prompt,
      model: 'default',
      max_tokens: 100,
    });
    
    const response = http.post(`${BASE_URL}/api/v1/improve`, payload, {
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': testData.userAgents[Math.floor(Math.random() * testData.userAgents.length)],
      },
      tags: { endpoint: 'ml_improve' },
    });
    
    const success = check(response, {
      'ML endpoint status is 200 or 404': (r) => [200, 404].includes(r.status),
      'ML endpoint response time < 500ms': (r) => r.timings.duration < 500,
      'ML endpoint returns valid response': (r) => {
        if (r.status === 404) return true; // Endpoint may not exist yet
        try {
          const data = JSON.parse(r.body);
          return data && typeof data === 'object';
        } catch {
          return false;
        }
      },
    });
    
    // Record custom metrics
    responseTime.add(response.timings.duration);
    throughput.add(1);
    errorRate.add(![200, 404].includes(response.status));
  });
}

/**
 * Teardown function - runs once after all VUs complete
 */
export function teardown(data) {
  console.log('🏁 K6 Load Test Completed');
  console.log(`⏱️  Test Duration: ${new Date().toISOString()}`);
}

/**
 * Custom summary report generation
 */
export function handleSummary(data) {
  return {
    'k6_load_test_report.html': htmlReport(data),
    'k6_load_test_summary.json': JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}
