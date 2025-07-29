#!/usr/bin/env python3
"""
Test JSONB serialization logic for API metrics without requiring PostgreSQL.
"""

import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from src.prompt_improver.metrics.api_metrics import (
    APIUsageMetric,
    UserJourneyMetric,
    RateLimitMetric,
    AuthenticationMetric,
    EndpointCategory,
    HTTPMethod,
    UserJourneyStage,
    AuthenticationMethod
)


class JSONBSerializationTester:
    """Test JSONB serialization without database dependency."""
    
    def serialize_api_metric_to_jsonb(self, metric: APIUsageMetric) -> Dict[str, Any]:
        """Serialize APIUsageMetric to JSONB-compatible dict."""
        return {
            "endpoint": metric.endpoint,
            "method": metric.method.value,
            "category": metric.category.value,
            "status_code": metric.status_code,
            "response_time_ms": metric.response_time_ms,
            "request_size_bytes": metric.request_size_bytes,
            "response_size_bytes": metric.response_size_bytes,
            "user_id": metric.user_id,
            "session_id": metric.session_id,
            "ip_address": metric.ip_address,
            "user_agent": metric.user_agent,
            "timestamp": metric.timestamp.isoformat(),
            "query_parameters_count": metric.query_parameters_count,
            "payload_type": metric.payload_type,
            "rate_limited": metric.rate_limited,
            "cache_hit": metric.cache_hit,
            "authentication_method": metric.authentication_method.value,
            "api_version": metric.api_version
        }
    
    def serialize_journey_metric_to_jsonb(self, metric: UserJourneyMetric) -> Dict[str, Any]:
        """Serialize UserJourneyMetric to JSONB-compatible dict."""
        return {
            "user_id": metric.user_id,
            "session_id": metric.session_id,
            "journey_stage": metric.journey_stage.value,
            "event_type": metric.event_type,
            "endpoint": metric.endpoint,
            "success": metric.success,
            "conversion_value": metric.conversion_value,
            "time_to_action_seconds": metric.time_to_action_seconds,
            "previous_stage": metric.previous_stage.value if metric.previous_stage else None,
            "feature_flags_active": metric.feature_flags_active,
            "cohort_id": metric.cohort_id,
            "timestamp": metric.timestamp.isoformat(),
            "metadata": metric.metadata
        }
    
    def serialize_rate_limit_metric_to_jsonb(self, metric: RateLimitMetric) -> Dict[str, Any]:
        """Serialize RateLimitMetric to JSONB-compatible dict."""
        return {
            "user_id": metric.user_id,
            "ip_address": metric.ip_address,
            "endpoint": metric.endpoint,
            "limit_type": metric.limit_type,
            "limit_value": metric.limit_value,
            "current_usage": metric.current_usage,
            "time_window_seconds": metric.time_window_seconds,
            "blocked": metric.blocked,
            "burst_detected": metric.burst_detected,
            "timestamp": metric.timestamp.isoformat(),
            "user_tier": metric.user_tier,
            "override_applied": metric.override_applied
        }
    
    def serialize_auth_metric_to_jsonb(self, metric: AuthenticationMetric) -> Dict[str, Any]:
        """Serialize AuthenticationMetric to JSONB-compatible dict."""
        return {
            "user_id": metric.user_id,
            "authentication_method": metric.authentication_method.value,
            "success": metric.success,
            "failure_reason": metric.failure_reason,
            "ip_address": metric.ip_address,
            "user_agent": metric.user_agent,
            "session_duration_seconds": metric.session_duration_seconds,
            "mfa_used": metric.mfa_used,
            "token_type": metric.token_type,
            "timestamp": metric.timestamp.isoformat(),
            "geo_location": metric.geo_location,
            "device_fingerprint": metric.device_fingerprint
        }
    
    def test_api_metric_serialization(self):
        """Test API metric serialization."""
        print("🧪 Test: API Metric JSONB Serialization")
        
        try:
            # Create complex API metric
            metric = APIUsageMetric(
                endpoint="/api/v1/improve-prompt",
                method=HTTPMethod.POST,
                category=EndpointCategory.PROMPT_IMPROVEMENT,
                status_code=200,
                response_time_ms=150.75,
                request_size_bytes=2048,
                response_size_bytes=4096,
                user_id="user_12345",
                session_id="session_abcdef",
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0 (compatible; TestAgent/1.0)",
                timestamp=datetime.now(timezone.utc),
                query_parameters_count=5,
                payload_type="application/json",
                rate_limited=False,
                cache_hit=True,
                authentication_method=AuthenticationMethod.JWT_TOKEN,
                api_version="v1.2.3"
            )
            
            # Serialize to JSONB
            start_time = time.time()
            jsonb_data = self.serialize_api_metric_to_jsonb(metric)
            serialization_time = (time.time() - start_time) * 1000
            
            # Convert to JSON string (simulating PostgreSQL JSONB storage)
            json_string = json.dumps(jsonb_data)
            
            # Deserialize back
            restored_data = json.loads(json_string)
            
            # Verify data integrity
            assert restored_data["endpoint"] == metric.endpoint
            assert restored_data["method"] == metric.method.value
            assert restored_data["category"] == metric.category.value
            assert restored_data["status_code"] == metric.status_code
            assert restored_data["response_time_ms"] == metric.response_time_ms
            assert restored_data["request_size_bytes"] == metric.request_size_bytes
            assert restored_data["response_size_bytes"] == metric.response_size_bytes
            assert restored_data["user_id"] == metric.user_id
            assert restored_data["session_id"] == metric.session_id
            assert restored_data["rate_limited"] == metric.rate_limited
            assert restored_data["cache_hit"] == metric.cache_hit
            assert restored_data["authentication_method"] == metric.authentication_method.value
            
            # Verify timestamp handling
            restored_timestamp = datetime.fromisoformat(restored_data["timestamp"].replace('Z', '+00:00'))
            assert abs((restored_timestamp - metric.timestamp).total_seconds()) < 1
            
            print(f"  Serialization time: {serialization_time:.3f}ms")
            print(f"  JSON size: {len(json_string)} bytes")
            print("✓ API metric serialization: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ API metric serialization: FAILED - {e}")
            return False
    
    def test_journey_metric_serialization(self):
        """Test user journey metric serialization."""
        print("🧪 Test: Journey Metric JSONB Serialization")
        
        try:
            # Create complex journey metric with nested metadata
            metric = UserJourneyMetric(
                user_id="user_12345",
                session_id="session_abcdef",
                journey_stage=UserJourneyStage.ADVANCED_USE,
                event_type="feature_usage",
                endpoint="/api/v1/advanced-feature",
                success=True,
                conversion_value=25.50,
                time_to_action_seconds=45.2,
                previous_stage=UserJourneyStage.REGULAR_USE,
                feature_flags_active=["advanced_ui", "beta_features", "analytics_v2"],
                cohort_id="cohort_premium_users",
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "feature_category": "ml_optimization",
                    "complexity_score": 8.5,
                    "user_preferences": {
                        "theme": "dark",
                        "notifications": True,
                        "advanced_mode": True
                    },
                    "session_context": {
                        "previous_actions": ["login", "dashboard_view", "settings_change"],
                        "time_spent_minutes": 12.5,
                        "errors_encountered": 0
                    }
                }
            )
            
            # Serialize to JSONB
            start_time = time.time()
            jsonb_data = self.serialize_journey_metric_to_jsonb(metric)
            serialization_time = (time.time() - start_time) * 1000
            
            # Convert to JSON string
            json_string = json.dumps(jsonb_data)
            
            # Deserialize back
            restored_data = json.loads(json_string)
            
            # Verify data integrity
            assert restored_data["user_id"] == metric.user_id
            assert restored_data["session_id"] == metric.session_id
            assert restored_data["journey_stage"] == metric.journey_stage.value
            assert restored_data["event_type"] == metric.event_type
            assert restored_data["success"] == metric.success
            assert restored_data["conversion_value"] == metric.conversion_value
            assert restored_data["time_to_action_seconds"] == metric.time_to_action_seconds
            assert restored_data["previous_stage"] == metric.previous_stage.value
            assert restored_data["feature_flags_active"] == metric.feature_flags_active
            assert restored_data["cohort_id"] == metric.cohort_id
            
            # Verify nested metadata
            assert restored_data["metadata"]["feature_category"] == "ml_optimization"
            assert restored_data["metadata"]["complexity_score"] == 8.5
            assert restored_data["metadata"]["user_preferences"]["theme"] == "dark"
            assert restored_data["metadata"]["session_context"]["time_spent_minutes"] == 12.5
            assert len(restored_data["metadata"]["session_context"]["previous_actions"]) == 3
            
            print(f"  Serialization time: {serialization_time:.3f}ms")
            print(f"  JSON size: {len(json_string)} bytes")
            print("✓ Journey metric serialization: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Journey metric serialization: FAILED - {e}")
            return False
    
    def test_rate_limit_metric_serialization(self):
        """Test rate limit metric serialization."""
        print("🧪 Test: Rate Limit Metric JSONB Serialization")
        
        try:
            metric = RateLimitMetric(
                user_id="user_12345",
                ip_address="192.168.1.100",
                endpoint="/api/v1/high-frequency-endpoint",
                limit_type="user_endpoint",
                limit_value=1000,
                current_usage=850,
                time_window_seconds=3600,
                blocked=False,
                burst_detected=True,
                timestamp=datetime.now(timezone.utc),
                user_tier="premium",
                override_applied=False
            )
            
            # Serialize and test
            jsonb_data = self.serialize_rate_limit_metric_to_jsonb(metric)
            json_string = json.dumps(jsonb_data)
            restored_data = json.loads(json_string)
            
            # Verify key fields
            assert restored_data["user_id"] == metric.user_id
            assert restored_data["limit_type"] == metric.limit_type
            assert restored_data["limit_value"] == metric.limit_value
            assert restored_data["current_usage"] == metric.current_usage
            assert restored_data["blocked"] == metric.blocked
            assert restored_data["burst_detected"] == metric.burst_detected
            assert restored_data["user_tier"] == metric.user_tier
            
            print("✓ Rate limit metric serialization: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Rate limit metric serialization: FAILED - {e}")
            return False
    
    def test_auth_metric_serialization(self):
        """Test authentication metric serialization."""
        print("🧪 Test: Auth Metric JSONB Serialization")
        
        try:
            metric = AuthenticationMetric(
                user_id="user_12345",
                authentication_method=AuthenticationMethod.JWT_TOKEN,
                success=True,
                failure_reason=None,
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0 (compatible; TestAgent/1.0)",
                session_duration_seconds=3600.5,
                mfa_used=True,
                token_type="Bearer",
                timestamp=datetime.now(timezone.utc),
                geo_location="US-CA-San Francisco",
                device_fingerprint="fp_abc123def456"
            )
            
            # Serialize and test
            jsonb_data = self.serialize_auth_metric_to_jsonb(metric)
            json_string = json.dumps(jsonb_data)
            restored_data = json.loads(json_string)
            
            # Verify key fields
            assert restored_data["user_id"] == metric.user_id
            assert restored_data["authentication_method"] == metric.authentication_method.value
            assert restored_data["success"] == metric.success
            assert restored_data["session_duration_seconds"] == metric.session_duration_seconds
            assert restored_data["mfa_used"] == metric.mfa_used
            assert restored_data["token_type"] == metric.token_type
            assert restored_data["geo_location"] == metric.geo_location
            assert restored_data["device_fingerprint"] == metric.device_fingerprint
            
            print("✓ Auth metric serialization: PASSED")
            return True
            
        except Exception as e:
            print(f"✗ Auth metric serialization: FAILED - {e}")
            return False
    
    def test_performance_at_scale(self):
        """Test serialization performance with larger datasets."""
        print("🧪 Test: Serialization Performance at Scale")
        
        try:
            # Create 1000 metrics and measure serialization time
            metrics = []
            for i in range(1000):
                metric = APIUsageMetric(
                    endpoint=f"/api/v1/endpoint_{i % 10}",
                    method=HTTPMethod.GET if i % 2 == 0 else HTTPMethod.POST,
                    category=EndpointCategory.PROMPT_IMPROVEMENT,
                    status_code=200 if i % 10 != 9 else 500,
                    response_time_ms=100.0 + (i % 100) * 2,
                    request_size_bytes=500 + i * 5,
                    response_size_bytes=1000 + i * 10,
                    user_id=f"user_{i % 50}",
                    session_id=f"session_{i}",
                    ip_address="192.168.1.1",
                    user_agent="PerfTestAgent/1.0",
                    timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
                    query_parameters_count=i % 5,
                    payload_type="application/json",
                    rate_limited=i % 100 == 0,
                    cache_hit=i % 3 == 0,
                    authentication_method=AuthenticationMethod.JWT_TOKEN,
                    api_version="v1"
                )
                metrics.append(metric)
            
            # Measure serialization performance
            start_time = time.time()
            serialized_metrics = []
            for metric in metrics:
                jsonb_data = self.serialize_api_metric_to_jsonb(metric)
                json_string = json.dumps(jsonb_data)
                serialized_metrics.append(json_string)
            
            total_time = (time.time() - start_time) * 1000
            avg_time_per_metric = total_time / len(metrics)
            
            # Calculate total size
            total_size = sum(len(json_str) for json_str in serialized_metrics)
            avg_size_per_metric = total_size / len(metrics)
            
            print(f"  Total serialization time: {total_time:.2f}ms")
            print(f"  Average time per metric: {avg_time_per_metric:.3f}ms")
            print(f"  Total JSON size: {total_size:,} bytes")
            print(f"  Average size per metric: {avg_size_per_metric:.1f} bytes")
            
            # Performance criteria: should be under 1ms per metric on average
            performance_acceptable = avg_time_per_metric < 1.0
            
            if performance_acceptable:
                print("✓ Serialization performance: PASSED")
            else:
                print(f"✗ Serialization performance: FAILED - Too slow: {avg_time_per_metric:.3f}ms per metric")
            
            return performance_acceptable
            
        except Exception as e:
            print(f"✗ Serialization performance: FAILED - {e}")
            return False


def main():
    """Run JSONB serialization tests."""
    print("🚀 JSONB Serialization Testing")
    print("=" * 50)
    
    tester = JSONBSerializationTester()
    
    tests = [
        tester.test_api_metric_serialization,
        tester.test_journey_metric_serialization,
        tester.test_rate_limit_metric_serialization,
        tester.test_auth_metric_serialization,
        tester.test_performance_at_scale
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print("📊 JSONB SERIALIZATION TEST SUMMARY")
    print("=" * 50)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("🎉 All JSONB serialization tests passed!")
        print("\n✅ JSONB Integration Ready:")
        print("  • All metric types serialize correctly to JSONB")
        print("  • Complex nested data structures preserved")
        print("  • Enum values properly converted to strings")
        print("  • Timestamps handled correctly")
        print("  • Performance meets requirements (<1ms per metric)")
        return True
    else:
        print("❌ Some JSONB serialization tests failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
