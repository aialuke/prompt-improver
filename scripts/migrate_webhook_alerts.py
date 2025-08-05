#!/usr/bin/env python3
"""
Migration script to update webhook alerts to use unified HTTP client
Part of Phase 3: HTTP Client Standardization
"""

import re
import os
import sys
from pathlib import Path

def update_webhook_alert_channel():
    """Update WebhookAlertChannel to use unified HTTP client"""
    
    file_path = Path(__file__).parent.parent / "src/prompt_improver/performance/baseline/regression_detector.py"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match the webhook alert implementation
    old_pattern = r"""            async with aiohttp\.ClientSession\(\) as session:
                async with session\.post\(
                    self\.webhook_url,
                    json=payload,
                    timeout=aiohttp\.ClientTimeout\(total=self\.timeout\)
                \) as response:
                    if response\.status == 200:
                        logger\.info\(f"Webhook alert sent successfully for \{alert\.metric_name\}"\)
                        return True
                    else:
                        logger\.error\(f"Webhook alert failed with status \{response\.status\}"\)
                        return False"""
    
    # New implementation using unified HTTP client
    new_implementation = """            # Use unified HTTP client for webhook alerts with circuit breaker and monitoring
            from ...monitoring.unified_http_client import make_webhook_request
            
            try:
                async with make_webhook_request(self.webhook_url, payload) as response:
                    if response.status == 200:
                        logger.info(f"Webhook alert sent successfully for {alert.metric_name}")
                        return True
                    else:
                        logger.error(f"Webhook alert failed with status {response.status}")
                        return False
            except Exception as request_error:
                logger.error(f"Unified HTTP client request failed: {request_error}")
                return False"""
    
    # Perform the replacement
    updated_content = re.sub(old_pattern, new_implementation, content, flags=re.MULTILINE)
    
    if updated_content == content:
        print("No changes made - pattern not found exactly as expected")
        # Let's try a simpler pattern
        simple_pattern = r"async with aiohttp\.ClientSession\(\) as session:"
        if simple_pattern in content:
            print("Found aiohttp.ClientSession usage, but pattern doesn't match exactly")
            # Find and replace the entire WebhookAlertChannel send_alert method
            webhook_method_pattern = r"""    async def send_alert\(self, alert: RegressionAlert\) -> bool:
        \"\"\"Send alert via webhook\.\"\"\"
        if not WEBHOOK_AVAILABLE:
            logger\.warning\("Webhook alerts not available \(aiohttp not installed\)"\)
            return False
        
        try:
            payload = \{[^}]+\}
            
            async with aiohttp\.ClientSession\(\) as session:
                async with session\.post\(
                    self\.webhook_url,
                    json=payload,
                    timeout=aiohttp\.ClientTimeout\(total=self\.timeout\)
                \) as response:
                    if response\.status == 200:
                        logger\.info\(f"Webhook alert sent successfully for \{alert\.metric_name\}"\)
                        return True
                    else:
                        logger\.error\(f"Webhook alert failed with status \{response\.status\}"\)
                        return False
        
        except Exception as e:
            logger\.error\(f"Failed to send webhook alert: \{e\}"\)
            return False"""
            
            new_method = """    async def send_alert(self, alert: RegressionAlert) -> bool:
        \"\"\"Send alert via webhook using unified HTTP client.\"\"\"
        try:
            payload = {
                'alert_type': 'performance_regression',
                'severity': alert.severity.value,
                'metric_name': alert.metric_name,
                'message': alert.message,
                'current_value': alert.current_value,
                'baseline_value': alert.baseline_value,
                'degradation_percentage': alert.degradation_percentage,
                'timestamp': alert.alert_timestamp.isoformat(),
                'alert_id': alert.alert_id,
                'affected_operations': alert.affected_operations,
                'probable_causes': alert.probable_causes,
                'remediation_suggestions': alert.remediation_suggestions
            }
            
            # Use unified HTTP client for webhook alerts with circuit breaker and monitoring
            from ...monitoring.unified_http_client import make_webhook_request
            
            async with make_webhook_request(self.webhook_url, payload) as response:
                if response.status == 200:
                    logger.info(f"Webhook alert sent successfully for {alert.metric_name}")
                    return True
                else:
                    logger.error(f"Webhook alert failed with status {response.status}")
                    return False
        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False"""
            
            updated_content = re.sub(webhook_method_pattern, new_method, content, flags=re.MULTILINE | re.DOTALL)
    
    if updated_content != content:
        # Write the updated content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Successfully updated {file_path}")
        return True
    else:
        print("No updates made - content unchanged")
        return False

def update_capture_baselines_script():
    """Update capture_baselines.py to use unified HTTP client"""
    
    file_path = Path(__file__).parent / "capture_baselines.py"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match the health check implementation
    old_pattern = r"""        async with aiohttp\.ClientSession\(timeout=aiohttp\.ClientTimeout\(total=30\)\) as session:
            for endpoint in endpoints:
                try:
                    # Capture metrics before request
                    memory_before = self\.process\.memory_info\(\)\.rss / \(1024\*1024\)
                    cpu_before = self\.process\.cpu_percent\(\)
                    
                    start_time = time\.time\(\)
                    async with session\.get\(f"\{base_url\}\{endpoint\}"\) as response:
                        content = await response\.read\(\)
                        response_time = \(time\.time\(\) - start_time\) \* 1000"""
    
    # New implementation using unified HTTP client
    new_implementation = """        # Use unified HTTP client for health checks
        from prompt_improver.monitoring.unified_http_client import make_health_check_request
        
        for endpoint in endpoints:
            try:
                # Capture metrics before request
                memory_before = self.process.memory_info().rss / (1024*1024)
                cpu_before = self.process.cpu_percent()
                
                start_time = time.time()
                async with make_health_check_request(f"{base_url}{endpoint}") as response:
                    content = await response.read()
                    response_time = (time.time() - start_time) * 1000"""
    
    # Perform the replacement
    updated_content = re.sub(old_pattern, new_implementation, content, flags=re.MULTILINE)
    
    if updated_content != content:
        # Write the updated content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Successfully updated {file_path}")
        return True
    else:
        print("No updates made to capture_baselines.py")
        return False

def update_install_production_tools():
    """Update install_production_tools.py to use unified HTTP client"""
    
    file_path = Path(__file__).parent / "install_production_tools.py"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match the download implementation
    old_pattern = r"""            async with aiohttp\.ClientSession\(\) as session:
                async with session\.get\(download_url\) as response:
                    if response\.status == 200:"""
    
    # New implementation using unified HTTP client
    new_implementation = """            # Use unified HTTP client for downloads
            from prompt_improver.monitoring.unified_http_client import download_file
            
            async with download_file(download_url) as response:
                if response.status == 200:"""
    
    # Perform the replacement
    updated_content = re.sub(old_pattern, new_implementation, content, flags=re.MULTILINE)
    
    if updated_content != content:
        # Write the updated content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Successfully updated {file_path}")
        return True
    else:
        print("No updates made to install_production_tools.py")
        return False

def main():
    """Main migration function"""
    print("Starting Phase 3: HTTP Client Standardization Migration")
    print("=" * 60)
    
    success_count = 0
    total_count = 3
    
    # Update webhook alerts
    print("\n1. Updating WebhookAlertChannel in regression_detector.py...")
    if update_webhook_alert_channel():
        success_count += 1
    
    # Update capture baselines script
    print("\n2. Updating capture_baselines.py...")
    if update_capture_baselines_script():
        success_count += 1
    
    # Update install production tools
    print("\n3. Updating install_production_tools.py...")
    if update_install_production_tools():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Migration Summary: {success_count}/{total_count} files updated successfully")
    
    if success_count == total_count:
        print("✅ Phase 3 HTTP Client Standardization migration completed successfully!")
        return 0
    else:
        print("⚠️  Some files were not updated. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())