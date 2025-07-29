#!/usr/bin/env python3
"""
Database Health Check and Diagnostic Tools
Comprehensive utilities for diagnosing database connectivity and configuration issues.
"""
import asyncio
import logging
import subprocess
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prompt_improver.core.config import AppConfig

logger = logging.getLogger(__name__)


class DatabaseDiagnostics:
    """Comprehensive database health check and diagnostic tools."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize diagnostics with database configuration."""
        self.config = config or AppConfig().database
        self.results = []
        
    async def run_comprehensive_check(self) -> Dict[str, any]:
        """Run all diagnostic checks and return comprehensive results."""
        print("🔍 Starting comprehensive database health check...")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "docker_status": await self._check_docker_status(),
            "container_status": await self._check_container_status(),
            "connection_tests": await self._test_connections(),
            "credential_validation": await self._validate_credentials(),
            "database_inventory": await self._inventory_databases(),
            "cleanup_verification": await self._verify_cleanup_capability(),
            "performance_check": await self._check_performance(),
            "recommendations": []
        }
        
        results["recommendations"] = self._generate_recommendations(results)
        
        self._print_summary(results)
        return results
        
    async def _check_docker_status(self) -> Dict[str, any]:
        """Check Docker daemon and container status."""
        print("🐳 Checking Docker status...")
        
        try:
            # Check Docker daemon
            result = subprocess.run(
                ["docker", "version"], 
                capture_output=True, text=True, timeout=10
            )
            docker_running = result.returncode == 0
            
            # Check PostgreSQL container
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=apes_postgres", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"], 
                capture_output=True, text=True, timeout=10
            )
            
            container_info = result.stdout.strip()
            container_running = "apes_postgres" in container_info and "Up" in container_info
            
            status = {
                "docker_daemon": docker_running,
                "postgres_container": container_running,
                "container_details": container_info if container_running else "Container not running",
                "status": "✅ HEALTHY" if (docker_running and container_running) else "❌ UNHEALTHY"
            }
            
            print(f"  Docker daemon: {'✅' if docker_running else '❌'}")
            print(f"  PostgreSQL container: {'✅' if container_running else '❌'}")
            
            return status
            
        except Exception as e:
            status = {
                "docker_daemon": False,
                "postgres_container": False,
                "error": str(e),
                "status": "❌ ERROR"
            }
            print(f"  Docker check failed: {e}")
            return status
            
    async def _check_container_status(self) -> Dict[str, any]:
        """Check internal container status and configuration."""
        print("🔧 Checking container configuration...")
        
        try:
            # Check PostgreSQL process inside container
            result = subprocess.run([
                "docker", "exec", "apes_postgres", "pg_isready", "-U", "apes_user"
            ], capture_output=True, text=True, timeout=10)
            
            pg_ready = result.returncode == 0
            
            # Check database existence
            result = subprocess.run([
                "docker", "exec", "apes_postgres", "psql", "-U", "apes_user", "-d", "apes_production",
                "-c", "SELECT version();"
            ], capture_output=True, text=True, timeout=10)
            
            db_accessible = result.returncode == 0
            pg_version = result.stdout.strip() if db_accessible else "Unknown"
            
            # Check user permissions
            result = subprocess.run([
                "docker", "exec", "apes_postgres", "psql", "-U", "apes_user", "-d", "apes_production",
                "-c", "SELECT current_user, session_user;"
            ], capture_output=True, text=True, timeout=10)
            
            user_info = result.stdout.strip() if result.returncode == 0 else "Permission check failed"
            
            status = {
                "postgres_ready": pg_ready,
                "database_accessible": db_accessible,
                "postgres_version": pg_version,
                "user_permissions": user_info,
                "status": "✅ HEALTHY" if (pg_ready and db_accessible) else "❌ UNHEALTHY"
            }
            
            print(f"  PostgreSQL ready: {'✅' if pg_ready else '❌'}")
            print(f"  Database accessible: {'✅' if db_accessible else '❌'}")
            
            return status
            
        except Exception as e:
            status = {
                "postgres_ready": False,
                "database_accessible": False,
                "error": str(e),
                "status": "❌ ERROR"
            }
            print(f"  Container check failed: {e}")
            return status
            
    async def _test_connections(self) -> Dict[str, any]:
        """Test various connection methods."""
        print("🔌 Testing database connections...")
        
        results = {}
        
        # Test asyncpg connection
        try:
            conn = await asyncpg.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                user=self.config.postgres_username,
                password=self.config.postgres_password,
                database=self.config.postgres_database,
                timeout=5.0
            )
            await conn.close()
            results["asyncpg"] = {"success": True, "status": "✅ CONNECTED"}
            print("  asyncpg connection: ✅")
        except Exception as e:
            results["asyncpg"] = {"success": False, "error": str(e), "status": "❌ FAILED"}
            print(f"  asyncpg connection: ❌ ({e})")
            
        # Test SQLAlchemy connection
        try:
            engine = create_async_engine(
                f"postgresql+asyncpg://{self.config.postgres_username}:{self.config.postgres_password}@{self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_database}"
            )
            async with engine.connect() as conn:
                from sqlalchemy import text
                await conn.execute(text("SELECT 1"))
            await engine.dispose()
            results["sqlalchemy"] = {"success": True, "status": "✅ CONNECTED"}
            print("  SQLAlchemy connection: ✅")
        except Exception as e:
            results["sqlalchemy"] = {"success": False, "error": str(e), "status": "❌ FAILED"}
            print(f"  SQLAlchemy connection: ❌ ({e})")
            
        # Test Docker exec connection
        try:
            result = subprocess.run([
                "docker", "exec", "apes_postgres", "psql", "-U", self.config.postgres_username, "-d", self.config.postgres_database,
                "-c", "SELECT 1;"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                results["docker_exec"] = {"success": True, "status": "✅ CONNECTED"}
                print("  Docker exec connection: ✅")
            else:
                results["docker_exec"] = {"success": False, "error": result.stderr, "status": "❌ FAILED"}
                print(f"  Docker exec connection: ❌ ({result.stderr})")
        except Exception as e:
            results["docker_exec"] = {"success": False, "error": str(e), "status": "❌ FAILED"}
            print(f"  Docker exec connection: ❌ ({e})")
            
        return results
        
    async def _validate_credentials(self) -> Dict[str, any]:
        """Validate database credentials and configuration."""
        print("🔑 Validating credentials...")
        
        expected_config = {
            "host": "localhost",
            "port": 5432,
            "user": "apes_user",
            "password": "apes_secure_password_2024",
            "database": "apes_production"
        }
        
        current_config = {
            "host": self.config.postgres_host,
            "port": self.config.postgres_port,
            "user": self.config.postgres_username,
            "password": self.config.postgres_password,
            "database": self.config.postgres_database
        }
        
        mismatches = []
        for key, expected in expected_config.items():
            if current_config[key] != expected:
                mismatches.append(f"{key}: expected '{expected}', got '{current_config[key]}'")
        
        status = {
            "expected_config": expected_config,
            "current_config": current_config,
            "mismatches": mismatches,
            "status": "✅ VALID" if not mismatches else "❌ INVALID"
        }
        
        if mismatches:
            print("  Credential validation: ❌")
            for mismatch in mismatches:
                print(f"    {mismatch}")
        else:
            print("  Credential validation: ✅")
            
        return status
        
    async def _inventory_databases(self) -> Dict[str, any]:
        """Inventory all databases and test databases."""
        print("📊 Inventorying databases...")
        
        try:
            # Get all databases
            result = subprocess.run([
                "docker", "exec", "apes_postgres", "psql", "-U", "apes_user", "-d", "apes_production",
                "-c", "SELECT datname FROM pg_database ORDER BY datname;"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                databases = [line.strip() for line in result.stdout.split('\n') 
                           if line.strip() and not line.startswith('-') and 'datname' not in line and '(' not in line]
                
                test_databases = [db for db in databases if db.startswith('apes_test_')]
                production_databases = [db for db in databases if not db.startswith('apes_test_')]
                
                status = {
                    "total_databases": len(databases),
                    "production_databases": production_databases,
                    "test_databases": test_databases,
                    "test_db_count": len(test_databases),
                    "status": "✅ HEALTHY"
                }
                
                print(f"  Total databases: {len(databases)}")
                print(f"  Production databases: {len(production_databases)}")
                print(f"  Test databases: {len(test_databases)}")
                
                if len(test_databases) > 10:
                    print(f"  ⚠️  Warning: {len(test_databases)} test databases found (cleanup recommended)")
                    
            else:
                status = {
                    "error": result.stderr,
                    "status": "❌ FAILED"
                }
                print(f"  Database inventory failed: {result.stderr}")
                
        except Exception as e:
            status = {
                "error": str(e),
                "status": "❌ ERROR"
            }
            print(f"  Database inventory error: {e}")
            
        return status
        
    async def _verify_cleanup_capability(self) -> Dict[str, any]:
        """Verify database cleanup functionality."""
        print("🧹 Verifying cleanup capability...")
        
        try:
            # Test creating a temporary database
            test_db = "apes_test_diagnostic_temp"
            
            # Create test database
            result = subprocess.run([
                "docker", "exec", "apes_postgres", "psql", "-U", "apes_user", "-d", "apes_production",
                "-c", f'CREATE DATABASE "{test_db}";'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return {
                    "create_capability": False,
                    "error": result.stderr,
                    "status": "❌ FAILED"
                }
            
            # Test dropping the database
            result = subprocess.run([
                "docker", "exec", "apes_postgres", "psql", "-U", "apes_user", "-d", "apes_production",
                "-c", f'DROP DATABASE "{test_db}";'
            ], capture_output=True, text=True, timeout=10)
            
            drop_success = result.returncode == 0
            
            status = {
                "create_capability": True,
                "drop_capability": drop_success,
                "status": "✅ WORKING" if drop_success else "⚠️ PARTIAL"
            }
            
            print(f"  Database creation: ✅")
            print(f"  Database deletion: {'✅' if drop_success else '❌'}")
            
            return status
            
        except Exception as e:
            status = {
                "create_capability": False,
                "drop_capability": False,
                "error": str(e),
                "status": "❌ ERROR"
            }
            print(f"  Cleanup capability check failed: {e}")
            return status
            
    async def _check_performance(self) -> Dict[str, any]:
        """Check database performance metrics."""
        print("⚡ Checking performance metrics...")
        
        try:
            # Test connection time
            start_time = asyncio.get_event_loop().time()
            conn = await asyncpg.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                user=self.config.postgres_username,
                password=self.config.postgres_password,
                database=self.config.postgres_database,
                timeout=5.0
            )
            connection_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Test query performance
            start_time = asyncio.get_event_loop().time()
            await conn.execute("SELECT 1")
            query_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            await conn.close()
            
            # Check container resource usage
            result = subprocess.run([
                "docker", "stats", "apes_postgres", "--no-stream", "--format", "table {{.CPUPerc}}\t{{.MemUsage}}"
            ], capture_output=True, text=True, timeout=10)
            
            resource_info = result.stdout.strip() if result.returncode == 0 else "Unable to get stats"
            
            status = {
                "connection_time_ms": round(connection_time, 2),
                "query_time_ms": round(query_time, 2),
                "resource_usage": resource_info,
                "status": "✅ GOOD" if connection_time < 1000 else "⚠️ SLOW"
            }
            
            print(f"  Connection time: {connection_time:.2f}ms")
            print(f"  Query time: {query_time:.2f}ms")
            
            return status
            
        except Exception as e:
            status = {
                "connection_time_ms": None,
                "query_time_ms": None,
                "error": str(e),
                "status": "❌ ERROR"
            }
            print(f"  Performance check failed: {e}")
            return status
            
    def _generate_recommendations(self, results: Dict[str, any]) -> List[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = []
        
        # Docker recommendations
        if not results["docker_status"]["docker_daemon"]:
            recommendations.append("🐳 Start Docker daemon")
        if not results["docker_status"]["postgres_container"]:
            recommendations.append("🐳 Start PostgreSQL container: docker-compose up -d")
            
        # Connection recommendations
        if not results["connection_tests"]["asyncpg"]["success"]:
            recommendations.append("🔌 Check asyncpg connection - verify credentials and network")
        if not results["connection_tests"]["sqlalchemy"]["success"]:
            recommendations.append("🔌 Check SQLAlchemy connection - may need connection string fix")
            
        # Credential recommendations
        if results["credential_validation"]["mismatches"]:
            recommendations.append("🔑 Update database configuration to match Docker setup")
            
        # Database cleanup recommendations
        if results["database_inventory"]["test_db_count"] > 10:
            recommendations.append("🧹 Clean up old test databases to improve performance")
            
        # Performance recommendations
        if results["performance_check"]["connection_time_ms"] and results["performance_check"]["connection_time_ms"] > 1000:
            recommendations.append("⚡ Connection time is slow - check network and container resources")
            
        return recommendations
        
    def _print_summary(self, results: Dict[str, any]) -> None:
        """Print diagnostic summary."""
        print("\n" + "=" * 60)
        print("📋 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        # Overall status
        critical_issues = 0
        if not results["docker_status"]["docker_daemon"]:
            critical_issues += 1
        if not results["docker_status"]["postgres_container"]:
            critical_issues += 1
        if not results["connection_tests"]["docker_exec"]["success"]:
            critical_issues += 1
            
        if critical_issues == 0:
            print("🎉 Overall Status: HEALTHY")
        elif critical_issues < 2:
            print("⚠️  Overall Status: DEGRADED")
        else:
            print("❌ Overall Status: UNHEALTHY")
            
        # Recommendations
        if results["recommendations"]:
            print("\n📝 RECOMMENDATIONS:")
            for rec in results["recommendations"]:
                print(f"  {rec}")
        else:
            print("\n✅ No recommendations - system is healthy!")
            
        print("\n" + "=" * 60)


async def main():
    """Run diagnostic tool."""
    diagnostics = DatabaseDiagnostics()
    results = await diagnostics.run_comprehensive_check()
    
    # Save results to file
    import json
    with open("database_diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: database_diagnostic_results.json")
    
    return 0 if results["docker_status"]["docker_daemon"] else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))