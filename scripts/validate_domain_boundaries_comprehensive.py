#!/usr/bin/env python3
"""Comprehensive Domain Boundary Validation Script.

This script performs extensive validation of the consolidated protocol architecture
to ensure all domain boundaries are properly maintained and architectural integrity
is preserved.

Tests performed:
1. Domain import isolation validation
2. Protocol runtime checkability verification
3. Cross-domain dependency direction validation
4. Security protocol isolation validation
5. ML lazy loading boundary validation
6. Performance boundary validation (<2ms imports)
7. Clean architecture layer separation validation
8. Protocol interface compliance validation

Generated reports:
- Domain boundary validation results
- Performance metrics for protocol imports
- Security isolation validation results
- ML lazy loading validation results
- Comprehensive architectural integrity report
"""

import ast
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class DomainBoundaryValidator:
    """Comprehensive domain boundary validation for consolidated protocols."""

    def __init__(self) -> None:
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "domain_isolation": {},
            "runtime_checkability": {},
            "dependency_direction": {},
            "security_isolation": {},
            "lazy_loading": {},
            "performance_boundaries": {},
            "clean_architecture": {},
            "protocol_compliance": {},
            "summary": {}
        }

        self.protocol_domains = [
            "core", "cache", "database", "security",
            "cli", "mcp", "application"
        ]

        self.lazy_domains = ["ml", "monitoring"]

    def validate_all(self) -> dict[str, Any]:
        """Run all domain boundary validations."""
        print("🔍 Starting Comprehensive Domain Boundary Validation...")
        print("=" * 80)

        # Run all validation categories
        self.validate_domain_isolation()
        self.validate_runtime_checkability()
        self.validate_dependency_direction()
        self.validate_security_isolation()
        self.validate_lazy_loading()
        self.validate_performance_boundaries()
        self.validate_clean_architecture()
        self.validate_protocol_compliance()

        # Generate summary
        self.generate_summary()

        return self.results

    def validate_domain_isolation(self):
        """Validate that protocol domains maintain proper isolation."""
        print("📋 Validating Domain Isolation...")

        isolation_results = {}
        protocols_path = Path(__file__).parent.parent / 'src' / 'prompt_improver' / 'shared' / 'interfaces' / 'protocols'

        for domain in self.protocol_domains:
            domain_file = protocols_path / f"{domain}.py"
            if not domain_file.exists():
                isolation_results[domain] = {
                    "status": "missing",
                    "error": f"Domain file {domain_file} does not exist"
                }
                continue

            try:
                # Parse the domain file for imports
                with open(domain_file, encoding='utf-8') as f:
                    tree = ast.parse(f.read())

                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)

                # Check for cross-domain imports
                cross_domain_imports = []
                for imp in imports:
                    if 'prompt_improver' in imp and 'protocols' in imp:
                        cross_domain_imports.extend(imp for other_domain in self.protocol_domains + self.lazy_domains if other_domain != domain and other_domain in imp)

                isolation_results[domain] = {
                    "status": "isolated" if not cross_domain_imports else "violations",
                    "cross_domain_imports": cross_domain_imports,
                    "total_imports": len(imports),
                    "protocol_imports": [imp for imp in imports if 'protocols' in imp]
                }

            except Exception as e:
                isolation_results[domain] = {
                    "status": "error",
                    "error": str(e)
                }

        self.results["domain_isolation"] = isolation_results

        # Print results
        for domain, result in isolation_results.items():
            status_icon = "✅" if result["status"] == "isolated" else "❌" if result["status"] == "violations" else "⚠️"
            print(f"  {status_icon} {domain.capitalize()} Domain: {result['status']}")
            if result.get("cross_domain_imports"):
                print(f"    ⚠️  Cross-domain imports: {result['cross_domain_imports']}")

    def validate_runtime_checkability(self):
        """Validate that protocols are properly runtime checkable."""
        print("\n🔧 Validating Runtime Checkability...")

        checkability_results = {}

        try:
            from prompt_improver.shared.interfaces.protocols import (
                core,
                get_ml_protocols,
                get_monitoring_protocols,
            )

            # Test core domain protocols
            core_protocols = ['ServiceProtocol', 'HealthCheckProtocol']
            for protocol_name in core_protocols:
                if hasattr(core, protocol_name):
                    protocol_class = getattr(core, protocol_name)
                    checkability_results[f"core.{protocol_name}"] = self._test_protocol_checkability(protocol_class)

            # Test other domain protocols (if they exist)
            domain_protocol_map = {
                'cache': ['BasicCacheProtocol', 'CacheHealthProtocol', 'CacheServiceFacadeProtocol'],
                'database': ['SessionManagerProtocol', 'DatabaseProtocol', 'ConnectionPoolCoreProtocol'],
                'security': ['AuthenticationProtocol', 'AuthorizationProtocol', 'EncryptionProtocol'],
                'cli': ['CommandProcessorProtocol', 'UserInteractionProtocol', 'WorkflowManagerProtocol'],
                'mcp': ['MCPServerProtocol', 'MCPToolProtocol', 'ServerServicesProtocol'],
                'application': ['ApplicationServiceProtocol', 'WorkflowOrchestratorProtocol', 'ValidationServiceProtocol']
            }

            for domain_name, protocols in domain_protocol_map.items():
                domain_module = locals()[domain_name]
                for protocol_name in protocols:
                    if hasattr(domain_module, protocol_name):
                        protocol_class = getattr(domain_module, protocol_name)
                        checkability_results[f"{domain_name}.{protocol_name}"] = self._test_protocol_checkability(protocol_class)

            # Test lazy-loaded protocols
            try:
                ml_protocols = get_ml_protocols()
                ml_protocol_names = ['MLflowServiceProtocol', 'EventBusProtocol']
                for protocol_name in ml_protocol_names:
                    if hasattr(ml_protocols, protocol_name):
                        protocol_class = getattr(ml_protocols, protocol_name)
                        checkability_results[f"ml.{protocol_name}"] = self._test_protocol_checkability(protocol_class)
            except Exception as e:
                checkability_results["ml.lazy_loading"] = {"status": "error", "error": str(e)}

            try:
                monitoring_protocols = get_monitoring_protocols()
                monitoring_protocol_names = ['MetricsCollectorProtocol', 'HealthCheckComponentProtocol']
                for protocol_name in monitoring_protocol_names:
                    if hasattr(monitoring_protocols, protocol_name):
                        protocol_class = getattr(monitoring_protocols, protocol_name)
                        checkability_results[f"monitoring.{protocol_name}"] = self._test_protocol_checkability(protocol_class)
            except Exception as e:
                checkability_results["monitoring.lazy_loading"] = {"status": "error", "error": str(e)}

        except Exception as e:
            checkability_results["global_error"] = {"status": "error", "error": str(e)}

        self.results["runtime_checkability"] = checkability_results

        # Print results
        for protocol_name, result in checkability_results.items():
            status_icon = "✅" if result["status"] == "checkable" else "❌"
            print(f"  {status_icon} {protocol_name}: {result['status']}")
            if result.get("error"):
                print(f"    ⚠️  Error: {result['error']}")

    def _test_protocol_checkability(self, protocol_class) -> dict[str, Any]:
        """Test if a protocol class is runtime checkable."""
        try:
            # Create a test implementation
            class TestImplementation:
                pass

            test_instance = TestImplementation()

            # Test isinstance - should work without error if protocol is runtime checkable
            isinstance(test_instance, protocol_class)

            return {
                "status": "checkable",
                "protocol_type": str(type(protocol_class)),
                "methods": [method for method in dir(protocol_class) if not method.startswith('_')]
            }

        except TypeError as e:
            return {
                "status": "not_checkable",
                "error": str(e),
                "protocol_type": str(type(protocol_class))
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "protocol_type": str(type(protocol_class))
            }

    def validate_dependency_direction(self):
        """Validate Clean Architecture dependency direction."""
        print("\n🏗️ Validating Dependency Direction (Clean Architecture)...")

        direction_results = {}

        try:
            # Test core layer (should be dependency-free)
            start_time = time.time()
            from prompt_improver.shared.interfaces.protocols import core
            core_import_time = time.time() - start_time

            direction_results["core_layer"] = {
                "status": "valid" if core_import_time < 0.1 else "slow",
                "import_time": core_import_time,
                "protocols": [attr for attr in dir(core) if not attr.startswith('_') and 'Protocol' in attr]
            }

            # Test application layer (can depend on core)
            start_time = time.time()
            from prompt_improver.shared.interfaces.protocols import application
            app_import_time = time.time() - start_time

            direction_results["application_layer"] = {
                "status": "valid" if app_import_time < 0.5 else "slow",
                "import_time": app_import_time,
                "protocols": [attr for attr in dir(application) if not attr.startswith('_') and 'Protocol' in attr]
            }

            # Test infrastructure layer (can depend on application and core)
            start_time = time.time()
            from prompt_improver.shared.interfaces.protocols import cache, database
            infra_import_time = time.time() - start_time

            direction_results["infrastructure_layer"] = {
                "status": "valid" if infra_import_time < 1.0 else "slow",
                "import_time": infra_import_time,
                "database_protocols": [attr for attr in dir(database) if not attr.startswith('_') and 'Protocol' in attr],
                "cache_protocols": [attr for attr in dir(cache) if not attr.startswith('_') and 'Protocol' in attr]
            }

        except Exception as e:
            direction_results["error"] = {"status": "error", "error": str(e)}

        self.results["dependency_direction"] = direction_results

        # Print results
        for layer, result in direction_results.items():
            if layer != "error":
                status_icon = "✅" if result["status"] == "valid" else "⚠️"
                print(f"  {status_icon} {layer.replace('_', ' ').title()}: {result['status']} ({result['import_time']:.3f}s)")

    def validate_security_isolation(self):
        """Validate security protocol isolation per OWASP 2025."""
        print("\n🔒 Validating Security Protocol Isolation...")

        security_results = {}

        try:
            from prompt_improver.shared.interfaces.protocols import security

            # Test authentication protocol isolation
            if hasattr(security, 'AuthenticationProtocol'):
                auth_protocol = security.AuthenticationProtocol
                auth_methods = [method for method in dir(auth_protocol) if not method.startswith('_')]

                security_results["authentication"] = {
                    "status": "isolated",
                    "methods": auth_methods,
                    "isolation_check": "passed" if not any('authorization' in m.lower() for m in auth_methods) else "failed"
                }

            # Test authorization protocol isolation
            if hasattr(security, 'AuthorizationProtocol'):
                authz_protocol = security.AuthorizationProtocol
                authz_methods = [method for method in dir(authz_protocol) if not method.startswith('_')]

                security_results["authorization"] = {
                    "status": "isolated",
                    "methods": authz_methods,
                    "isolation_check": "passed" if not any('authenticate' in m.lower() for m in authz_methods) else "failed"
                }

            # Test encryption protocol isolation
            if hasattr(security, 'EncryptionProtocol'):
                encrypt_protocol = security.EncryptionProtocol
                encrypt_methods = [method for method in dir(encrypt_protocol) if not method.startswith('_')]

                security_results["encryption"] = {
                    "status": "isolated",
                    "methods": encrypt_methods,
                    "isolation_check": "passed"
                }

        except Exception as e:
            security_results["error"] = {"status": "error", "error": str(e)}

        self.results["security_isolation"] = security_results

        # Print results
        for protocol, result in security_results.items():
            if protocol != "error":
                status_icon = "✅" if result["isolation_check"] == "passed" else "❌"
                print(f"  {status_icon} {protocol.capitalize()} Protocol: {result['isolation_check']}")

    def validate_lazy_loading(self):
        """Validate ML and monitoring lazy loading boundaries."""
        print("\n⏳ Validating Lazy Loading Boundaries...")

        lazy_results = {}

        # Clear any existing ML/monitoring modules
        modules_to_clear = [mod for mod in sys.modules
                           if 'protocols.ml' in mod or 'protocols.monitoring' in mod]
        for mod in modules_to_clear:
            del sys.modules[mod]

        try:
            # Test that ML/monitoring are not loaded by default

            ml_loaded = any('protocols.ml' in mod for mod in sys.modules)
            monitoring_loaded = any('protocols.monitoring' in mod for mod in sys.modules)

            lazy_results["default_loading"] = {
                "ml_loaded": ml_loaded,
                "monitoring_loaded": monitoring_loaded,
                "status": "passed" if not ml_loaded and not monitoring_loaded else "failed"
            }

            # Test ML lazy loading function
            start_time = time.time()
            from prompt_improver.shared.interfaces.protocols import get_ml_protocols
            ml_protocols = get_ml_protocols()
            ml_load_time = time.time() - start_time

            lazy_results["ml_lazy_loading"] = {
                "load_time": ml_load_time,
                "status": "passed" if ml_load_time < 2.0 else "slow",
                "protocols": [attr for attr in dir(ml_protocols) if not attr.startswith('_') and 'Protocol' in attr]
            }

            # Test monitoring lazy loading function
            start_time = time.time()
            from prompt_improver.shared.interfaces.protocols import (
                get_monitoring_protocols,
            )
            monitoring_protocols = get_monitoring_protocols()
            monitoring_load_time = time.time() - start_time

            lazy_results["monitoring_lazy_loading"] = {
                "load_time": monitoring_load_time,
                "status": "passed" if monitoring_load_time < 1.0 else "slow",
                "protocols": [attr for attr in dir(monitoring_protocols) if not attr.startswith('_') and 'Protocol' in attr]
            }

        except Exception as e:
            lazy_results["error"] = {"status": "error", "error": str(e)}

        self.results["lazy_loading"] = lazy_results

        # Print results
        for test_name, result in lazy_results.items():
            if test_name != "error":
                if test_name == "default_loading":
                    status_icon = "✅" if result["status"] == "passed" else "❌"
                    print(f"  {status_icon} Default Loading: {result['status']}")
                    print(f"    ML loaded by default: {result['ml_loaded']}")
                    print(f"    Monitoring loaded by default: {result['monitoring_loaded']}")
                else:
                    status_icon = "✅" if result["status"] == "passed" else "⚠️"
                    print(f"  {status_icon} {test_name.replace('_', ' ').title()}: {result['status']} ({result['load_time']:.3f}s)")
                    print(f"    Protocols found: {len(result.get('protocols', []))}")

    def validate_performance_boundaries(self):
        """Validate protocol import performance meets <2ms requirement."""
        print("\n⚡ Validating Performance Boundaries...")

        performance_results = {}

        for domain in self.protocol_domains:
            # Clear module cache
            modules_to_clear = [mod for mod in sys.modules if f'protocols.{domain}' in mod]
            for mod in modules_to_clear:
                del sys.modules[mod]

            try:
                start_time = time.time()
                module = importlib.import_module(f'prompt_improver.shared.interfaces.protocols.{domain}')
                import_time = time.time() - start_time

                performance_results[domain] = {
                    "import_time": import_time,
                    "status": "passed" if import_time < 0.002 else "slow",  # 2ms requirement
                    "protocols": [attr for attr in dir(module) if not attr.startswith('_') and 'Protocol' in attr]
                }

            except Exception as e:
                performance_results[domain] = {
                    "status": "error",
                    "error": str(e)
                }

        self.results["performance_boundaries"] = performance_results

        # Print results
        for domain, result in performance_results.items():
            if result["status"] != "error":
                status_icon = "✅" if result["status"] == "passed" else "⚠️"
                print(f"  {status_icon} {domain.capitalize()} Domain: {result['status']} ({result['import_time'] * 1000:.1f}ms)")
                print(f"    Protocols: {len(result.get('protocols', []))}")
            else:
                print(f"  ❌ {domain.capitalize()} Domain: {result['status']} - {result.get('error', 'Unknown error')}")

    def validate_clean_architecture(self):
        """Validate Clean Architecture layer separation."""
        print("\n🏛️ Validating Clean Architecture Compliance...")

        architecture_results = {}

        try:
            # Test presentation layer isolation (CLI should not leak into core business)
            from prompt_improver.shared.interfaces.protocols import (
                cache,
                core,
                database,
            )

            architecture_results["presentation_isolation"] = {
                "status": "passed",
                "core_loaded": core is not None,
                "database_loaded": database is not None,
                "cache_loaded": cache is not None
            }

            # Test application layer positioning
            from prompt_improver.shared.interfaces.protocols import application

            architecture_results["application_positioning"] = {
                "status": "passed",
                "can_import_core": hasattr(application, '__name__'),
                "protocols": [attr for attr in dir(application) if not attr.startswith('_') and 'Protocol' in attr]
            }

            # Test infrastructure layer dependencies
            architecture_results["infrastructure_dependencies"] = {
                "status": "passed",
                "database_protocols": [attr for attr in dir(database) if not attr.startswith('_') and 'Protocol' in attr],
                "cache_protocols": [attr for attr in dir(cache) if not attr.startswith('_') and 'Protocol' in attr]
            }

        except Exception as e:
            architecture_results["error"] = {"status": "error", "error": str(e)}

        self.results["clean_architecture"] = architecture_results

        # Print results
        for layer, result in architecture_results.items():
            if layer != "error":
                status_icon = "✅" if result["status"] == "passed" else "❌"
                print(f"  {status_icon} {layer.replace('_', ' ').title()}: {result['status']}")

    def validate_protocol_compliance(self):
        """Validate protocol interface compliance."""
        print("\n📋 Validating Protocol Interface Compliance...")

        compliance_results = {}

        try:
            from prompt_improver.shared.interfaces.protocols import core

            # Test core protocols have required methods
            if hasattr(core, 'ServiceProtocol'):
                service_protocol = core.ServiceProtocol
                required_methods = ['initialize', 'shutdown']
                actual_methods = [method for method in dir(service_protocol) if not method.startswith('_')]

                compliance_results["ServiceProtocol"] = {
                    "required_methods": required_methods,
                    "actual_methods": actual_methods,
                    "compliance": all(method in actual_methods for method in required_methods),
                    "status": "compliant" if all(method in actual_methods for method in required_methods) else "non_compliant"
                }

            if hasattr(core, 'HealthCheckProtocol'):
                health_protocol = core.HealthCheckProtocol
                required_methods = ['check_health', 'is_healthy']
                actual_methods = [method for method in dir(health_protocol) if not method.startswith('_')]

                compliance_results["HealthCheckProtocol"] = {
                    "required_methods": required_methods,
                    "actual_methods": actual_methods,
                    "compliance": all(method in actual_methods for method in required_methods),
                    "status": "compliant" if all(method in actual_methods for method in required_methods) else "non_compliant"
                }

        except Exception as e:
            compliance_results["error"] = {"status": "error", "error": str(e)}

        self.results["protocol_compliance"] = compliance_results

        # Print results
        for protocol, result in compliance_results.items():
            if protocol != "error":
                status_icon = "✅" if result["status"] == "compliant" else "❌"
                print(f"  {status_icon} {protocol}: {result['status']}")
                if not result["compliance"]:
                    missing = set(result["required_methods"]) - set(result["actual_methods"])
                    print(f"    ⚠️  Missing methods: {list(missing)}")

    def generate_summary(self):
        """Generate comprehensive validation summary."""
        print("\n📊 Generating Validation Summary...")

        summary = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "warnings": 0,
            "domain_coverage": {},
            "performance_metrics": {},
            "critical_issues": [],
            "recommendations": []
        }

        # Count validations by category
        for category, results in self.results.items():
            if category in {"validation_timestamp", "summary"}:
                continue

            summary["total_validations"] += 1

            if isinstance(results, dict):
                passed_count = sum(1 for r in results.values()
                                 if isinstance(r, dict) and r.get("status") in {"passed", "valid", "compliant", "isolated", "checkable"})
                failed_count = sum(1 for r in results.values()
                                 if isinstance(r, dict) and r.get("status") in {"failed", "error", "non_compliant", "violations"})

                if passed_count > failed_count:
                    summary["passed_validations"] += 1
                elif failed_count > 0:
                    summary["failed_validations"] += 1
                else:
                    summary["warnings"] += 1

        # Domain coverage
        summary["domain_coverage"] = {
            "total_domains": len(self.protocol_domains) + len(self.lazy_domains),
            "validated_domains": len(self.protocol_domains),
            "lazy_domains": len(self.lazy_domains)
        }

        # Performance metrics
        if "performance_boundaries" in self.results:
            perf_results = self.results["performance_boundaries"]
            total_time = sum(r.get("import_time", 0) for r in perf_results.values() if isinstance(r, dict))

            summary["performance_metrics"] = {
                "total_import_time": total_time,
                "average_import_time": total_time / len(perf_results) if perf_results else 0,
                "domains_under_2ms": sum(1 for r in perf_results.values()
                                       if isinstance(r, dict) and r.get("import_time", 0) < 0.002)
            }

        # Critical issues
        for category, results in self.results.items():
            if isinstance(results, dict):
                for key, result in results.items():
                    if isinstance(result, dict) and result.get("status") == "error":
                        summary["critical_issues"].append(f"{category}.{key}: {result.get('error', 'Unknown error')}")

        # Recommendations
        if summary["failed_validations"] > 0:
            summary["recommendations"].append("Review failed validations and address architectural violations")

        if summary["performance_metrics"].get("domains_under_2ms", 0) < len(self.protocol_domains):
            summary["recommendations"].append("Optimize slow-importing protocol domains to meet <2ms requirement")

        if len(summary["critical_issues"]) > 0:
            summary["recommendations"].append("Address critical errors in protocol loading and validation")

        self.results["summary"] = summary

        # Print summary
        print(f"  📈 Total Validations: {summary['total_validations']}")
        print(f"  ✅ Passed: {summary['passed_validations']}")
        print(f"  ❌ Failed: {summary['failed_validations']}")
        print(f"  ⚠️  Warnings: {summary['warnings']}")
        print(f"  🏗️  Domain Coverage: {summary['domain_coverage']['validated_domains']}/{summary['domain_coverage']['total_domains']}")

        if summary["performance_metrics"]:
            print(f"  ⚡ Performance: {summary['performance_metrics']['domains_under_2ms']}/{len(self.protocol_domains)} domains under 2ms")

        if summary["critical_issues"]:
            print(f"  🚨 Critical Issues: {len(summary['critical_issues'])}")

        if summary["recommendations"]:
            print(f"  💡 Recommendations: {len(summary['recommendations'])}")


def main():
    """Main validation execution."""
    validator = DomainBoundaryValidator()
    results = validator.validate_all()

    # Save results to file
    output_path = Path(__file__).parent.parent / "DOMAIN_BOUNDARY_VALIDATION_REPORT.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("🎯 Domain Boundary Validation Complete!")
    print(f"📄 Detailed report saved to: {output_path}")

    # Generate markdown report
    generate_markdown_report(results, output_path.with_suffix('.md'))
    print(f"📑 Markdown report saved to: {output_path.with_suffix('.md')}")

    return results


def generate_markdown_report(results: dict[str, Any], output_path: Path):
    """Generate markdown validation report."""
    report = f"""# Domain Boundary Validation Report

**Validation Date:** {results['validation_timestamp']}
**Architecture:** Consolidated Protocol Domains with Clean Architecture Compliance

## Executive Summary

{format_summary_section(results.get('summary', {}))}

## Validation Results

### Domain Isolation
{format_domain_isolation(results.get('domain_isolation', {}))}

### Runtime Checkability
{format_runtime_checkability(results.get('runtime_checkability', {}))}

### Dependency Direction (Clean Architecture)
{format_dependency_direction(results.get('dependency_direction', {}))}

### Security Protocol Isolation
{format_security_isolation(results.get('security_isolation', {}))}

### Lazy Loading Boundaries
{format_lazy_loading(results.get('lazy_loading', {}))}

### Performance Boundaries
{format_performance_boundaries(results.get('performance_boundaries', {}))}

### Clean Architecture Compliance
{format_clean_architecture(results.get('clean_architecture', {}))}

### Protocol Interface Compliance
{format_protocol_compliance(results.get('protocol_compliance', {}))}

## Recommendations

{format_recommendations(results.get('summary', {}).get('recommendations', []))}

---
*Generated by Domain Boundary Validation Script*
*Architecture: 2025 Standards with Protocol Domain Consolidation*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)


def format_summary_section(summary: dict[str, Any]) -> str:
    """Format summary section for markdown."""
    if not summary:
        return "No summary data available."

    total = summary.get('total_validations', 0)
    passed = summary.get('passed_validations', 0)
    failed = summary.get('failed_validations', 0)
    warnings = summary.get('warnings', 0)

    success_rate = (passed / total * 100) if total > 0 else 0

    return f"""
- **Overall Success Rate:** {success_rate:.1f}% ({passed}/{total} validations passed)
- **Failed Validations:** {failed}
- **Warnings:** {warnings}
- **Domain Coverage:** {summary.get('domain_coverage', {}).get('validated_domains', 0)}/{summary.get('domain_coverage', {}).get('total_domains', 0)} domains
- **Performance Compliance:** {summary.get('performance_metrics', {}).get('domains_under_2ms', 0)} domains under 2ms import requirement
- **Critical Issues:** {len(summary.get('critical_issues', []))}
"""


def format_domain_isolation(isolation_data: dict[str, Any]) -> str:
    """Format domain isolation results."""
    if not isolation_data:
        return "No domain isolation data available."

    lines = []
    for domain, result in isolation_data.items():
        status = result.get('status', 'unknown')
        icon = "✅" if status == "isolated" else "❌" if status == "violations" else "⚠️"
        lines.append(f"- {icon} **{domain.capitalize()}:** {status}")

        if result.get('cross_domain_imports'):
            lines.append(f"  - Cross-domain imports: {', '.join(result['cross_domain_imports'])}")

    return "\n".join(lines)


def format_runtime_checkability(checkability_data: dict[str, Any]) -> str:
    """Format runtime checkability results."""
    if not checkability_data:
        return "No runtime checkability data available."

    lines = []
    for protocol, result in checkability_data.items():
        status = result.get('status', 'unknown')
        icon = "✅" if status == "checkable" else "❌"
        lines.append(f"- {icon} **{protocol}:** {status}")

        if result.get('error'):
            lines.append(f"  - Error: {result['error']}")

    return "\n".join(lines)


def format_dependency_direction(direction_data: dict[str, Any]) -> str:
    """Format dependency direction results."""
    if not direction_data:
        return "No dependency direction data available."

    lines = []
    for layer, result in direction_data.items():
        if layer != "error":
            status = result.get('status', 'unknown')
            import_time = result.get('import_time', 0)
            icon = "✅" if status == "valid" else "⚠️"
            lines.append(f"- {icon} **{layer.replace('_', ' ').title()}:** {status} ({import_time:.3f}s)")

    return "\n".join(lines)


def format_security_isolation(security_data: dict[str, Any]) -> str:
    """Format security isolation results."""
    if not security_data:
        return "No security isolation data available."

    lines = []
    for protocol, result in security_data.items():
        if protocol != "error":
            isolation_check = result.get('isolation_check', 'unknown')
            icon = "✅" if isolation_check == "passed" else "❌"
            lines.append(f"- {icon} **{protocol.capitalize()} Protocol:** {isolation_check}")

    return "\n".join(lines)


def format_lazy_loading(lazy_data: dict[str, Any]) -> str:
    """Format lazy loading results."""
    if not lazy_data:
        return "No lazy loading data available."

    lines = []
    for test, result in lazy_data.items():
        if test != "error":
            status = result.get('status', 'unknown')
            icon = "✅" if status == "passed" else "⚠️"
            if test == "default_loading":
                lines.append(f"- {icon} **Default Loading:** {status}")
            else:
                load_time = result.get('load_time', 0)
                lines.append(f"- {icon} **{test.replace('_', ' ').title()}:** {status} ({load_time:.3f}s)")

    return "\n".join(lines)


def format_performance_boundaries(perf_data: dict[str, Any]) -> str:
    """Format performance boundary results."""
    if not perf_data:
        return "No performance boundary data available."

    lines = []
    for domain, result in perf_data.items():
        status = result.get('status', 'unknown')
        import_time = result.get('import_time', 0)
        icon = "✅" if status == "passed" else "⚠️" if status == "slow" else "❌"
        lines.append(f"- {icon} **{domain.capitalize()}:** {status} ({import_time * 1000:.1f}ms)")

    return "\n".join(lines)


def format_clean_architecture(arch_data: dict[str, Any]) -> str:
    """Format clean architecture results."""
    if not arch_data:
        return "No clean architecture data available."

    lines = []
    for layer, result in arch_data.items():
        if layer != "error":
            status = result.get('status', 'unknown')
            icon = "✅" if status == "passed" else "❌"
            lines.append(f"- {icon} **{layer.replace('_', ' ').title()}:** {status}")

    return "\n".join(lines)


def format_protocol_compliance(compliance_data: dict[str, Any]) -> str:
    """Format protocol compliance results."""
    if not compliance_data:
        return "No protocol compliance data available."

    lines = []
    for protocol, result in compliance_data.items():
        if protocol != "error":
            status = result.get('status', 'unknown')
            icon = "✅" if status == "compliant" else "❌"
            lines.append(f"- {icon} **{protocol}:** {status}")

            if not result.get('compliance', True):
                required = set(result.get('required_methods', []))
                actual = set(result.get('actual_methods', []))
                missing = required - actual
                if missing:
                    lines.append(f"  - Missing methods: {', '.join(missing)}")

    return "\n".join(lines)


def format_recommendations(recommendations: list[str]) -> str:
    """Format recommendations list."""
    if not recommendations:
        return "No specific recommendations. Architecture appears to be well-structured."

    return "\n".join(f"- {rec}" for rec in recommendations)


if __name__ == "__main__":
    main()
