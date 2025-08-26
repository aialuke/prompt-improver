"""
Test quality validator for ML testing.

This module contains test quality validation utilities extracted from conftest.py
to maintain clean architecture and separation of concerns.
"""
import inspect
from typing import Any, Protocol


class TestQualityValidator:
    """Test quality validator for ensuring fixture reliability.

    Validates fixture behavior, Protocol compliance, and integration patterns
    to maintain high-quality test infrastructure.
    """
    
    def __init__(self):
        self.validation_results = []

    def validate_protocol_compliance(
        self, implementation: Any, protocol_class: Protocol
    ) -> dict[str, Any]:
        """Validate that implementation follows Protocol interface."""
        validation_result = {
            "implementation": implementation.__class__.__name__,
            "protocol": protocol_class.__name__,
            "compliant": True,
            "missing_methods": [],
            "signature_mismatches": [],
        }
        protocol_methods = [
            name
            for name, method in inspect.getmembers(
                protocol_class, inspect.isfunction
            )
        ]
        for method_name in protocol_methods:
            if not hasattr(implementation, method_name):
                validation_result["compliant"] = False
                validation_result["missing_methods"].append(method_name)
            else:
                protocol_method = getattr(protocol_class, method_name)
                impl_method = getattr(implementation, method_name)
                protocol_sig = inspect.signature(protocol_method)
                impl_sig = inspect.signature(impl_method)
                if protocol_sig != impl_sig:
                    validation_result["signature_mismatches"].append({
                        "method": method_name,
                        "protocol_signature": str(protocol_sig),
                        "implementation_signature": str(impl_sig),
                    })
        self.validation_results.append(validation_result)
        return validation_result

    def validate_fixture_isolation(
        self, fixture_instances: list[Any]
    ) -> dict[str, Any]:
        """Validate that fixtures are properly isolated."""
        isolation_result = {
            "fixtures_tested": len(fixture_instances),
            "isolation_violations": [],
            "shared_state_detected": False,
        }
        for i, fixture1 in enumerate(fixture_instances):
            for j, fixture2 in enumerate(fixture_instances[i + 1 :], i + 1):
                if hasattr(fixture1, "__dict__") and hasattr(fixture2, "__dict__"):
                    fixture1_objects = {
                        id(obj)
                        for obj in fixture1.__dict__.values()
                        if hasattr(obj, "__dict__")
                    }
                    fixture2_objects = {
                        id(obj)
                        for obj in fixture2.__dict__.values()
                        if hasattr(obj, "__dict__")
                    }
                    shared_objects = fixture1_objects.intersection(fixture2_objects)
                    if shared_objects:
                        isolation_result["shared_state_detected"] = True
                        isolation_result["isolation_violations"].append({
                            "fixture1_index": i,
                            "fixture2_index": j,
                            "shared_object_ids": list(shared_objects),
                        })
        return isolation_result

    def get_validation_summary(self) -> dict[str, Any]:
        """Get comprehensive validation summary."""
        if not self.validation_results:
            return {"status": "no_validations_run"}
        compliant_results = [r for r in self.validation_results if r["compliant"]]
        return {
            "total_validations": len(self.validation_results),
            "compliant_implementations": len(compliant_results),
            "compliance_rate": len(compliant_results)
            / len(self.validation_results)
            * 100,
            "validation_details": self.validation_results,
        }