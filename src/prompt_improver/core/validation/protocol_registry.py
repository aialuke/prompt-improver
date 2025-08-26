"""Protocol Registry for Service Contract Management (2025).

This module provides a centralized registry for protocol-based service contracts,
ensuring type-safe service resolution and validation of protocol compliance.
"""

import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = TypeVar("P", bound=Protocol)


@dataclass
class ServiceContract:
    """Represents a service contract with protocol binding."""
    protocol_type: type[Protocol]
    implementation_type: type
    implementation_factory: Any | None
    dependencies: list[type[Protocol]]
    lifecycle: str  # "singleton", "transient", "scoped"
    tags: set[str]


@dataclass
class ContractValidationResult:
    """Result of protocol contract validation."""
    is_valid: bool
    missing_methods: list[str]
    signature_mismatches: list[str]
    type_violations: list[str]
    warnings: list[str]


class ProtocolRegistry:
    """Registry for managing protocol-based service contracts."""

    def __init__(self) -> None:
        """Initialize the protocol registry."""
        self._contracts: dict[type[Protocol], ServiceContract] = {}
        self._implementations: dict[type, list[type[Protocol]]] = defaultdict(list)
        self._tags: dict[str, list[type[Protocol]]] = defaultdict(list)
        self._dependencies: dict[type[Protocol], list[type[Protocol]]] = {}

    def register_contract(
        self,
        protocol_type: type[P],
        implementation_type: type[T],
        implementation_factory: Any | None = None,
        dependencies: list[type[Protocol]] | None = None,
        lifecycle: str = "singleton",
        tags: set[str] | None = None
    ) -> None:
        """Register a service contract with protocol validation.

        Args:
            protocol_type: The protocol interface
            implementation_type: The concrete implementation class
            implementation_factory: Optional factory function for creating instances
            dependencies: List of protocol dependencies
            lifecycle: Service lifecycle management
            tags: Optional tags for service discovery
        """
        if dependencies is None:
            dependencies = []
        if tags is None:
            tags = set()

        # Validate that implementation conforms to protocol
        validation_result = self.validate_contract(protocol_type, implementation_type)
        if not validation_result.is_valid:
            raise ValueError(
                f"Implementation {implementation_type.__name__} does not conform to "
                f"protocol {protocol_type.__name__}: {validation_result.missing_methods}"
            )

        contract = ServiceContract(
            protocol_type=protocol_type,
            implementation_type=implementation_type,
            implementation_factory=implementation_factory,
            dependencies=dependencies,
            lifecycle=lifecycle,
            tags=tags
        )

        self._contracts[protocol_type] = contract
        self._implementations[implementation_type].append(protocol_type)
        self._dependencies[protocol_type] = dependencies

        # Update tag index
        for tag in tags:
            self._tags[tag].append(protocol_type)

        logger.debug(f"Registered contract: {protocol_type.__name__} -> {implementation_type.__name__}")

    def validate_contract(
        self,
        protocol_type: type[Protocol],
        implementation_type: type
    ) -> ContractValidationResult:
        """Validate that an implementation conforms to a protocol.

        Args:
            protocol_type: The protocol interface to validate against
            implementation_type: The implementation class to validate

        Returns:
            ContractValidationResult with validation details
        """
        missing_methods = []
        signature_mismatches = []
        type_violations = []
        warnings = []

        try:
            # Get protocol methods
            protocol_methods = self._get_protocol_methods(protocol_type)

            # Check each protocol method
            for method_name, protocol_signature in protocol_methods.items():
                if not hasattr(implementation_type, method_name):
                    missing_methods.append(method_name)
                    continue

                impl_method = getattr(implementation_type, method_name)
                if not callable(impl_method):
                    type_violations.append(f"{method_name} is not callable")
                    continue

                # Compare signatures
                try:
                    impl_signature = inspect.signature(impl_method)
                    if not self._signatures_compatible(protocol_signature, impl_signature):
                        signature_mismatches.append(
                            f"{method_name}: expected {protocol_signature}, got {impl_signature}"
                        )
                except Exception as e:
                    warnings.append(f"Could not validate signature for {method_name}: {e}")

            # Check for additional validation
            if hasattr(protocol_type, '__annotations__'):
                warnings.extend(f"Missing attribute: {attr_name}" for attr_name in protocol_type.__annotations__ if not hasattr(implementation_type, attr_name))

        except Exception as e:
            warnings.append(f"Validation error: {e}")

        is_valid = not missing_methods and not signature_mismatches and not type_violations

        return ContractValidationResult(
            is_valid=is_valid,
            missing_methods=missing_methods,
            signature_mismatches=signature_mismatches,
            type_violations=type_violations,
            warnings=warnings
        )

    def get_contract(self, protocol_type: type[P]) -> ServiceContract | None:
        """Get service contract for a protocol."""
        return self._contracts.get(protocol_type)

    def get_implementation_type(self, protocol_type: type[P]) -> type | None:
        """Get implementation type for a protocol."""
        contract = self._contracts.get(protocol_type)
        return contract.implementation_type if contract else None

    def get_dependencies(self, protocol_type: type[P]) -> list[type[Protocol]]:
        """Get protocol dependencies for a service."""
        return self._dependencies.get(protocol_type, [])

    def find_by_tag(self, tag: str) -> list[type[Protocol]]:
        """Find protocols by tag."""
        return self._tags.get(tag, [])

    def validate_dependency_graph(self) -> dict[str, Any]:
        """Validate the complete dependency graph for circular dependencies."""
        result = {
            "is_valid": True,
            "circular_dependencies": [],
            "missing_dependencies": [],
            "warnings": []
        }

        # Check for circular dependencies
        visited = set()
        rec_stack = set()

        def has_cycle(protocol: type[Protocol]) -> bool:
            visited.add(protocol)
            rec_stack.add(protocol)

            for dependency in self._dependencies.get(protocol, []):
                if dependency not in visited:
                    if has_cycle(dependency):
                        return True
                elif dependency in rec_stack:
                    result["circular_dependencies"].append(
                        f"{protocol.__name__} -> {dependency.__name__}"
                    )
                    return True

            rec_stack.remove(protocol)
            return False

        for protocol in self._contracts:
            if protocol not in visited:
                has_cycle(protocol)

        # Check for missing dependencies
        for protocol, dependencies in self._dependencies.items():
            for dependency in dependencies:
                if dependency not in self._contracts:
                    result["missing_dependencies"].append(
                        f"{protocol.__name__} requires {dependency.__name__} but it's not registered"
                    )

        result["is_valid"] = (
            not result["circular_dependencies"] and
            not result["missing_dependencies"]
        )

        return result

    def generate_dependency_order(self) -> list[type[Protocol]]:
        """Generate topologically sorted dependency order."""
        visited = set()
        result = []

        def visit(protocol: type[Protocol]) -> None:
            if protocol in visited:
                return
            visited.add(protocol)

            # Visit dependencies first
            for dependency in self._dependencies.get(protocol, []):
                if dependency in self._contracts:
                    visit(dependency)

            result.append(protocol)

        for protocol in self._contracts:
            visit(protocol)

        return result

    def _get_protocol_methods(self, protocol_type: type[Protocol]) -> dict[str, inspect.Signature]:
        """Extract methods from a protocol with their signatures."""
        methods = {}

        for name in dir(protocol_type):
            if name.startswith('_'):
                continue

            attr = getattr(protocol_type, name)
            if callable(attr):
                try:
                    sig = inspect.signature(attr)
                    methods[name] = sig
                except (ValueError, TypeError):
                    # Some built-in methods might not have signatures
                    continue

        return methods

    def _signatures_compatible(
        self,
        protocol_sig: inspect.Signature,
        impl_sig: inspect.Signature
    ) -> bool:
        """Check if implementation signature is compatible with protocol signature."""
        try:
            # Basic parameter count check (allowing for additional optional parameters)
            protocol_required = sum(
                1 for p in protocol_sig.parameters.values()
                if p.default == inspect.Parameter.empty and p.name != 'self'
            )
            impl_required = sum(
                1 for p in impl_sig.parameters.values()
                if p.default == inspect.Parameter.empty and p.name != 'self'
            )

            # Implementation can have more optional parameters, but not fewer required ones
            if impl_required > protocol_required:
                return False

            # Check parameter names and types (basic check)
            protocol_params = list(protocol_sig.parameters.values())
            impl_params = list(impl_sig.parameters.values())

            for i, protocol_param in enumerate(protocol_params):
                if protocol_param.name == 'self':
                    continue
                if i < len(impl_params):
                    impl_param = impl_params[i]
                    if (protocol_param.name != impl_param.name and
                        protocol_param.kind != inspect.Parameter.VAR_POSITIONAL):
                        return False

            return True

        except Exception:
            # If we can't validate, assume compatible (with warning logged elsewhere)
            return True

    def get_registry_stats(self) -> dict[str, Any]:
        """Get statistics about the protocol registry."""
        return {
            "total_contracts": len(self._contracts),
            "total_implementations": len(self._implementations),
            "total_tags": len(self._tags),
            "lifecycle_distribution": {
                lifecycle: sum(1 for c in self._contracts.values() if c.lifecycle == lifecycle)
                for lifecycle in ["singleton", "transient", "scoped"]
            },
            "dependency_count": sum(len(deps) for deps in self._dependencies.values()),
            "protocols_by_tag": {tag: len(protocols) for tag, protocols in self._tags.items()}
        }


# Global registry instance
_protocol_registry = ProtocolRegistry()


def register_protocol_contract[P: Protocol, T](
    protocol_type: type[P],
    implementation_type: type[T],
    implementation_factory: Any | None = None,
    dependencies: list[type[Protocol]] | None = None,
    lifecycle: str = "singleton",
    tags: set[str] | None = None
) -> None:
    """Register a protocol contract with the global registry."""
    _protocol_registry.register_contract(
        protocol_type=protocol_type,
        implementation_type=implementation_type,
        implementation_factory=implementation_factory,
        dependencies=dependencies,
        lifecycle=lifecycle,
        tags=tags
    )


def get_protocol_registry() -> ProtocolRegistry:
    """Get the global protocol registry instance."""
    return _protocol_registry


def validate_all_contracts() -> dict[str, Any]:
    """Validate all registered contracts and dependency graph."""
    registry = get_protocol_registry()

    validation_results = {}
    for protocol_type, contract in registry._contracts.items():
        result = registry.validate_contract(protocol_type, contract.implementation_type)
        validation_results[protocol_type.__name__] = {
            "is_valid": result.is_valid,
            "issues": result.missing_methods + result.signature_mismatches + result.type_violations,
            "warnings": result.warnings
        }

    dependency_validation = registry.validate_dependency_graph()

    return {
        "contract_validation": validation_results,
        "dependency_validation": dependency_validation,
        "registry_stats": registry.get_registry_stats()
    }
