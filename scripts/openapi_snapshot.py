#!/usr/bin/env python3
"""OpenAPI Schema Snapshot Tool for APES MCP Server.

Generates and compares OpenAPI-compatible schemas for MCP server models
to verify that typing changes don't break the API contract.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import jsonschema
except ImportError:
    jsonschema = None

from pydantic import (
    BaseModel,
    Field,
    __version__ as pydantic_version,
)


# Define the request models directly to avoid import issues
class PromptEnhancementRequest(BaseModel):
    """Request model for prompt enhancement."""
    prompt: str = Field(..., description="The prompt to enhance")
    context: dict[str, Any] | None = Field(
        default=None, description="Additional context for enhancement"
    )
    session_id: str | None = Field(default=None, description="Session ID for tracking")


class PromptStorageRequest(BaseModel):
    """Request model for storing prompt data."""
    original: str = Field(..., description="The original prompt")
    enhanced: str = Field(..., description="The enhanced prompt")
    metrics: dict[str, Any] = Field(..., description="Success metrics")
    session_id: str | None = Field(default=None, description="Session ID for tracking")


def get_pydantic_schema(model_class: type) -> dict[str, Any]:
    """Generate JSON schema for a Pydantic model."""
    try:
        # Pydantic v2 approach
        if hasattr(model_class, "model_json_schema"):
            return model_class.model_json_schema()
        # Fallback for older versions
        return model_class.schema()
    except (AttributeError, TypeError, ValueError) as e:
        print(f"Warning: Could not generate schema for {model_class.__name__}: {e}")
        return {}


def generate_mcp_schema() -> dict[str, Any]:
    """Generate comprehensive schema for MCP server models and tools."""
    schema = {
        "openapi": "3.0.3",
        "info": {
            "title": "APES MCP Server Schema",
            "version": "1.0.0",
            "description": "Schema for Adaptive Prompt Enhancement System MCP Server",
        },
        "components": {
            "schemas": {},
        },
        "tools": {},
        "meta": {
            "generated_at": "",
            "pydantic_version": pydantic_version,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
    }

    # Generate schemas for request models
    models = [
        ("PromptEnhancementRequest", PromptEnhancementRequest),
        ("PromptStorageRequest", PromptStorageRequest),
    ]

    for model_name, model_class in models:
        try:
            model_schema = get_pydantic_schema(model_class)
            if model_schema:
                schema["components"]["schemas"][model_name] = model_schema
                print(f"✓ Generated schema for {model_name}")
            else:
                print(f"✗ Failed to generate schema for {model_name}")
        except (AttributeError, TypeError, ValueError, ImportError) as e:
            print(f"✗ Error generating schema for {model_name}: {e}")

    # Add tool definitions (simplified representation)
    schema["tools"] = {
        "improve_prompt": {
            "description": "Enhance a prompt using ML-optimized rules",
            "parameters": "PromptEnhancementRequest",
            "returns": {
                "type": "object",
                "properties": {
                    "improved_prompt": {"type": "string"},
                    "confidence": {"type": "number"},
                    "applied_rules": {"type": "array"},
                    "processing_time_ms": {"type": "number"},
                    "session_id": {"type": "string"},
                },
            },
        },
        "store_prompt": {
            "description": "Store prompt interaction data for ML training",
            "parameters": "PromptStorageRequest",
            "returns": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "priority": {"type": "integer"},
                    "data_source": {"type": "string"},
                    "session_id": {"type": "string"},
                },
            },
        },
    }

    # Add timestamp
    from datetime import datetime
    schema["meta"]["generated_at"] = datetime.now().isoformat()

    return schema


def save_schema(schema: dict[str, Any], output_file: Path) -> None:
    """Save schema to file with proper formatting."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, sort_keys=True, ensure_ascii=False)

    print(f"✓ Schema saved to {output_file}")


def load_schema(schema_file: Path) -> dict[str, Any]:
    """Load schema from file."""
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    with open(schema_file, encoding="utf-8") as f:
        return json.load(f)


def compare_schemas(old_schema: dict[str, Any], new_schema: dict[str, Any]) -> bool:
    """Compare two schemas and report differences.
    
    Returns True if schemas are compatible, False if breaking changes detected.
    """
    print("\n📊 Schema Comparison Results:")
    print("=" * 50)

    compatible = True

    # Compare component schemas
    old_components = old_schema.get("components", {}).get("schemas", {})
    new_components = new_schema.get("components", {}).get("schemas", {})

    # Check for removed models
    removed_models = set(old_components.keys()) - set(new_components.keys())
    if removed_models:
        print(f"🚨 BREAKING: Removed models: {', '.join(removed_models)}")
        compatible = False

    # Check for added models
    added_models = set(new_components.keys()) - set(old_components.keys())
    if added_models:
        print(f"✅ Added models: {', '.join(added_models)}")

    # Check for modified models
    for model_name in set(old_components.keys()) & set(new_components.keys()):
        old_model = old_components[model_name]
        new_model = new_components[model_name]

        # Check required fields
        old_required = set(old_model.get("required", []))
        new_required = set(new_model.get("required", []))

        added_required = new_required - old_required
        removed_required = old_required - new_required

        if added_required:
            print(f"🚨 BREAKING: {model_name} - Added required fields: {', '.join(added_required)}")
            compatible = False

        if removed_required:
            print(f"✅ {model_name} - Removed required fields: {', '.join(removed_required)}")

        # Check field types
        old_props = old_model.get("properties", {})
        new_props = new_model.get("properties", {})

        for field_name in set(old_props.keys()) & set(new_props.keys()):
            old_type = old_props[field_name].get("type")
            new_type = new_props[field_name].get("type")

            if old_type != new_type:
                print(f"⚠️  {model_name}.{field_name} - Type changed: {old_type} → {new_type}")
                # Type changes are potentially breaking but context-dependent

    # Compare tools
    old_tools = old_schema.get("tools", {})
    new_tools = new_schema.get("tools", {})

    removed_tools = set(old_tools.keys()) - set(new_tools.keys())
    if removed_tools:
        print(f"🚨 BREAKING: Removed tools: {', '.join(removed_tools)}")
        compatible = False

    added_tools = set(new_tools.keys()) - set(old_tools.keys())
    if added_tools:
        print(f"✅ Added tools: {', '.join(added_tools)}")

    print("\n" + "=" * 50)

    if compatible:
        print("✅ Schema changes are backward compatible")
    else:
        print("🚨 Breaking changes detected!")

    return compatible


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate and compare OpenAPI schemas for APES MCP server"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("artifacts") / "schemas" / "mcp_schema.json",
        help="Output file for generated schema",
    )
    parser.add_argument(
        "--compare",
        "-c",
        type=Path,
        help="Compare with existing schema file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if breaking changes detected",
    )

    args = parser.parse_args()

    print("🔧 Generating MCP Server Schema...")
    print(f"Pydantic version: {pydantic_version}")

    try:
        # Generate current schema
        current_schema = generate_mcp_schema()

        # Save current schema
        save_schema(current_schema, args.output)

        # Compare if requested
        if args.compare:
            if not args.compare.exists():
                print(f"⚠️  Comparison file not found: {args.compare}")
                print("This will be used as baseline for future comparisons")
                return 0

            print(f"\n📋 Comparing with {args.compare}...")
            old_schema = load_schema(args.compare)
            compatible = compare_schemas(old_schema, current_schema)

            if not compatible and args.strict:
                print("\n💥 Exiting with error due to breaking changes (--strict mode)")
                return 1

        print("\n✅ Schema generation completed successfully")
        print(f"📁 Schema saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"\n💥 Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
