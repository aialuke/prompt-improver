#!/usr/bin/env python3
"""Debug script to identify NullType column issue"""

import sys
import asyncio
from datetime import datetime

# Add the source directory to the Python path
sys.path.insert(0, '/Users/lukemckenzie/prompt-improver/src')

from prompt_improver.database.models import PromptSession, RuleMetadata, RulePerformance

def debug_model_fields():
    """Debug function to check model field types"""
    
    print("=== DEBUGGING MODEL FIELDS ===")
    
    # Create sample instances
    session = PromptSession(
        id=1,
        session_id="test_session_1",
        original_prompt="Make this better",
        improved_prompt="Please improve this",
        user_context={"context": "test"},
        quality_score=0.8,
        improvement_score=0.75,
        confidence_level=0.9,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    
    rule = RuleMetadata(
        rule_id="test_rule",
        rule_name="Test Rule",
        category="core", 
        description="Test description",
        enabled=True,
        priority=5,
        default_parameters={"weight": 1.0},
        parameter_constraints={"weight": {"min": 0.0, "max": 1.0}},
        created_at=datetime.utcnow(),
    )
    
    perf = RulePerformance(
        id=1,
        session_id="test_session_1",
        rule_id="test_rule",
        improvement_score=0.8,
        confidence_level=0.9,
        execution_time_ms=150,
        parameters_used={"weight": 1.0},
        created_at=datetime.utcnow(),
    )
    
    print("PromptSession fields:")
    for field_name, field_info in PromptSession.model_fields.items():
        print(f"  {field_name}: {field_info}")
    
    print("\nRuleMetadata fields:")
    for field_name, field_info in RuleMetadata.model_fields.items():
        print(f"  {field_name}: {field_info}")
    
    print("\nRulePerformance fields:")
    for field_name, field_info in RulePerformance.model_fields.items():
        print(f"  {field_name}: {field_info}")

if __name__ == "__main__":
    debug_model_fields()