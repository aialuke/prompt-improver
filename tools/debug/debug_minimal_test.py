#!/usr/bin/env python3
"""Minimal test to isolate NullType issue"""

import asyncio
import sys
from datetime import datetime

# Add the source directory to the Python path
sys.path.insert(0, '/Users/lukemckenzie/prompt-improver/src')

from prompt_improver.database.models import PromptSession, RuleMetadata, RulePerformance
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlmodel import SQLModel

async def test_database_insertion():
    """Test basic database insertion to identify NullType issue"""
    
    # Create test database engine
    test_db_url = "postgresql+psycopg://postgres:password@localhost:5432/apes_test"
    
    try:
        engine = create_async_engine(test_db_url, echo=True)
        async_session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        
        async with async_session() as session:
            # Test 1: Create PromptSession with explicit created_at
            print("=== Testing PromptSession ===")
            prompt_session = PromptSession(
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
            session.add(prompt_session)
            await session.commit()
            print("✅ PromptSession inserted successfully")
            
            # Test 2: Create RuleMetadata with explicit created_at
            print("=== Testing RuleMetadata ===")
            rule_metadata = RuleMetadata(
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
            session.add(rule_metadata)
            await session.commit()
            print("✅ RuleMetadata inserted successfully")
            
            # Test 3: Create RulePerformance with explicit created_at
            print("=== Testing RulePerformance ===")
            rule_performance = RulePerformance(
                id=1,
                session_id="test_session_1",
                rule_id="test_rule",
                improvement_score=0.8,
                confidence_level=0.9,
                execution_time_ms=150,
                parameters_used={"weight": 1.0},
                created_at=datetime.utcnow(),
            )
            session.add(rule_performance)
            await session.commit()
            print("✅ RulePerformance inserted successfully")
            
        await engine.dispose()
        print("✅ All tests passed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_database_insertion())