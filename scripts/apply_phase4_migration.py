#!/usr/bin/env python3
"""
Apply Phase 4 Pre-Computed Intelligence Schema Migration

This script applies the Phase 4 database migration that creates the feature store
tables for pre-computed ML intelligence.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prompt_improver.database.connection import get_session_context
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def apply_migration():
    """Apply the Phase 4 migration."""
    try:
        # Read migration file
        migration_path = project_root / "database" / "migrations" / "003_phase4_precomputed_intelligence_schema.sql"

        if not migration_path.exists():
            logger.error(f"Migration file not found: {migration_path}")
            return False

        with open(migration_path, 'r') as f:
            migration_sql = f.read()

        logger.info("Applying Phase 4 pre-computed intelligence schema migration...")

        # Get database session and execute migration
        async with get_session_context() as session:
            # Split migration into individual statements
            statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]

            for i, statement in enumerate(statements):
                if statement:
                    try:
                        logger.info(f"Executing statement {i+1}/{len(statements)}")
                        await session.execute(text(statement))
                        await session.commit()
                    except Exception as e:
                        logger.warning(f"Statement {i+1} failed (may already exist): {e}")
                        await session.rollback()
                        continue

        logger.info("✅ Phase 4 migration applied successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False


async def verify_migration():
    """Verify that the migration was applied correctly."""
    try:
        # Check if tables exist
        tables_to_check = [
            'rule_intelligence_cache',
            'rule_combination_intelligence',
            'pattern_discovery_cache',
            'ml_rule_predictions'
        ]

        async with get_session_context() as session:
            for table in tables_to_check:
                result = await session.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = '{table}'
                    );
                """))
                exists = result.scalar()

                if exists:
                    logger.info(f"✅ Table '{table}' exists")
                else:
                    logger.error(f"❌ Table '{table}' not found")
                    return False

        logger.info("✅ Migration verification successful!")
        return True

    except Exception as e:
        logger.error(f"❌ Migration verification failed: {e}")
        return False


async def main():
    """Main function."""
    logger.info("🚀 Starting Phase 4 migration application...")

    # Apply migration
    if await apply_migration():
        # Verify migration
        if await verify_migration():
            logger.info("🎉 Phase 4 migration completed successfully!")
            return 0
        else:
            logger.error("Migration verification failed")
            return 1
    else:
        logger.error("Migration application failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
