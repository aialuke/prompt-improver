"""Update training_prompts table with missing columns

Revision ID: update_training_v2
Revises: training_prompt_v1
Create Date: 2025-01-20

"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "update_training_v2"
down_revision: str | Sequence[str] | None = "training_prompt_v1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add missing columns to training_prompts table"""
    # Add missing columns that weren't in the original table
    op.add_column('training_prompts', sa.Column('updated_at', sa.DateTime(), nullable=True))
    op.add_column('training_prompts', sa.Column('deleted_at', sa.DateTime(), nullable=True))
    op.add_column('training_prompts', sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'))
    
    # Update existing rows to have is_active = true
    op.execute("UPDATE training_prompts SET is_active = true WHERE is_active IS NULL")
    
    # Create additional indexes for new columns
    op.create_index('idx_training_active', 'training_prompts', ['is_active'])
    
    # Ensure existing indexes exist (in case they're missing)
    try:
        op.create_index('idx_training_data_source', 'training_prompts', ['data_source'])
    except Exception:
        pass  # Index might already exist
        
    try:
        op.create_index('idx_training_created', 'training_prompts', ['created_at'])
    except Exception:
        pass  # Index might already exist
        
    try:
        op.create_index('idx_training_priority', 'training_prompts', ['training_priority'])
    except Exception:
        pass  # Index might already exist


def downgrade() -> None:
    """Remove added columns from training_prompts table"""
    # Drop indexes first
    op.drop_index('idx_training_active', 'training_prompts')
    
    # Remove columns
    op.drop_column('training_prompts', 'is_active')
    op.drop_column('training_prompts', 'deleted_at') 
    op.drop_column('training_prompts', 'updated_at')