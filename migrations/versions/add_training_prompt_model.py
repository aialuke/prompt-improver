"""Add TrainingPrompt model for ML pipeline

Revision ID: training_prompt_v1
Revises: 29408fe0f0d5
Create Date: 2025-01-20

"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision: str = "training_prompt_v1"
down_revision: str | Sequence[str] | None = "29408fe0f0d5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add TrainingPrompt table following 2025 SQLModel patterns"""
    op.create_table(
        'training_prompts',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('prompt_text', sa.String(10000), nullable=False, index=True),
        sa.Column('enhancement_result', JSON, nullable=False),
        sa.Column('data_source', sa.String(50), nullable=False, index=True),
        sa.Column('training_priority', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, index=True),
        sa.Column('session_id', sa.String(255), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['prompt_sessions.session_id']),
    )
    
    # Add indexes for performance (2025 best practice)
    op.create_index('idx_training_data_source', 'training_prompts', ['data_source'])
    op.create_index('idx_training_active', 'training_prompts', ['is_active'])
    op.create_index('idx_training_created', 'training_prompts', ['created_at'])
    op.create_index('idx_training_priority', 'training_prompts', ['training_priority'])


def downgrade() -> None:
    """Remove TrainingPrompt table"""
    # Drop indexes first
    op.drop_index('idx_training_priority', 'training_prompts')
    op.drop_index('idx_training_created', 'training_prompts')
    op.drop_index('idx_training_active', 'training_prompts')
    op.drop_index('idx_training_data_source', 'training_prompts')
    
    # Drop table
    op.drop_table('training_prompts')