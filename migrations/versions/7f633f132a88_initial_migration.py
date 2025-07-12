"""Initial migration

Revision ID: 7f633f132a88
Revises: 
Create Date: 2025-07-11 21:29:14.913954

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7f633f132a88'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create userfeedback table if it doesn't exist
    op.create_table(
        'userfeedback',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('feedback_id', sa.UUID(), nullable=False),
        sa.Column('original_prompt', sa.Text(), nullable=False),
        sa.Column('improved_prompt', sa.Text(), nullable=False),
        sa.Column('user_rating', sa.Integer(), nullable=False),
        sa.Column('applied_rules', sa.JSON(), nullable=False),
        sa.Column('user_context', sa.JSON(), nullable=True),
        sa.Column('improvement_areas', sa.JSON(), nullable=True),
        sa.Column('user_notes', sa.Text(), nullable=True),
        sa.Column('session_id', sa.String(length=100), nullable=True),
        sa.Column('ml_optimized', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('model_id', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('feedback_id'),
        sa.CheckConstraint('user_rating >= 1 AND user_rating <= 5', name='chk_user_rating_range')
    )
    
    # Create index on session_id
    op.create_index('ix_userfeedback_session_id', 'userfeedback', ['session_id'])
    
    # Create index on created_at
    op.create_index('ix_userfeedback_created_at', 'userfeedback', ['created_at'])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index('ix_userfeedback_created_at', table_name='userfeedback')
    op.drop_index('ix_userfeedback_session_id', table_name='userfeedback')
    
    # Drop table
    op.drop_table('userfeedback')
