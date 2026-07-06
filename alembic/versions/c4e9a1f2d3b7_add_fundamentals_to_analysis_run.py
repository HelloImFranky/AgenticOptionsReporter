"""add fundamentals + data_warnings to analysis_run

Revision ID: c4e9a1f2d3b7
Revises: a7d2f4c9e1b0
Create Date: 2026-07-06 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4e9a1f2d3b7'
down_revision: Union[str, Sequence[str], None] = 'a7d2f4c9e1b0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('analysis_run', schema=None) as batch_op:
        batch_op.add_column(sa.Column('fundamentals', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('data_warnings', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('analysis_run', schema=None) as batch_op:
        batch_op.drop_column('data_warnings')
        batch_op.drop_column('fundamentals')
