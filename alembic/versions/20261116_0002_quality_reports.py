"""Add quality_reports for data quality, training gate, and reproduction evidence."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20261116_0002"
down_revision = "20260316_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the append-only quality report evidence table."""

    op.create_table(
        "quality_reports",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("policy_version", sa.String(length=32), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=True),
        sa.Column("model_version_id", sa.String(length=36), nullable=True),
        sa.Column("horizon_days", sa.Integer(), nullable=True),
        sa.Column("check_count", sa.Integer(), nullable=False),
        sa.Column("failure_count", sa.Integer(), nullable=False),
        sa.Column("warning_count", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("checks_json", sa.JSON(), nullable=False),
        sa.Column("request_json", sa.JSON(), nullable=False),
        sa.Column("context_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            ["dataset_versions.id"],
            name="fk_quality_reports_dataset_version_id_dataset_versions",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_quality_reports_model_version_id_model_versions",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_quality_reports"),
    )
    op.create_index("ix_quality_reports_stage", "quality_reports", ["stage"], unique=False)
    op.create_index("ix_quality_reports_status", "quality_reports", ["status"], unique=False)
    op.create_index(
        "ix_quality_reports_dataset_version_id",
        "quality_reports",
        ["dataset_version_id"],
        unique=False,
    )
    op.create_index(
        "ix_quality_reports_model_version_id",
        "quality_reports",
        ["model_version_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop the quality report evidence table."""

    op.drop_index("ix_quality_reports_model_version_id", table_name="quality_reports")
    op.drop_index("ix_quality_reports_dataset_version_id", table_name="quality_reports")
    op.drop_index("ix_quality_reports_status", table_name="quality_reports")
    op.drop_index("ix_quality_reports_stage", table_name="quality_reports")
    op.drop_table("quality_reports")
