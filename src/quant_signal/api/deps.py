"""API dependency helpers."""

from quant_signal.governance import GovernanceService
from quant_signal.serving.service import SignalService


def get_signal_service() -> SignalService:
    """Return the read-only signal service."""

    return SignalService()


def get_governance_service() -> GovernanceService:
    """Return the read-only governance service."""

    return GovernanceService()
