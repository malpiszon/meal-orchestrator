import pytest

from meal_orchestrator.providers import build_provider_adapter


def test_build_provider_adapter_rejects_unsupported_provider() -> None:
    with pytest.raises(ValueError, match="unsupported provider"):
        build_provider_adapter("missing_provider")
