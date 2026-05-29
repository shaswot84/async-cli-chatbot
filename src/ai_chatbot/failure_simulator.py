"""Simulated provider failures for resilience testing."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


class SimulatedProviderError(RuntimeError):
    """Error raised when the failure simulator triggers an artificial failure."""

    def __init__(self, kind: str, status_code: int | None = None) -> None:
        super().__init__(f"simulated provider failure: {kind}")
        self.kind = kind
        self.status_code = status_code


@dataclass
class FailureSimulator:
    """Configurable failure injection for testing retry and error-handling paths."""

    enabled: bool = False
    rate: float = 0.0
    kind: str = "429"

    def enable(self) -> None:
        """Turn on failure simulation."""
        self.enabled = True
        logger.info(
            "failure_simulation_enabled",
            extra={"enabled": self.enabled, "failure_rate": self.rate, "failure_kind": self.kind},
        )

    def disable(self) -> None:
        """Turn off failure simulation."""
        self.enabled = False
        logger.info(
            "failure_simulation_enabled",
            extra={"enabled": self.enabled, "failure_rate": self.rate, "failure_kind": self.kind},
        )

    def set_rate(self, rate: float) -> None:
        """Set the probability (0.0-1.0) that a request triggers a failure."""
        if not 0 <= rate <= 1:
            raise ValueError("failure simulation rate must be between 0.0 and 1.0")
        self.rate = rate
        logger.info(
            "failure_simulation_enabled",
            extra={"enabled": self.enabled, "failure_rate": self.rate, "failure_kind": self.kind},
        )

    def set_kind(self, kind: str) -> None:
        """Set the kind of failure to simulate (429, 500, timeout, etc.)."""
        if kind not in supported_failure_kinds():
            raise ValueError(f"unsupported failure kind `{kind}`")
        self.kind = kind
        logger.info(
            "failure_simulation_enabled",
            extra={"enabled": self.enabled, "failure_rate": self.rate, "failure_kind": self.kind},
        )

    async def maybe_fail(self, request_id: str, model: str) -> None:
        """Optionally raise or delay based on the configured rate and kind."""
        if not self.enabled or self.rate <= 0 or random.random() >= self.rate:
            return

        logger.warning(
            "failure_simulation_triggered",
            extra={
                "request_id": request_id,
                "model": model,
                "failure_kind": self.kind,
                "failure_rate": self.rate,
            },
        )

        if self.kind == "timeout":
            raise httpx.ReadTimeout("simulated timeout")
        if self.kind == "malformed_json":
            raise ValueError("simulated malformed provider response")
        if self.kind == "empty_response":
            raise SimulatedProviderError("empty_response", status_code=200)
        if self.kind == "slow_response":
            await asyncio.sleep(2)
            return
        if self.kind == "500":
            raise SimulatedProviderError("provider_error", status_code=500)
        raise SimulatedProviderError("rate_limit", status_code=429)


def supported_failure_kinds() -> set[str]:
    """Return the set of failure kinds that can be simulated."""
    return {"429", "500", "timeout", "malformed_json", "empty_response", "slow_response"}
