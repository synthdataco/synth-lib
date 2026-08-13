"""LiteLLM admin client: budgeted virtual keys + spend reads."""

from __future__ import annotations

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


class LiteLLMAdmin:
    def __init__(self, base_url: str, master_key: str, timeout: int = 30):
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {master_key}"}
        self._timeout = timeout

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30), reraise=True)
    def generate_key(self, alias: str, max_budget_usd: float | None) -> str:
        """max_budget_usd=None mints an UNCAPPED key: no proxy-side enforcement, the driver's
        ledger polling is the only budget control."""
        resp = requests.post(
            f"{self._base}/key/generate",
            headers=self._headers,
            json={"key_alias": alias, "max_budget": max_budget_usd},
            timeout=self._timeout,
        )
        if not resp.ok:
            # surface the proxy's reason — a bare raise_for_status hides the body, and every
            # /key/generate failure mode (duplicate alias, schema, validation) is only in the body
            raise requests.HTTPError(
                f"{resp.status_code} on /key/generate for alias {alias!r}: {resp.text[:500]}", response=resp
            )
        return resp.json()["key"]

    # No retry here on purpose: failure is the outage signal (BudgetTracker pauses the clock).
    def key_info(self, key: str) -> dict:
        """{'spend': float, 'max_budget': float}. Raises a network exception if the proxy is down."""
        resp = requests.get(f"{self._base}/key/info", headers=self._headers, params={"key": key}, timeout=self._timeout)
        resp.raise_for_status()
        info = resp.json()["info"]
        return {"spend": float(info.get("spend") or 0.0), "max_budget": info.get("max_budget")}
