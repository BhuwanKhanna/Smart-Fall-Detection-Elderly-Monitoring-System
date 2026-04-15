from __future__ import annotations

from typing import Any

import requests


class APIClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get(self, path: str) -> Any:
        response = requests.get(f"{self.base_url}{path}", timeout=5)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        response = requests.post(f"{self.base_url}{path}", json=payload or {}, timeout=8)
        response.raise_for_status()
        return response.json()
