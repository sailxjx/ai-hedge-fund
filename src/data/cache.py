from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Cache:
    """Persistent cache for API responses keyed by request parameters."""

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        base_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "ai-hedge-fund"
        base_dir.mkdir(parents=True, exist_ok=True)

        self._cache_dir = base_dir
        self._prices_cache: dict[str, list[dict[str, Any]]] = self._load_resource("prices")
        self._financial_metrics_cache: dict[str, list[dict[str, Any]]] = self._load_resource("financial_metrics")
        self._line_items_cache: dict[str, list[dict[str, Any]]] = self._load_resource("line_items")
        self._insider_trades_cache: dict[str, list[dict[str, Any]]] = self._load_resource("insider_trades")
        self._company_news_cache: dict[str, list[dict[str, Any]]] = self._load_resource("company_news")

    def _resource_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.json"

    def _load_resource(self, name: str) -> dict[str, list[dict[str, Any]]]:
        path = self._resource_path(name)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _persist_resource(self, name: str, data: dict[str, list[dict[str, Any]]]) -> None:
        path = self._resource_path(name)
        tmp_path = path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(data))
            tmp_path.replace(path)
        except OSError:
            # If persisting fails (e.g., due to permissions), ignore and keep in-memory cache
            pass

    @staticmethod
    def _merge_data(existing: list[dict[str, Any]] | None, new_data: list[dict[str, Any]], key_field: str) -> list[dict[str, Any]]:
        """Merge payloads on a unique key to avoid duplicate rows."""
        if not existing:
            return new_data

        keyed_existing = {item.get(key_field): item for item in existing if key_field in item}
        passthrough = [item for item in existing if key_field not in item]

        for item in new_data:
            key = item.get(key_field)
            if key is None:
                passthrough.append(item)
            else:
                keyed_existing[key] = item

        merged = list(keyed_existing.values()) + passthrough
        return merged

    def get_prices(self, key: str) -> list[dict[str, Any]] | None:
        return self._prices_cache.get(key)

    def set_prices(self, key: str, data: list[dict[str, Any]]) -> None:
        merged = self._merge_data(self._prices_cache.get(key), data, key_field="time")
        merged.sort(key=lambda item: item.get("time"))
        self._prices_cache[key] = merged
        self._persist_resource("prices", self._prices_cache)

    def get_financial_metrics(self, key: str) -> list[dict[str, Any]] | None:
        return self._financial_metrics_cache.get(key)

    def set_financial_metrics(self, key: str, data: list[dict[str, Any]]) -> None:
        merged = self._merge_data(self._financial_metrics_cache.get(key), data, key_field="report_period")
        self._financial_metrics_cache[key] = merged
        self._persist_resource("financial_metrics", self._financial_metrics_cache)

    def get_line_items(self, key: str) -> list[dict[str, Any]] | None:
        return self._line_items_cache.get(key)

    def set_line_items(self, key: str, data: list[dict[str, Any]]) -> None:
        merged = self._merge_data(self._line_items_cache.get(key), data, key_field="report_period")
        self._line_items_cache[key] = merged
        self._persist_resource("line_items", self._line_items_cache)

    def get_insider_trades(self, key: str) -> list[dict[str, Any]] | None:
        return self._insider_trades_cache.get(key)

    def set_insider_trades(self, key: str, data: list[dict[str, Any]]) -> None:
        merged = self._merge_data(self._insider_trades_cache.get(key), data, key_field="filing_date")
        self._insider_trades_cache[key] = merged
        self._persist_resource("insider_trades", self._insider_trades_cache)

    def get_company_news(self, key: str) -> list[dict[str, Any]] | None:
        return self._company_news_cache.get(key)

    def set_company_news(self, key: str, data: list[dict[str, Any]]) -> None:
        merged = self._merge_data(self._company_news_cache.get(key), data, key_field="date")
        self._company_news_cache[key] = merged
        self._persist_resource("company_news", self._company_news_cache)


# Global cache instance stored under a user-specific directory
_cache = Cache()


def get_cache() -> Cache:
    """Get the global cache instance."""
    return _cache
