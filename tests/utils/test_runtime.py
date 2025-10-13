from __future__ import annotations

from src.utils import runtime


def test_async_personas_default_true(monkeypatch):
    monkeypatch.delenv("ASYNC_PERSONAS", raising=False)
    assert runtime.async_personas_enabled() is True


def test_async_personas_respects_explicit_default(monkeypatch):
    monkeypatch.delenv("ASYNC_PERSONAS", raising=False)
    assert runtime.async_personas_enabled(default=False) is False


def test_async_personas_truthy_values(monkeypatch):
    monkeypatch.setenv("ASYNC_PERSONAS", "True")
    assert runtime.async_personas_enabled() is True
    monkeypatch.setenv("ASYNC_PERSONAS", "YES")
    assert runtime.async_personas_enabled() is True
    monkeypatch.setenv("ASYNC_PERSONAS", "1")
    assert runtime.async_personas_enabled() is True


def test_async_personas_falsy_values(monkeypatch):
    monkeypatch.setenv("ASYNC_PERSONAS", "0")
    assert runtime.async_personas_enabled() is False
    monkeypatch.setenv("ASYNC_PERSONAS", "off")
    assert runtime.async_personas_enabled() is False
    monkeypatch.setenv("ASYNC_PERSONAS", "False")
    assert runtime.async_personas_enabled() is False


def test_async_personas_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("ASYNC_PERSONAS", "maybe")
    assert runtime.async_personas_enabled() is True


def test_resolve_int_env_defaults(monkeypatch):
    monkeypatch.delenv("LLM_ASYNC_MAX_CONCURRENCY", raising=False)
    assert runtime.resolve_int_env("LLM_ASYNC_MAX_CONCURRENCY", 5) == 5


def test_resolve_int_env_parses_positive(monkeypatch):
    monkeypatch.setenv("LLM_ASYNC_MAX_CONCURRENCY", "12")
    assert runtime.resolve_int_env("LLM_ASYNC_MAX_CONCURRENCY", 5) == 12


def test_resolve_int_env_handles_invalid(monkeypatch):
    monkeypatch.setenv("LLM_ASYNC_MAX_CONCURRENCY", "-2")
    assert runtime.resolve_int_env("LLM_ASYNC_MAX_CONCURRENCY", 7) == 7
    monkeypatch.setenv("LLM_ASYNC_MAX_CONCURRENCY", "oops")
    assert runtime.resolve_int_env("LLM_ASYNC_MAX_CONCURRENCY", 4) == 4
