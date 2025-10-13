"""Helper functions for LLM"""

import asyncio
import concurrent.futures
import json
import os
from collections.abc import Callable

from pydantic import BaseModel

from src.graph.state import AgentState
from src.llm.models import get_model, get_model_info
from src.utils.progress import progress
from src.utils.runtime import resolve_int_env


def _parse_timeout(value: str | None, default: float) -> float | None:
    """Parse timeout values allowing <=0 to disable the timeout."""
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return None if parsed <= 0 else parsed


LLM_CALL_TIMEOUT_SECONDS = _parse_timeout(os.getenv("LLM_CALL_TIMEOUT_SECONDS"), 120.0)
LLM_MAX_RETRIES = resolve_int_env("LLM_MAX_RETRIES", 3)
LLM_ASYNC_MAX_CONCURRENCY = resolve_int_env("LLM_ASYNC_MAX_CONCURRENCY", 8)

_ASYNC_LLM_SEMAPHORE: asyncio.Semaphore | None = None


def _get_async_llm_semaphore() -> asyncio.Semaphore:
    global _ASYNC_LLM_SEMAPHORE
    if _ASYNC_LLM_SEMAPHORE is None:
        _ASYNC_LLM_SEMAPHORE = asyncio.Semaphore(LLM_ASYNC_MAX_CONCURRENCY)
    return _ASYNC_LLM_SEMAPHORE


def _resolve_model_configuration(state: AgentState | None, agent_name: str | None) -> tuple[str, str]:
    if state and agent_name:
        model_name, model_provider = get_agent_model_config(state, agent_name)
    else:
        model_name = "gpt-4.1"
        model_provider = "OPENAI"
    return model_name, model_provider


def _extract_api_keys(state: AgentState | None):
    if not state:
        return None
    request = state.get("metadata", {}).get("request")
    if request and hasattr(request, "api_keys"):
        return request.api_keys
    return None


def _initialize_llm(
    *,
    model_name: str,
    model_provider: str,
    api_keys,
    pydantic_model: type[BaseModel],
):
    model_info = get_model_info(model_name, model_provider)
    llm = get_model(model_name, model_provider, api_keys)
    if llm is None:
        return None, model_info

    if not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )

    return llm, model_info


def call_llm(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int | None = None,
    default_factory=None,
) -> BaseModel:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates and model config extraction
        state: Optional state object to extract agent-specific model configuration
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """
    model_name, model_provider = _resolve_model_configuration(state, agent_name)
    api_keys = _extract_api_keys(state)

    llm, model_info = _initialize_llm(
        model_name=model_name,
        model_provider=model_provider,
        api_keys=api_keys,
        pydantic_model=pydantic_model,
    )

    if llm is None:
        if agent_name:
            progress.update_status(agent_name, None, "LLM unavailable, using fallback decision")
        if default_factory:
            return default_factory()
        return create_default_response(pydantic_model)

    retries = max_retries if max_retries is not None else LLM_MAX_RETRIES
    # Call the LLM with retries
    for attempt in range(retries):
        try:
            # Call the LLM
            result = _invoke_with_timeout(llm, prompt, LLM_CALL_TIMEOUT_SECONDS)

            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                return result

        except TimeoutError as exc:
            if agent_name:
                progress.update_status(
                    agent_name,
                    None,
                    f"LLM timeout after {LLM_CALL_TIMEOUT_SECONDS}s - retry {attempt + 1}/{retries}",
                )
            if attempt == retries - 1:
                print(f"Timeout in LLM call after {retries} attempts: {exc}")
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)
        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{retries}")

            if attempt == retries - 1:
                print(f"Error in LLM call after {retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)


async def async_call_llm(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int | None = None,
    default_factory: Callable[[], BaseModel] | None = None,
) -> BaseModel:
    """Async wrapper for structured LLM calls with retry and concurrency limits."""

    model_name, model_provider = _resolve_model_configuration(state, agent_name)
    api_keys = _extract_api_keys(state)

    llm, model_info = _initialize_llm(
        model_name=model_name,
        model_provider=model_provider,
        api_keys=api_keys,
        pydantic_model=pydantic_model,
    )

    if llm is None:
        if agent_name:
            progress.update_status(agent_name, None, "LLM unavailable, using fallback decision")
        if default_factory:
            return default_factory()
        return create_default_response(pydantic_model)

    retries = max_retries if max_retries is not None else LLM_MAX_RETRIES
    semaphore = _get_async_llm_semaphore()

    async with semaphore:
        for attempt in range(retries):
            try:
                result = await _ainvoke_with_timeout(llm, prompt, LLM_CALL_TIMEOUT_SECONDS)

                if model_info and not model_info.has_json_mode():
                    parsed_result = extract_json_from_response(result.content)
                    if parsed_result:
                        return pydantic_model(**parsed_result)
                else:
                    return result

            except TimeoutError as exc:
                if agent_name:
                    progress.update_status(
                        agent_name,
                        None,
                        f"Async LLM timeout after {LLM_CALL_TIMEOUT_SECONDS}s - retry {attempt + 1}/{retries}",
                    )
                if attempt == retries - 1:
                    if default_factory:
                        return default_factory()
                    print(f"Async timeout in LLM call after {retries} attempts: {exc}")
                    return create_default_response(pydantic_model)
            except Exception as exc:  # pragma: no cover - defensive logging
                if agent_name:
                    progress.update_status(agent_name, None, f"Async error - retry {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    if default_factory:
                        return default_factory()
                    print(f"Async error in LLM call after {retries} attempts: {exc}")
                    return create_default_response(pydantic_model)
            await asyncio.sleep(min(2.0, 0.5 * (attempt + 1)))

    return create_default_response(pydantic_model)


def create_default_response(model_class: type[BaseModel]) -> BaseModel:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> dict | None:
    """Extracts JSON from markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None


def _invoke_with_timeout(llm, prompt, timeout_seconds: float | None):
    """Invoke the LLM with an optional timeout to avoid hanging the workflow."""
    if not timeout_seconds:
        return llm.invoke(prompt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(llm.invoke, prompt)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"LLM call exceeded {timeout_seconds} seconds") from exc


async def _ainvoke_with_timeout(llm, prompt, timeout_seconds: float | None):
    """Async counterpart to invoke LLM with optional timeout enforcement."""
    if hasattr(llm, "ainvoke"):
        coroutine = llm.ainvoke(prompt)
    else:
        coroutine = asyncio.to_thread(llm.invoke, prompt)

    if not timeout_seconds:
        return await coroutine

    return await asyncio.wait_for(coroutine, timeout_seconds)


def get_agent_model_config(state, agent_name):
    """
    Get model configuration for a specific agent from the state.
    Falls back to global model configuration if agent-specific config is not available.
    Always returns valid model_name and model_provider values.
    """
    request = state.get("metadata", {}).get("request")

    if request and hasattr(request, "get_agent_model_config"):
        # Get agent-specific model configuration
        model_name, model_provider = request.get_agent_model_config(agent_name)
        # Ensure we have valid values
        if model_name and model_provider:
            return model_name, model_provider.value if hasattr(model_provider, "value") else str(model_provider)

    # Fall back to global configuration (system defaults)
    model_name = state.get("metadata", {}).get("model_name") or "gpt-4.1"
    model_provider = state.get("metadata", {}).get("model_provider") or "OPENAI"

    # Convert enum to string if necessary
    if hasattr(model_provider, "value"):
        model_provider = model_provider.value

    return model_name, model_provider
