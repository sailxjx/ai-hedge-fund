from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.graph.state import AgentState
from src.utils.llm import async_call_llm, call_llm


class PersonaDecision(BaseModel):
    signal: str
    confidence: float
    reasoning: str
    constraints: dict[str, Any] | None = None


def _to_builtin(value: Any) -> Any:
    """Best-effort conversion of numpy/pandas objects to Python builtins."""

    if isinstance(value, np.generic):
        return value.item()

    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except TypeError:
            pass

    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(v) for v in value]

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except TypeError:
            pass

    return value


def _normalize_context(context: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _to_builtin(v) for k, v in context.items()}


def _build_persona_prompt(
    *,
    persona_name: str,
    persona_role: str,
    persona_backstory: str,
    allowed_signals: Sequence[str],
    context: dict[str, Any],
    default_decision: dict[str, Any],
    persona_instructions: Iterable[str] | None,
) -> tuple[ChatPromptTemplate, Callable[[], PersonaDecision]]:
    normalized_context = _normalize_context(context)
    context_json = json.dumps(normalized_context, indent=2, sort_keys=True)
    escaped_context_json = context_json.replace("{", "{{").replace("}", "}}")

    instruction_lines = list(persona_instructions or [])

    system_message = f"You are {persona_name}, {persona_role}. {persona_backstory}\n" "Study the provided quantitative context, weigh narrative vs statistics, and issue a trading judgement.\n" "You may adopt or override the model recommendation when your qualitative assessment warrants it.\n" "Always speak in first person, referencing the persona's judgement."

    if instruction_lines:
        system_message += "\n" + "\n".join(instruction_lines)

    human_message = "Context:\n" f"{escaped_context_json}\n\n" "Allowed signals: " + ", ".join(str(sig) for sig in allowed_signals) + "\n" "Respond with strict JSON containing the keys: signal (one of the allowed signals), confidence (0-100 number), reasoning (string), constraints (object or null)."

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", human_message),
        ]
    )

    def _default_factory() -> PersonaDecision:
        constraints = default_decision.get("constraints")
        return PersonaDecision(
            signal=str(default_decision.get("signal", "neutral")),
            confidence=float(default_decision.get("confidence", 0.0)),
            reasoning=str(default_decision.get("reasoning", "No reasoning provided.")),
            constraints=constraints if isinstance(constraints, dict) else None,
        )

    return prompt_template, _default_factory


def invoke_persona(
    *,
    state: AgentState,
    agent_id: str,
    persona_name: str,
    persona_role: str,
    persona_backstory: str,
    allowed_signals: Sequence[str],
    context: dict[str, Any],
    default_decision: dict[str, Any],
    persona_instructions: Iterable[str] | None = None,
) -> PersonaDecision:
    """Call an LLM persona with structured context and a deterministic fallback."""

    prompt_template, default_factory = _build_persona_prompt(
        persona_name=persona_name,
        persona_role=persona_role,
        persona_backstory=persona_backstory,
        allowed_signals=allowed_signals,
        context=context,
        default_decision=default_decision,
        persona_instructions=persona_instructions,
    )

    prompt_value = prompt_template.invoke({})

    return call_llm(
        prompt=prompt_value,
        pydantic_model=PersonaDecision,
        agent_name=agent_id,
        state=state,
        default_factory=default_factory,
    )


async def async_invoke_persona(
    *,
    state: AgentState,
    agent_id: str,
    persona_name: str,
    persona_role: str,
    persona_backstory: str,
    allowed_signals: Sequence[str],
    context: dict[str, Any],
    default_decision: dict[str, Any],
    persona_instructions: Iterable[str] | None = None,
) -> PersonaDecision:
    """Async equivalent of invoke_persona powered by async_call_llm."""

    prompt_template, default_factory = _build_persona_prompt(
        persona_name=persona_name,
        persona_role=persona_role,
        persona_backstory=persona_backstory,
        allowed_signals=allowed_signals,
        context=context,
        default_decision=default_decision,
        persona_instructions=persona_instructions,
    )

    prompt_value = prompt_template.invoke({})

    return await async_call_llm(
        prompt=prompt_value,
        pydantic_model=PersonaDecision,
        agent_name=agent_id,
        state=state,
        default_factory=default_factory,
    )


def resolve_constraints(decision: PersonaDecision, fallback: dict[str, Any] | None) -> dict[str, Any]:
    """Return persona constraints, falling back to model defaults when persona omitted them."""

    if decision.constraints is None:
        return dict(fallback or {})
    return dict(decision.constraints)


def apply_persona(
    *,
    state: AgentState,
    agent_id: str,
    persona_name: str,
    persona_role: str,
    persona_backstory: str,
    allowed_signals: Sequence[str],
    persona_instructions: Iterable[str] | None,
    payload: dict[str, Any],
    context: dict[str, Any],
) -> PersonaDecision:
    """Invoke persona and mutate payload with final decision. Returns the decision object."""

    payload.setdefault("constraints", {})
    default = {
        "signal": payload.get("signal", "neutral"),
        "confidence": payload.get("confidence", 0),
        "reasoning": payload.get("reasoning", "No reasoning provided."),
        "constraints": payload.get("constraints"),
    }

    decision = invoke_persona(
        state=state,
        agent_id=agent_id,
        persona_name=persona_name,
        persona_role=persona_role,
        persona_backstory=persona_backstory,
        allowed_signals=allowed_signals,
        context=context,
        default_decision=default,
        persona_instructions=persona_instructions,
    )

    final_constraints = resolve_constraints(decision, payload.get("constraints"))
    payload["signal"] = decision.signal
    payload["confidence"] = int(max(0, min(round(decision.confidence), 100)))
    payload["reasoning"] = decision.reasoning
    payload["constraints"] = final_constraints
    return decision


async def async_apply_persona(
    *,
    state: AgentState,
    agent_id: str,
    persona_name: str,
    persona_role: str,
    persona_backstory: str,
    allowed_signals: Sequence[str],
    persona_instructions: Iterable[str] | None,
    payload: dict[str, Any],
    context: dict[str, Any],
) -> PersonaDecision:
    """Async variant of apply_persona for async persona workflows."""

    payload.setdefault("constraints", {})
    default = {
        "signal": payload.get("signal", "neutral"),
        "confidence": payload.get("confidence", 0),
        "reasoning": payload.get("reasoning", "No reasoning provided."),
        "constraints": payload.get("constraints"),
    }

    decision = await async_invoke_persona(
        state=state,
        agent_id=agent_id,
        persona_name=persona_name,
        persona_role=persona_role,
        persona_backstory=persona_backstory,
        allowed_signals=allowed_signals,
        context=context,
        default_decision=default,
        persona_instructions=persona_instructions,
    )

    final_constraints = resolve_constraints(decision, payload.get("constraints"))
    payload["signal"] = decision.signal
    payload["confidence"] = int(max(0, min(round(decision.confidence), 100)))
    payload["reasoning"] = decision.reasoning
    payload["constraints"] = final_constraints
    return decision


def persona_from_observations(
    *,
    state: AgentState,
    agent_id: str,
    persona_name: str,
    persona_role: str,
    persona_backstory: str,
    allowed_signals: Sequence[str],
    observations: dict[str, Any],
    persona_instructions: Iterable[str] | None = None,
    default_signal: str = "neutral",
    default_confidence: float = 0.0,
    default_reasoning: str = "Fell back to neutral after missing persona output.",
) -> PersonaDecision:
    """Invoke a persona using raw observations without precomputed signals."""

    context = {
        "observations": _normalize_context(observations),
    }
    default_decision = {
        "signal": default_signal,
        "confidence": default_confidence,
        "reasoning": default_reasoning,
        "constraints": {},
    }
    return invoke_persona(
        state=state,
        agent_id=agent_id,
        persona_name=persona_name,
        persona_role=persona_role,
        persona_backstory=persona_backstory,
        allowed_signals=allowed_signals,
        context=context,
        default_decision=default_decision,
        persona_instructions=persona_instructions,
    )


async def async_persona_from_observations(
    *,
    state: AgentState,
    agent_id: str,
    persona_name: str,
    persona_role: str,
    persona_backstory: str,
    allowed_signals: Sequence[str],
    observations: dict[str, Any],
    persona_instructions: Iterable[str] | None = None,
    default_signal: str = "neutral",
    default_confidence: float = 0.0,
    default_reasoning: str = "Fell back to neutral after missing persona output.",
) -> PersonaDecision:
    """Async persona invocation using raw observations."""

    context = {
        "observations": _normalize_context(observations),
    }
    default_decision = {
        "signal": default_signal,
        "confidence": default_confidence,
        "reasoning": default_reasoning,
        "constraints": {},
    }
    return await async_invoke_persona(
        state=state,
        agent_id=agent_id,
        persona_name=persona_name,
        persona_role=persona_role,
        persona_backstory=persona_backstory,
        allowed_signals=allowed_signals,
        context=context,
        default_decision=default_decision,
        persona_instructions=persona_instructions,
    )
