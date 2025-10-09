"""Amplify risk manager overrides into high-conviction analyst signals."""

from __future__ import annotations

import json
from typing import Any, Mapping

from langchain_core.messages import HumanMessage

from src.agents.persona_utils import apply_persona
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress


PERSONA_NAME = "Atlas"
PERSONA_ROLE = "a risk amplifier translating override directives into trading instructions"
PERSONA_BACKSTORY = (
    "Atlas coordinated PM desks and risk teams, now narrating overrides to ensure humans understand the intent before acting."
)
PERSONA_INSTRUCTIONS = (
    "Respect the risk manager's directives but articulate whether to enforce, temper, or clarify them.",
    "Call out when conflicting flags (e.g., block new shorts vs prefer short) require nuance.",
    "Explain the guidance in first person, referencing the override fields that mattered.",
)
ALLOWED_SIGNALS = ["risk_force_cover", "risk_trim_short", "risk_build_long", "risk_guardrail", "risk_prefer_short", "risk_prefer_long", "risk_add_long", "risk_neutral"]

def _optional_non_negative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _summarise_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "preferred_direction",
        "target_long_shares",
        "target_short_shares",
        "block_new_shorts",
        "max_additional_short_shares",
        "force_cover_qty",
        "force_cover_reason",
        "force_buy_reason",
        "crash_allocator_raw_target_shares",
    )
    summary = {key: overrides.get(key) for key in keys if key in overrides}
    return summary


def _format_reason(parts: list[str], default: str) -> str:
    filtered = [part for part in parts if part]
    if not filtered:
        return default
    return "; ".join(filtered)


def _amplify_for_ticker(
    *,
    ticker: str,
    overrides: Mapping[str, Any] | None,
    position: Mapping[str, Any],
    state: AgentState,
    agent_id: str,
) -> dict[str, Any]:
    """Amplify risk overrides into persona-reviewed directives."""

    long_shares = _optional_non_negative_int(position.get("long")) or 0
    short_shares = _optional_non_negative_int(position.get("short")) or 0

    normalized_overrides: Mapping[str, Any] = overrides or {}
    overrides_summary = _summarise_overrides(normalized_overrides)

    def finalize(
        auto_signal: str,
        auto_confidence: int,
        reasoning: str,
        directives: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        directives_dict = dict(directives or {})
        bounded_confidence = int(max(0, min(auto_confidence, 100)))
        payload: dict[str, Any] = {
            "signal": auto_signal,
            "confidence": bounded_confidence,
            "reasoning": reasoning,
            "directives": directives_dict,
            "overrides": overrides_summary,
            "constraints": dict(directives_dict),
            "auto_reasoning": reasoning,
        }

        persona_context = {
            "ticker": ticker,
            "position": {"long": long_shares, "short": short_shares},
            "overrides": normalized_overrides,
            "directives": directives_dict,
            "model_recommendation": {
                "signal": auto_signal,
                "confidence": bounded_confidence,
                "reasoning": reasoning,
                "constraints": dict(directives_dict),
            },
        }

        decision = apply_persona(
            state=state,
            agent_id=agent_id,
            persona_name=PERSONA_NAME,
            persona_role=PERSONA_ROLE,
            persona_backstory=PERSONA_BACKSTORY,
            allowed_signals=ALLOWED_SIGNALS,
            persona_instructions=PERSONA_INSTRUCTIONS,
            payload=payload,
            context=persona_context,
        )
        payload.setdefault("meta", {})
        payload["meta"].update(
            {
                "auto_signal": auto_signal,
                "auto_confidence": bounded_confidence,
                "persona_overrode_auto": decision.signal != auto_signal,
            }
        )
        return payload

    if not normalized_overrides:
        return finalize(
            auto_signal="risk_neutral",
            auto_confidence=55,
            reasoning="Risk manager supplied no overrides for amplification.",
            directives={},
        )

    preferred_direction = str(normalized_overrides.get("preferred_direction") or "").lower()
    target_long = _optional_non_negative_int(normalized_overrides.get("target_long_shares"))
    target_short = _optional_non_negative_int(normalized_overrides.get("target_short_shares"))
    block_new_shorts = bool(normalized_overrides.get("block_new_shorts"))
    force_cover_qty = _positive_int(normalized_overrides.get("force_cover_qty"))
    force_cover_reason = normalized_overrides.get("force_cover_reason")
    max_short_add = _optional_non_negative_int(normalized_overrides.get("max_additional_short_shares"))

    reason_parts: list[str] = []
    directives: dict[str, Any] = {}

    if force_cover_qty:
        directives.update(
            {
                "action": "cover",
                "target_quantity": force_cover_qty,
                "enforce_immediately": True,
                "target_short_shares": max(0, target_short or 0),
            }
        )
        if block_new_shorts:
            directives["block_new_shorts"] = True
        if target_short is not None:
            reason_parts.append(f"Target short ≤ {target_short} shares.")
        if force_cover_reason:
            reason_parts.append(str(force_cover_reason))
        return finalize(
            auto_signal="risk_force_cover",
            auto_confidence=100,
            reasoning=_format_reason(
                reason_parts,
                f"Cover {force_cover_qty} short shares per risk override.",
            ),
            directives=directives,
        )

    if target_short is not None and short_shares > target_short:
        to_cover = short_shares - target_short
        directives.update(
            {
                "action": "cover",
                "target_quantity": to_cover,
                "target_short_shares": target_short,
            }
        )
        reason_parts.append(
            f"Trim short exposure from {short_shares} to {target_short} shares per risk cap."
        )
        return finalize(
            auto_signal="risk_trim_short",
            auto_confidence=97,
            reasoning=_format_reason(
                reason_parts,
                "Reduce short exposure according to risk cap.",
            ),
            directives=directives,
        )

    if preferred_direction == "long" and target_long is not None and long_shares < target_long:
        to_buy = target_long - long_shares
        directives.update(
            {
                "action": "buy",
                "target_quantity": to_buy,
                "target_long_shares": target_long,
            }
        )
        reason_parts.append(
            f"Increase long holdings by {to_buy} shares to reach target {target_long}."
        )
        return finalize(
            auto_signal="risk_build_long",
            auto_confidence=94,
            reasoning=_format_reason(
                reason_parts,
                "Build long exposure to satisfy risk manager target.",
            ),
            directives=directives,
        )

    if block_new_shorts:
        directives["block_new_shorts"] = True
        reason_parts.append("Block new short exposure per risk override.")

    if preferred_direction == "short":
        no_short_capacity = block_new_shorts or (max_short_add is not None and max_short_add <= 0)
        if short_shares <= 0 and no_short_capacity:
            reason_parts.append(
                "Short bias flagged but new shorts are blocked; maintain guardrail stance."
            )
            return finalize(
                auto_signal="risk_guardrail",
                auto_confidence=84,
                reasoning=_format_reason(
                    reason_parts,
                    "Short bias noted but new shorts prohibited; maintain guardrail.",
                ),
                directives=directives,
            )

        reason_parts.append(
            "Risk manager prefers short bias; align positioning with override."
        )
        directives.setdefault("preferred_direction", "short")
        return finalize(
            auto_signal="risk_prefer_short",
            auto_confidence=95,
            reasoning=_format_reason(
                reason_parts,
                "Adopt the risk manager short directive.",
            ),
            directives=directives,
        )

    if preferred_direction == "long":
        reason_parts.append(
            "Risk manager prefers a long tilt; honour direction even if analysts diverge."
        )
        directives.setdefault("preferred_direction", "long")
        return finalize(
            auto_signal="risk_prefer_long",
            auto_confidence=92,
            reasoning=_format_reason(
                reason_parts,
                "Adopt the risk manager long directive.",
            ),
            directives=directives,
        )

    if target_long is not None and long_shares < target_long:
        to_buy = target_long - long_shares
        directives.update(
            {
                "action": "buy",
                "target_quantity": to_buy,
                "target_long_shares": target_long,
            }
        )
        reason_parts.append(f"Add {to_buy} shares to reach long target {target_long}.")
        return finalize(
            auto_signal="risk_add_long",
            auto_confidence=88,
            reasoning=_format_reason(
                reason_parts,
                "Increase long exposure per risk guidance.",
            ),
            directives=directives,
        )

    if target_short is not None:
        reason_parts.append(f"Maintain short exposure ≤ {target_short} shares.")
    if target_long is not None:
        reason_parts.append(f"Maintain at least {target_long} long shares.")

    return finalize(
        auto_signal="risk_guardrail",
        auto_confidence=82,
        reasoning=_format_reason(
            reason_parts,
            "Risk manager overrides present but no immediate trade required.",
        ),
        directives=directives,
    )


##### Risk Override Amplifier Agent #####
def risk_override_amplifier_agent(state: AgentState, agent_id: str = "risk_override_amplifier"):
    """Surface deterministic directives so the LLM defaults to risk overrides."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    portfolio = data.get("portfolio", {})
    positions = portfolio.get("positions") or {}
    analyst_signals = data.get("analyst_signals", {})
    risk_payload = analyst_signals.get("risk_management_agent", {}) or {}

    amplified: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Amplifying overrides")
        snapshot = risk_payload.get(ticker) or {}
        overrides = snapshot.get("overrides") if isinstance(snapshot, Mapping) else None
        position = positions.get(ticker, {})

        amplified[ticker] = _amplify_for_ticker(
            ticker=ticker,
            overrides=overrides if isinstance(overrides, Mapping) else None,
            position=position if isinstance(position, Mapping) else {},
            state=state,
            agent_id=agent_id,
        )

    analyst_signals[agent_id] = amplified
    data["analyst_signals"] = analyst_signals

    message = HumanMessage(content=json.dumps(amplified), name=agent_id)

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(amplified, "Risk Override Amplifier")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }
