from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import json
from .schemas import ObservationPlan

# Light industry-based default skip list.
INDUSTRY_NO_NEED_SATELLITE = {
    "internet", "software", "saas", "fintech", "media", "advertising", "social media",
    "gaming (pure software)", "payments (pure software)"
}

PLANNER_SYSTEM = """You are an Observation Planner for a finance research agent.

You must decide IF satellite observation is useful for a given ticker and industry,
and if yes, propose up to TWO high-value observation targets with feasible FREE signals.

Rules:
- If the industry is clearly software/internet/SaaS/fintech/media (pure software), set "use_satellite": false.
- Otherwise, prefer targets that can be observed with FREE public missions at >=10 m resolution:
  • S2 (Sentinel-2 optical, 10 m): features allowed => "NDVI_mean_30d_vs_prev30d", "NDWI_ship_count_wow", "built_area_edge_delta_6m"
  • S1 (Sentinel-1 SAR, 10 m):   features allowed => "SAR_VV_delta_30d"
  • VIIRS (night lights):         features allowed => "night_lights_pct_delta_30d"
  • MODIS (fire/smoke):           features allowed => "smoke_days_14d"

Hard constraints:
- DO NOT request sub-10m details (e.g., cars, containers, small vehicles).
- DO NOT invent proprietary/commercial datasets.
- Use at most TWO targets; choose the most informative ones.
- If runtime hints are provided (site_hints or proxy_hints), prefer them; otherwise, you may return use_satellite=false.

Output:
- STRICT JSON matching the ObservationPlan schema:
  {
    "ticker": str,
    "industry": str | null,
    "use_satellite": bool,
    "targets": [
      {
        "name": str,
        "lat": float | null,
        "lon": float | null,
        "radius_km": float | null,
        "polygon_geojson": object | null,
        "sensors": [
          {"type": "S2" | "S1" | "VIIRS" | "MODIS", "features": [str, ...]}
        ],
        "reason": str
      },
      ...
    ],
    "fallbacks": [ Target, ... ],
    "notes": str
  }
Return JSON only, no commentary.
"""

PLANNER_USER_TEMPLATE = """Ticker: {ticker}
Industry: {industry}

Runtime site_hints (user-provided, optional; do NOT persist; may be empty):
{site_hints_json}

Runtime proxy_hints (user-provided, optional; do NOT persist; may be empty):
{proxy_hints_json}

Reminder:
- If industry is in {industries_no_need_satellite}., set use_satellite=false and explain why in 'notes'.
- Else, propose up to TWO targets from the hints above (or set use_satellite=false if nothing feasible/safe).
- Use the allowed features only (see system message).
- Output STRICT JSON ONLY (no markdown, no extra text).
"""


def build_plan(
    llm: Callable[..., str],
    ticker: str,
    industry: Optional[str] = None,
    *,
    site_hints: Optional[List[Dict[str, Any]]] = None,
    proxy_hints: Optional[List[Dict[str, Any]]] = None,
) -> ObservationPlan:
    """
    Call the planner LLM to produce an ObservationPlan.

    Parameters
    ----------
    llm : Callable[..., str]
        A thin wrapper around your LLM that accepts keyword args `system` and `user`
        and returns a raw JSON string. Example signature:
            def llm(*, system: str, user: str) -> str: ...
        The model MUST return STRICT JSON that matches ObservationPlan.

    ticker : str
        Ticker or instrument (e.g., "TSLA", "CC=F").

    industry : Optional[str]
        Industry/category hint used for the skip rule and target selection.
        If None, the planner will rely on hints and may still set use_satellite=false.

    site_hints : Optional[List[Dict[str, Any]]]
        Ephemeral, user-provided **company site** candidates. Each item is a dict that
        may contain ("name", "lat", "lon", "radius_km") OR ("name", "polygon_geojson").
        Example:
            [{"name":"Giga Austin","lat":30.221,"lon":-97.620,"radius_km":4.0}]
        These are not persisted by this module.

    proxy_hints : Optional[List[Dict[str, Any]]]
        Ephemeral **industry proxy** candidates (ports, belts, clusters, regions).
        Same shape as site_hints (name + either point+radius or polygon).

    Returns
    -------
    ObservationPlan
        A validated plan. On any parsing/LLM error, returns a safe fallback:
            ObservationPlan(
                ticker=ticker,
                industry=industry,
                use_satellite=False,
                targets=[],
                fallbacks=[],
                notes="Planner JSON parse error or no feasible targets."
            )
    """
    # Serialize ephemeral hints (never persisted).
    site_hints_json = json.dumps(site_hints or [], ensure_ascii=False, indent=2)
    proxy_hints_json = json.dumps(proxy_hints or [], ensure_ascii=False, indent=2)

    # Build the user message.
    user_msg = PLANNER_USER_TEMPLATE.format(
        ticker=ticker,
        industry=industry or "unknown",
        site_hints_json=site_hints_json,
        proxy_hints_json=proxy_hints_json,
        industries_no_need_satellite=INDUSTRY_NO_NEED_SATELLITE
    )

    # Call the LLM. We expect STRICT JSON back.
    try:
        raw = llm(system=PLANNER_SYSTEM, user=user_msg)
    except Exception as e:
        # LLM transport error → fail safe: skip satellite.
        return ObservationPlan(
            ticker=ticker,
            industry=industry,
            use_satellite=False,
            targets=[],
            fallbacks=[],
            notes=f"Planner call failed: {type(e).__name__}: {e}",
        )

    # Parse JSON strictly with Pydantic; return a safe fallback on any error.
    try:
        plan = ObservationPlan.model_validate_json(raw)
    except Exception as e:
        return ObservationPlan(
            ticker=ticker,
            industry=industry,
            use_satellite=False,
            targets=[],
            fallbacks=[],
            notes=f"Planner JSON parse error: {type(e).__name__}: {e}",
        )

    # Optional: lightweight post-checks to enforce our guardrails at runtime too.
    # If the industry looks like industry that no need satellite, force use_satellite=False as a sanity check.
    if industry and industry.strip().lower() in INDUSTRY_NO_NEED_SATELLITE:
        plan.use_satellite = False
        plan.targets = []
        plan.fallbacks = []
        plan.notes = (plan.notes or "") + " Skipped due to industry."

    # If the plan requested more than 2 targets, trim (LLM should obey, but we enforce).
    if len(plan.targets) > 2:
        plan.targets = plan.targets[:2]
        plan.notes = (plan.notes or "") + " Trimmed targets to 2."

    return plan

