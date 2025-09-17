from typing import Dict
from .planner import build_plan
from .executor import execute_plan
from .summarizer import summarize
from .schemas import SatelliteSummary


def run_satellite_module(
    llm_planner,
    llm_summarizer,
    ticker: str,
    industry: str,
    sites_db: Dict[str, list],
    proxies_db: Dict[str, list]
) -> SatelliteSummary | dict:
    """
    High-level entry point: run the satellite pipeline for a given ticker/industry.

    Parameters
    ----------
    llm_planner : Callable
        A function/wrapper around your LLM that takes (system, user) and returns
        a raw JSON string. Used by build_plan().

    llm_summarizer : Callable
        A function/wrapper around your LLM that takes (system, user) and returns
        a raw JSON string. Used by summarize().

    ticker : str
        Stock ticker or instrument symbol (e.g., "TSLA", "CC=F").

    industry : str
        Industry/category label (used to skip irrelevant tickers like pure software).

    sites_db : Dict[str, list]
        Optional ephemeral hints of company sites.
        Example: {"TSLA": [{"name":"Giga Austin","lat":30.22,"lon":-97.62,"radius_km":4.0}]}
        Can be empty if not used.

    proxies_db : Dict[str, list]
        Optional ephemeral hints of industry-level proxies (e.g., cocoa belt, ports).
        Example: {"Agriculture": [{"name":"Ivory Coast cocoa belt","lat":7.6,"lon":-5.5,"radius_km":50.0}]}
        Can be empty if not used.

    Returns
    -------
    SatelliteSummary | dict
        If use_satellite=False, returns a simple dict with "Satellite not applicable".
        Otherwise, returns a SatelliteSummary model_dump() suitable for Strategist consumption.

    Flow
    ----
    1) PLAN   → call LLM planner to generate an ObservationPlan.
       - If plan.use_satellite is False, return a trivial "not applicable" block.

    2) OBSERVE → call executor to fetch and compute features for each planned target.
       - Produces an ObservationResult with numeric features and quality flags.

    3) EXPLAIN → call LLM summarizer to verify observations and produce human text.
       - Produces a SatelliteSummary with headline, bullets, confidence, attribution.

    Example return (dict form)
    --------------------------
    {
      "ticker": "TSLA",
      "headline": "Satellite: Night-lights and vegetation indicate ramp-up at Giga Austin",
      "bullets": [
        "NDVI +12% vs prior 30d in factory AOI (S2)",
        "Night-lights +9% vs baseline (VIIRS)"
      ],
      "confidence": 0.82,
      "attribution": ["S2","VIIRS"],
      "raw_counts": {"observations": 2, "gaps": 0}
    }
    """
    # 1) PLAN: use LLM to decide if satellite is relevant
    plan = build_plan(
        llm_planner,
        ticker=ticker,
        industry=industry,
        site_hints=sites_db.get(ticker, []),
        proxy_hints=proxies_db.get(industry, [])
    )

    # If planner says "not relevant", short-circuit with a trivial block
    if not plan.use_satellite:
        return {
            "ticker": ticker,
            "headline": "Satellite not applicable",
            "bullets": [plan.notes or "Pure software/internet industry; skipping satellite."],
            "confidence": 0.99,
            "attribution": []
        }

    # 2) OBSERVE: fetch Sentinel/MODIS/VIIRS data and compute requested features
    result = execute_plan(plan)

    # 3) EXPLAIN: pass observations to LLM summarizer for human-readable summary
    summary = summarize(llm_summarizer, ticker, industry, result)

    # Return as dict so it can be merged into Strategist inputs easily
    return summary.model_dump()
