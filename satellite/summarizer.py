from .schemas import ObservationResult, SatelliteSummary
import json

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# System prompt: sets rules for the LLM summarizer.
# - Ensures only public/free missions are referenced.
# - Requires dropping low-quality (<0.6) observations.
# - Encourages cautious tone and evidence-weighted phrasing.
# - Defines the required output structure.
SUMMARIZER_SYSTEM = """You verify and summarize satellite observations for a finance report.
Observations are derived from public missions (S2/S1/VIIRS/MODIS) at >=10m resolution.
Drop low-quality observations (quality < 0.6) or clearly implausible values.
Produce: a one-line HEADLINE, 2–4 concise bullets, an overall confidence (0..1),
and an attribution list like ["S2","VIIRS"]. Prefer cautious, evidence-weighted language."""

# User template: injects the ticker/industry context and the raw ObservationResult JSON.
# The LLM is asked to return STRICT JSON only, no extra text or markdown.
SUMMARIZER_USER_TMPL = """Ticker: {ticker}
Industry: {industry}
Observations JSON:
{observations_json}

Return STRICT JSON:
{{
  "headline": "...",
  "bullets": ["...", "..."],
  "confidence": 0.0,
  "attribution": ["S2"]
}}"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize(llm, ticker: str, industry: str, result: ObservationResult) -> SatelliteSummary:
    """
    Summarize satellite observations into a SatelliteSummary.

    Parameters
    ----------
    llm : Callable
        Wrapper around an LLM client. Must accept keyword args `system` and `user`,
        and return a raw JSON string. Example:
            def llm(*, system: str, user: str) -> str: ...

    ticker : str
        Stock ticker or instrument (e.g., "TSLA", "CC=F").

    industry : str
        Industry label (for context; helps the LLM interpret signals).

    result : ObservationResult
        Structured output from the executor containing numeric observations,
        quality scores, and gap reasons.

    Returns
    -------
    SatelliteSummary
        Pydantic model with:
        - headline (one sentence)
        - bullets (2–4 concise evidence points)
        - confidence (0..1)
        - attribution (list of sensors)
        - raw_counts (observations vs gaps, for transparency)

    Notes
    -----
    - We serialize ObservationResult to JSON and feed it to the LLM.
    - The LLM must return STRICT JSON; we parse it with json.loads.
    - Any dropped observations (low quality) should not appear in the bullets.
    - The output is cautious: no hype, only evidence-based language.
    """
    # Serialize ObservationResult → JSON string for LLM input
    obs_json = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

    # Fill the user template with context + observations
    user = SUMMARIZER_USER_TMPL.format(
        ticker=ticker,
        industry=industry,
        observations_json=obs_json
    )

    # Call the LLM to produce a summary (must be STRICT JSON)
    raw = llm(system=SUMMARIZER_SYSTEM, user=user)
    data = json.loads(raw)

    # Wrap in SatelliteSummary Pydantic model for downstream use
    return SatelliteSummary(
        ticker=ticker,
        headline=data["headline"],
        bullets=data["bullets"],
        confidence=float(data["confidence"]),
        attribution=list(data["attribution"]),
        raw_counts={
            "observations": len(result.observations),
            "gaps": len(result.gaps)
        }
    )
