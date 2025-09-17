from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


SensorType = Literal["S2", "S1", "VIIRS", "MODIS"]
"""
Which public Earth‑observation source to use.

- "S2": Sentinel‑2 optical imagery (10 m), good for NDVI/NDWI (vegetation/water).
- "S1": Sentinel‑1 SAR radar imagery (10 m), good for backscatter/structure change, cloud‑independent.
- "VIIRS": Night‑lights and active‑fire products (coarser), daily global.
- "MODIS": Coarser multispectral & atmosphere products (e.g., smoke/fire), daily global.

Note: These are *data sources*, not features. Features are defined per sensor below.
"""


# -----------------------------------------------------------------------------
# PLAN (LLM → code): What to observe?
# -----------------------------------------------------------------------------

class SensorSpec(BaseModel):
    """
    A single sensor + its requested derived features the executor should compute.

    Example:
      SensorSpec(
        type="S2",
        features=["NDVI_mean_30d_vs_prev30d"]
      )
    """
    type: SensorType = Field(
        ...,
        description="Public EO source to query (S2, S1, VIIRS, MODIS)."
    )
    features: List[str] = Field(
        ...,
        description=(
            "Names of lightweight, deterministic features to compute on the fetched data. "
            "Examples: "
            "- 'NDVI_mean_30d_vs_prev30d' (Sentinel-2)\n"
            "- 'NDWI_ship_count_wow' (Sentinel-2)\n"
            "- 'SAR_VV_delta_30d' (Sentinel-1)\n"
            "- 'night_lights_pct_delta_30d' (VIIRS)\n"
            "- 'smoke_days_14d' (MODIS/VIIRS)"
        )
    )


class Target(BaseModel):
    """
    One geographic target (site, port, belt/region) to observe.

    Privacy & safety:
    - All geo fields are OPTIONAL and ephemeral; do not persist sensitive coordinates.
    - If no precise geometry is available/safe, you can supply only a name, and the executor
      may skip or require a runtime lookup.

    Geometry options (provide ONE of the following if observing):
      - (lat, lon, radius_km): a rough circular area around a point
      - polygon_geojson: an explicit polygon (e.g., region/belt)

    If neither is provided, the executor should treat this target as a high-level hint
    and may skip with a gap reason.
    """
    name: str = Field(..., description="Human-readable label (e.g., 'Port of LA', 'Ivory Coast cocoa belt').")
    lat: Optional[float] = Field(
        None,
        description="Latitude (WGS84) for a rough circular AOI; optional."
    )
    lon: Optional[float] = Field(
        None,
        description="Longitude (WGS84) for a rough circular AOI; optional."
    )
    radius_km: Optional[float] = Field(
        5.0,
        description="Radius (km) for the circular AOI around (lat,lon); optional; defaults to 5 km."
    )
    polygon_geojson: Optional[Dict] = Field(
        None,
        description="Alternative to (lat,lon): a GeoJSON polygon dict for precise AOI; optional."
    )
    sensors: List[SensorSpec] = Field(
        ...,
        description="For this target, which sensors/features to compute."
    )
    reason: str = Field(
        ...,
        description="One-sentence rationale for observing this target (transparency & auditability)."
    )


class ObservationPlan(BaseModel):
    """
    The planner's decision about whether satellite is useful, and if so, what/where to observe.

    Typical flow:
      - If the industry is 'pure software/internet', set use_satellite=False and leave targets empty.
      - Otherwise, propose up to ~2 high-value targets with feasible FREE signals
        (>=10 m resolution) and minimal features.
    """
    ticker: str = Field(..., description="Ticker or instrument identifier (e.g., 'TSLA', 'CC=F').")
    industry: Optional[str] = Field(None, description="Optional industry/category hint for routing.")
    use_satellite: bool = Field(
        ...,
        description="Planner's decision: use satellite for this ticker right now?"
    )
    targets: List[Target] = Field(
        default_factory=list,
        description="List of targets to observe. May be empty if use_satellite=False."
    )
    fallbacks: List[Target] = Field(
        default_factory=list,
        description="Optional backup targets if primary ones are unobservable (clouds, no recent pass)."
    )
    notes: str = Field(
        "",
        description="Free-text planner notes (e.g., why skipped; assumptions; safety remarks)."
    )


# -----------------------------------------------------------------------------
# OBSERVE (code → data): What did we measure?
# -----------------------------------------------------------------------------

class Observation(BaseModel):
    """
    One computed metric for a single target and sensor.

    This is a small, numeric, reproducible value that downstream LLMs can reason over,
    accompanied by quality and provenance for auditability.
    """
    target: str = Field(..., description="Name of the target, copied from the plan.")
    sensor: SensorType = Field(..., description="Which data source produced this measurement.")
    metric: str = Field(
        ...,
        description="Feature name (e.g., 'NDVI_mean_30d_vs_prev30d')."
    )
    value: Optional[float] = Field(
        None,
        description="Computed numeric value (e.g., percent change). None if not computable."
    )
    quality: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="0..1 summary quality score (e.g., valid pixel ratio × recency factor)."
    )
    as_of: str = Field(
        ...,
        description="ISO timestamp of the most recent scene used in this measurement."
    )
    provenance: Dict = Field(
        default_factory=dict,
        description=(
            "Opaque metadata to reproduce/debug: scene IDs, bbox/polygon bounds, "
            "cloud coverage, sample counts, API versions, etc."
        )
    )
    note: str = Field(
        "",
        description="Short human-readable note (e.g., 'cloudy day; used prior pass')."
    )


class ObservationResult(BaseModel):
    """
    The executor's full result for a plan: a bag of Observations + gap reasons.

    - observations: zero or more successfully computed metrics
    - gaps: text reasons for missing/low-quality data (e.g., 'clouds', 'no recent pass')
    - summary_notes: optional free-text glue from the executor
    """
    ticker: str = Field(..., description="Propagated from plan.")
    observations: List[Observation] = Field(default_factory=list, description="Computed features.")
    gaps: List[str] = Field(default_factory=list, description="Reasons for missing/failed observations.")
    summary_notes: str = Field("", description="Optional executor notes.")


# -----------------------------------------------------------------------------
# EXPLAIN (LLM → human): How do we tell the summary the satellite results?
# -----------------------------------------------------------------------------

class SatelliteSummary(BaseModel):
    """
    A compact, human-readable summary ready to slot into the final research report.

    - headline: single sentence capturing the satellite take-away
    - bullets: 2–4 short points with the strongest evidence
    - confidence: overall 0..1 confidence based on quality/coverage/consistency
    - attribution: which sources contributed (e.g., ['S2','VIIRS'])
    - raw_counts: small integers for transparency (e.g., number of observations, gaps)
    """
    ticker: str = Field(..., description="Propagated identifier for the ticker/instrument.")
    headline: str = Field(..., description="One-line summary of the satellite signal.")
    bullets: List[str] = Field(..., description="2–4 concise supporting points.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence 0..1.")
    attribution: List[str] = Field(default_factory=list, description="Data sources used, e.g., ['S2'].")
    raw_counts: Dict[str, int] = Field(default_factory=dict, description="Tiny transparency stats.")
