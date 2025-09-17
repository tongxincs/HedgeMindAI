import os, io, math
from datetime import datetime, timedelta, timezone
from typing import List
import requests
import numpy as np
from tifffile import imread as tiff_read
from shapely.geometry import shape
from .schemas import Observation, ObservationResult, ObservationPlan, Target
from .features import ndvi, pct_change, mean_over_mask, quality_from_valid_ratio
from dotenv import load_dotenv

AUTH_API = "https://services.sentinel-hub.com/oauth/token"
PROCESS_API = "https://services.sentinel-hub.com/api/v1/process"

S2_EVALSCRIPT_BANDS = """
//VERSION=3
function setup() {
  return {
    input: ["B04","B08","dataMask"],
    output: { bands: 3, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  return [s.B04, s.B08, s.dataMask];
}
"""

def _auth_token() -> str:
    load_dotenv()
    cid = os.getenv("SENTINELHUB_CLIENT_ID")
    cs  = os.getenv("SENTINELHUB_CLIENT_SECRET")
    if not cid or not cs:
        raise RuntimeError("Set SENTINELHUB_CLIENT_ID/SECRET in .env")
    r = requests.post(AUTH_API, data={
        "grant_type":"client_credentials","client_id":cid,"client_secret":cs
    }, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def _bbox_from_target(t: Target) -> List[float]:
    # Simple circle buffer around lat/lon -> bbox (in WGS84 degrees, coarse)
    if t.polygon_geojson:
        g = shape(t.polygon_geojson)
        return list(g.bounds)  # xmin,ymin,xmax,ymax
    if t.lat is None or t.lon is None:
        raise ValueError("Target requires (lat,lon) or polygon_geojson")
    # very rough degree per km at mid-lat (good enough for demo)
    dlat = (t.radius_km or 5.0) / 110.574
    dlon = (t.radius_km or 5.0) / (111.320*math.cos(math.radians(t.lat)))
    return [t.lon - dlon, t.lat - dlat, t.lon + dlon, t.lat + dlat]


def _fetch_s2_stack(token: str, bbox: List[float], start: datetime, end: datetime, width=768, height=None):
    # Returns list of (timestamp_iso, array[H,W,3]=[B04,B08,mask])
    # For simplicity, sample at 2 dates: (end) and (end-15d) → enough for 30d average proxy
    dates = [end, end - timedelta(days=15)]
    out = []
    for dt in dates:
        payload = {
            "input": {
                "bounds": {"bbox": bbox},
                "data": [{"type":"S2L2A","dataFilter":{
                    "timeRange":{"from":(dt - timedelta(days=7)).isoformat()+"Z","to":dt.isoformat()+"Z"},
                    "maxCloudCoverage": 40
                }}]
            },
            "output": {
                "width": width,
                "height": height,
                "responses":[{"identifier":"default","format":{"type":"image/tiff"}}]
            },
            "evalscript": S2_EVALSCRIPT_BANDS
        }
        r = requests.post(PROCESS_API, headers={"Authorization":f"Bearer {token}"}, json=payload, timeout=90)
        if r.status_code != 200:
            continue
        arr = tiff_read(io.BytesIO(r.content))  # (H,W,3): B04,B08,mask
        if arr.ndim==2: arr = arr[...,None]
        out.append((dt.replace(tzinfo=timezone.utc).isoformat(), arr.astype(np.float32)))
    return out  # possibly length 0..2

def compute_ndvi_change_for_target(t: Target) -> List[Observation]:
    token = _auth_token()
    bbox = _bbox_from_target(t)
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    # Fetch two samples within last 30d window
    stack_curr = _fetch_s2_stack(token, bbox, now - timedelta(days=0), now)
    stack_prev = _fetch_s2_stack(token, bbox, now - timedelta(days=30), now - timedelta(days=30))
    obs: List[Observation] = []

    def _ndvi_mean(arr):
        b04, b08, mask = arr[...,0], arr[...,1], arr[...,2]>0.5
        val = mean_over_mask(ndvi(b08,b04), mask)
        valid_ratio = float(np.mean(mask))
        return val, valid_ratio

    if stack_curr:
        curr_vals = []
        curr_valids = []
        for ts, arr in stack_curr:
            v, vr = _ndvi_mean(arr)
            if not np.isnan(v):
                curr_vals.append(v); curr_valids.append(vr)
        curr_mean = float(np.mean(curr_vals)) if curr_vals else float("nan")
        curr_valid = float(np.mean(curr_valids)) if curr_valids else 0.0
    else:
        curr_mean, curr_valid = float("nan"), 0.0

    if stack_prev:
        prev_vals = []
        prev_valids = []
        for ts, arr in stack_prev:
            v, vr = _ndvi_mean(arr)
            if not np.isnan(v):
                prev_vals.append(v); prev_valids.append(vr)
        prev_mean = float(np.mean(prev_vals)) if prev_vals else float("nan")
        prev_valid = float(np.mean(prev_valids)) if prev_valids else 0.0
    else:
        prev_mean, prev_valid = float("nan"), 0.0

    value = None
    if (not math.isnan(curr_mean)) and (not math.isnan(prev_mean)):
        value = round(pct_change(curr_mean, prev_mean), 2)

    quality = quality_from_valid_ratio(curr_valid, scene_age_days=0.0)
    obs.append(Observation(
        target=t.name, sensor="S2", metric="NDVI_mean_30d_vs_prev30d",
        value=value, quality=quality, as_of=now.isoformat(),
        provenance={"bbox":bbox, "samples_curr":len(stack_curr), "samples_prev":len(stack_prev)},
        note="S2 NDVI over buffered bbox; simple 2-sample proxy for 30d windows"
    ))
    return obs

def execute_plan(plan: ObservationPlan) -> ObservationResult:
    result = ObservationResult(ticker=plan.ticker)
    if not plan.use_satellite or not plan.targets:
        result.summary_notes = plan.notes or "Satellite not applicable"
        return result

    for t in plan.targets:
        # Only implement NDVI for now; safely ignore unknown metrics
        for s in t.sensors:
            for feat in s.features:
                if s.type=="S2" and feat=="NDVI_mean_30d_vs_prev30d":
                    result.observations += compute_ndvi_change_for_target(t)
                # TODO: add handlers for NDWI_ship_count_wow, SAR_VV_delta_30d, etc.

    if not result.observations:
        result.gaps.append("No usable scenes or features computed")
    return result
