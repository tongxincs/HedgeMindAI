import numpy as np


def pct_change(curr: float, prev: float) -> float:
    """
    Compute the percent change between two scalar values.

    Parameters
    ----------
    curr : float
        Current value (e.g., mean NDVI in last 30 days).
    prev : float
        Previous value (e.g., mean NDVI in prior 30 days).

    Returns
    -------
    float
        Percent change = 100 * (curr - prev) / |prev|.
        Returns 0.0 if prev == 0 to avoid division by zero.
    """
    if prev == 0:
        return 0.0
    return 100.0 * (curr - prev) / abs(prev)


def ndvi(b08: np.ndarray, b04: np.ndarray) -> np.ndarray:
    """
    Compute NDVI (Normalized Difference Vegetation Index) from Sentinel-2 bands.

    NDVI = (NIR - Red) / (NIR + Red)

    Parameters
    ----------
    b08 : np.ndarray
        Near-Infrared band (Sentinel-2 B08).
    b04 : np.ndarray
        Red band (Sentinel-2 B04).

    Returns
    -------
    np.ndarray
        NDVI values in range [-1, 1] for each pixel.
    """
    return (b08 - b04) / (b08 + b04 + 1e-6)


def mean_over_mask(arr: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute the mean of array values restricted to a mask.

    Parameters
    ----------
    arr : np.ndarray
        Array of values (e.g., NDVI map).
    mask : np.ndarray
        Boolean mask (True = valid pixel, False = ignore).

    Returns
    -------
    float
        Mean value over the masked region. Returns NaN if no valid pixels.
    """
    m = arr[mask]
    if m.size == 0:
        return float("nan")
    return float(np.nanmean(m))


def quality_from_valid_ratio(valid_ratio: float, scene_age_days: float) -> float:
    """
    Combine valid-pixel coverage and recency into a quality score (0..1).

    Heuristic:
    - Weight 50% by valid pixel ratio (fraction of AOI not obscured by clouds/shadows).
    - Weight 50% by recency, scaled so that 0 days = 1.0, 14+ days = ~0.0.

    Parameters
    ----------
    valid_ratio : float
        Fraction (0..1) of pixels in the AOI that were valid (not cloudy).
    scene_age_days : float
        Age of the scene in days (0 = today, higher = older).

    Returns
    -------
    float
        Quality score in [0,1]. Higher is better (fresh, cloud-free).
    """
    q = 0.5 * valid_ratio + 0.5 * max(0.0, 1.0 - scene_age_days / 14.0)
    return max(0.0, min(1.0, q))
