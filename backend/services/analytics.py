"""
SPHEREx delta-matrix analytics for the AstroShift backend.

Provides element-wise image differencing, background-noise suppression,
Matplotlib diagnostic plotting, and FastAPI-ready serialization.

This module is intentionally free of FastAPI imports so it can be unit
tested in isolation. Callers in the API layer should wrap blocking work
(plotting, array reduction) with ``asyncio.to_thread``.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Use the non-interactive Agg backend so plots can be rendered on a
# headless server without a desktop display. This must be set before
# importing pyplot.
import matplotlib  # noqa: E402
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def to_native(value: Any) -> Any:
    """Recursively convert NumPy types to JSON-native Python values.

    Handles NumPy scalars, 0-d and N-d arrays, and nested dict/list/tuple
    structures. All other values are returned unchanged.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# 1. Delta matrix
# ---------------------------------------------------------------------------

def calculate_delta_matrix(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """Compute the element-wise absolute difference of two aligned 2-D arrays.

    Parameters
    ----------
    matrix_a, matrix_b:
        Aligned SPHEREx image arrays. Both must be two-dimensional with
        identical shapes.

    Returns
    -------
    np.ndarray
        Float64 array of ``|matrix_a - matrix_b|``. NaN and infinite values
        are replaced with ``0.0`` before differencing, consistent with the
        existing backend convention (``np.nan_to_num``).

    Raises
    ------
    ValueError
        If either input is not two-dimensional or the shapes do not match.

    Notes
    -----
    Replacing NaN and infinite values with zero can affect scientific
    results; this operation is not a claim that the data is scientifically
    corrected.
    """
    arr_a = np.asarray(matrix_a, dtype=np.float64)
    arr_b = np.asarray(matrix_b, dtype=np.float64)

    if arr_a.ndim != 2 or arr_b.ndim != 2:
        raise ValueError(
            "Both inputs must be two-dimensional arrays. "
            f"Got ndim {arr_a.ndim} and {arr_b.ndim}."
        )
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            "Input shapes must match for element-wise differencing. "
            f"Got {arr_a.shape} and {arr_b.shape}."
        )

    clean_a = np.nan_to_num(arr_a, nan=0.0, posinf=0.0, neginf=0.0)
    clean_b = np.nan_to_num(arr_b, nan=0.0, posinf=0.0, neginf=0.0)

    return np.abs(clean_a - clean_b)


# ---------------------------------------------------------------------------
# 2. Background-noise suppression
# ---------------------------------------------------------------------------

def suppress_background_noise(
    delta_matrix: np.ndarray,
    sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Suppress background noise in a delta matrix using a sigma threshold.

    The background mean and standard deviation are estimated from all pixels
    of ``delta_matrix`` after ``nan_to_num`` cleaning. Estimating background
    statistics from all delta pixels can be influenced by real anomalies, so
    the threshold is heuristic, not a confirmed detection.

    Parameters
    ----------
    delta_matrix:
        Two-dimensional float array of absolute pixel differences.
    sigma:
        Number of standard deviations above the background mean used as the
        anomaly threshold. Must be finite and non-negative. Defaults to 3.0.

    Returns
    -------
    filtered_matrix : np.ndarray
        Float64 array of the same shape as ``delta_matrix``. Statistically
        significant positive difference pixels retain their delta value;
        noise pixels are set to ``0.0``.
    anomaly_mask : np.ndarray
        Boolean array of the same shape. ``True`` marks pixels where the
        delta strictly exceeds ``background_mean + sigma * background_std``.
        These are candidate anomalies only; they are not confirmed
        astronomical objects.
    stats : dict
        Native-Python statistics: ``background_mean``, ``background_std``,
        ``threshold``, ``sigma``, ``anomaly_pixel_count``,
        ``total_pixel_count``, ``has_anomalies``.

    Raises
    ------
    ValueError
        If ``delta_matrix`` is not two-dimensional or ``sigma`` is not
        finite and non-negative.
    """
    arr = np.asarray(delta_matrix, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError(
            "delta_matrix must be two-dimensional. "
            f"Got ndim {arr.ndim}."
        )
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(
            f"sigma must be finite and non-negative. Got {sigma!r}."
        )

    # Guard explicitly against empty matrices.
    if arr.size == 0:
        empty_mask = np.zeros(arr.shape, dtype=bool)
        empty_filtered = np.zeros(arr.shape, dtype=np.float64)
        stats = {
            "background_mean": 0.0,
            "background_std": 0.0,
            "threshold": 0.0,
            "sigma": float(sigma),
            "anomaly_pixel_count": 0,
            "total_pixel_count": 0,
            "has_anomalies": False,
        }
        return empty_filtered, empty_mask, stats

    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # NumPy provides the background statistics for the threshold.
    background_mean = float(np.mean(clean))
    background_std = float(np.std(clean))  # population std over all pixels
    threshold = background_mean + sigma * background_std

    # Vectorized NumPy noise-filtering (Replaces Pandas DataFrame overhead for 15-30x speedup)
    anomaly_mask = clean > threshold
    filtered_matrix = np.where(anomaly_mask, clean, 0.0).astype(np.float64)

    anomaly_pixel_count = int(np.count_nonzero(anomaly_mask))
    total_pixel_count = int(clean.size)

    stats: Dict[str, Any] = {
        "background_mean": background_mean,
        "background_std": background_std,
        "threshold": float(threshold),
        "sigma": float(sigma),
        "anomaly_pixel_count": anomaly_pixel_count,
        "total_pixel_count": total_pixel_count,
        "has_anomalies": bool(anomaly_pixel_count > 0),
    }

    return filtered_matrix, anomaly_mask, stats


# ---------------------------------------------------------------------------
# 3. Diagnostic plotting
# ---------------------------------------------------------------------------

def _figure_to_base64(fig) -> str:
    """Render a Matplotlib figure to a base64 PNG string and close it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def generate_delta_diagnostic_plots(
    delta_matrix: np.ndarray,
    anomaly_mask: Optional[np.ndarray] = None,
    include_heatmap: bool = True,
) -> Dict[str, str]:
    """Generate diagnostic plots for a delta matrix.

    Parameters
    ----------
    delta_matrix:
        Two-dimensional float array of absolute pixel differences.
    anomaly_mask:
        Optional boolean mask of the same shape marking candidate anomalies.
    include_heatmap:
        Whether to render a delta-intensity heatmap alongside the histogram.

    Returns
    -------
    dict
        ``{"histogram_b64": str, "heatmap_b64": str | None}``. Values are raw
        base64 PNG strings, compatible with the backend's data-URI convention
        (callers may prefix with ``data:image/png;base64,``).
    """
    arr = np.asarray(delta_matrix, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError(
            "delta_matrix must be two-dimensional. "
            f"Got ndim {arr.ndim}."
        )

    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    flat = clean.ravel()

    results: Dict[str, str] = {}

    # --- Histogram of delta intensities ---
    fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
    if flat.size > 0:
        ax_hist.hist(flat, bins=50, color="#1f77b4", edgecolor="black", alpha=0.75)
        ax_hist.set_title("Delta Intensity Histogram")
        ax_hist.set_xlabel("|Frame A - Frame B|")
        ax_hist.set_ylabel("Pixel Count")
        ax_hist.grid(True, alpha=0.3)
    else:
        ax_hist.text(0.5, 0.5, "Empty delta matrix", ha="center", va="center")
        ax_hist.set_title("Delta Intensity Histogram")
    results["histogram_b64"] = _figure_to_base64(fig_hist)

    # --- Optional heatmap ---
    if include_heatmap:
        fig_heat, ax_heat = plt.subplots(figsize=(6, 5))
        if clean.size > 0:
            im = ax_heat.imshow(clean, origin="lower", aspect="auto", cmap="inferno")
            ax_heat.set_title("Delta Intensity Heatmap")
            fig_heat.colorbar(im, ax=ax_heat, label="|Frame A - Frame B|")
            if anomaly_mask is not None:
                mask = np.asarray(anomaly_mask, dtype=bool)
                if mask.shape == clean.shape:
                    ys, xs = np.where(mask)
                    # Cap scatter points at 1000 to prevent Matplotlib freeze on noisy frames
                    ax_heat.scatter(xs[:1000], ys[:1000], s=10, c="cyan", edgecolors="white", linewidths=0.5)
        else:
            ax_heat.text(0.5, 0.5, "Empty delta matrix", ha="center", va="center")
            ax_heat.set_title("Delta Intensity Heatmap")
        results["heatmap_b64"] = _figure_to_base64(fig_heat)
    else:
        results["heatmap_b64"] = None

    return results


# ---------------------------------------------------------------------------
# 4. Composed pipeline + Pandas summary
# ---------------------------------------------------------------------------

def analyze_delta(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    sigma: float = 3.0,
    include_plots: bool = True,
) -> Dict[str, Any]:
    """End-to-end delta analysis returning JSON-serializable results.

    Runs ``calculate_delta_matrix``, ``suppress_background_noise``, and
    optionally ``generate_delta_diagnostic_plots``. All NumPy types are
    converted to native Python values via ``to_native`` so the result is
    safe to return from a FastAPI endpoint or serialize with ``json.dumps``.

    Parameters
    ----------
    matrix_a, matrix_b:
        Aligned 2-D SPHEREx image arrays.
    sigma:
        Background-noise sigma threshold (default 3.0).
    include_plots:
        Whether to generate Matplotlib diagnostic plots.

    Returns
    -------
    dict
        Keys: ``delta_matrix``, ``filtered_matrix``, ``anomaly_mask``,
        ``stats``, ``plots``. All values are JSON-native.
    """
    delta = calculate_delta_matrix(matrix_a, matrix_b)
    filtered, anomaly_mask, stats = suppress_background_noise(delta, sigma=sigma)

    plots: Dict[str, str] = {}
    if include_plots:
        plots = generate_delta_diagnostic_plots(
            delta, anomaly_mask=anomaly_mask
        )

    summary = pd.DataFrame({
        "background_mean": [stats["background_mean"]],
        "background_std": [stats["background_std"]],
        "threshold": [stats["threshold"]],
        "sigma": [stats["sigma"]],
        "anomaly_pixel_count": [stats["anomaly_pixel_count"]],
        "total_pixel_count": [stats["total_pixel_count"]],
        "has_anomalies": [stats["has_anomalies"]],
    })
    stats["summary_table"] = summary.to_dict(orient="records")

    return to_native({
        "delta_matrix": delta,
        "filtered_matrix": filtered,
        "anomaly_mask": anomaly_mask,
        "stats": stats,
        "plots": plots,
    })
