"""Tests for backend/services/analytics.py."""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

from services.analytics import (
    analyze_delta,
    calculate_delta_matrix,
    generate_delta_diagnostic_plots,
    suppress_background_noise,
    to_native,
)


# ---------------------------------------------------------------------------
# to_native
# ---------------------------------------------------------------------------

def test_to_native_scalar_and_array():
    assert to_native(np.float64(3.5)) == 3.5
    assert to_native(np.int64(7)) == 7
    assert to_native(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert to_native({"a": np.float32(1.5), "b": np.array([1, 2])}) == {
        "a": 1.5,
        "b": [1, 2],
    }
    assert to_native((np.float64(1.0), [np.int32(2)])) == [1.0, [2]]
    assert to_native({"nested": {"x": np.float64(0.1)}}) == {"nested": {"x": 0.1}}


def test_to_native_passthrough():
    assert to_native("hello") == "hello"
    assert to_native(42) == 42
    assert to_native(None) is None


def test_to_native_zero_dimensional_array():
    result = to_native(np.array(5.0))
    assert result == 5.0
    assert isinstance(result, float)
    assert not isinstance(result, np.ndarray)
    assert not isinstance(result, np.generic)


# ---------------------------------------------------------------------------
# calculate_delta_matrix
# ---------------------------------------------------------------------------

def test_delta_normal():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[4.0, 2.0], [0.0, 4.0]])
    result = calculate_delta_matrix(a, b)
    expected = np.array([[3.0, 0.0], [3.0, 0.0]])
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float64


def test_delta_mismatched_shapes():
    a = np.zeros((2, 2))
    b = np.zeros((2, 3))
    with pytest.raises(ValueError, match="shapes must match"):
        calculate_delta_matrix(a, b)


def test_delta_non_2d():
    with pytest.raises(ValueError, match="two-dimensional"):
        calculate_delta_matrix(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match="two-dimensional"):
        calculate_delta_matrix(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))


def test_delta_nan_and_inf():
    a = np.array([[np.nan, np.inf], [-np.inf, 5.0]])
    b = np.array([[1.0, 1.0], [1.0, 1.0]])
    result = calculate_delta_matrix(a, b)
    # Invalid values replaced with 0.0 before differencing
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
    expected = np.array([[1.0, 1.0], [1.0, 4.0]])
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# suppress_background_noise
# ---------------------------------------------------------------------------

def test_suppression_normal():
    delta = np.array([[0.0, 10.0], [1.0, 2.0]])
    filtered, mask, stats = suppress_background_noise(delta, sigma=3.0)
    assert filtered.shape == delta.shape
    assert mask.shape == delta.shape
    assert mask.dtype == bool
    assert isinstance(stats, dict)
    assert "threshold" in stats and "has_anomalies" in stats


def test_suppression_constant_no_anomalies():
    delta = np.full((3, 3), 5.0)
    filtered, mask, stats = suppress_background_noise(delta, sigma=3.0)
    assert not np.any(mask)
    assert stats["has_anomalies"] is False
    assert stats["anomaly_pixel_count"] == 0
    np.testing.assert_array_equal(filtered, np.zeros_like(delta))


def test_suppression_sigma_validation():
    delta = np.zeros((2, 2))
    with pytest.raises(ValueError, match="finite and non-negative"):
        suppress_background_noise(delta, sigma=float("nan"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        suppress_background_noise(delta, sigma=-1.0)


def test_suppression_empty_matrix():
    delta = np.zeros((0, 0))
    filtered, mask, stats = suppress_background_noise(delta, sigma=3.0)
    assert filtered.shape == (0, 0)
    assert mask.shape == (0, 0)
    assert stats["total_pixel_count"] == 0
    assert stats["has_anomalies"] is False


def test_suppression_strict_greater_threshold():
    # delta equal to threshold should NOT be flagged (> not >=)
    delta = np.array([[0.0, 10.0]])
    _, mask, stats = suppress_background_noise(delta, sigma=0.0)
    # sigma=0 => threshold == mean; only values strictly above mean flagged
    assert stats["threshold"] == stats["background_mean"]
    # value equal to mean -> not flagged
    assert not mask[0, 0]


def test_suppression_filtered_matrix_known_case():
    # One large spike over a near-zero background should be preserved.
    delta = np.zeros((10, 10))
    delta[5, 5] = 100.0
    filtered, mask, stats = suppress_background_noise(delta, sigma=3.0)
    assert stats["has_anomalies"] is True
    assert stats["anomaly_pixel_count"] == 1
    assert mask[5, 5]
    # Only the spike pixel retains its value; all others are zeroed.
    np.testing.assert_array_equal(filtered, np.where(mask, delta, 0.0))
    assert filtered[5, 5] == 100.0


def test_suppression_pandas_filtering_expected_output():
    # Verify Pandas DataFrame filtering produces exact expected values.
    delta = np.array([[0.0, 10.0, 2.0], [5.0, 1.0, 20.0], [0.0, 3.0, 1.0]])
    filtered, mask, stats = suppress_background_noise(delta, sigma=1.0)
    # mean=6.22..., std=6.0..., threshold=mean+std≈12.2; pixels > threshold: 20.0 only
    expected_mask = np.array([[False, False, False], [False, False, True], [False, False, False]])
    np.testing.assert_array_equal(mask, expected_mask)
    expected_filtered = np.where(mask, delta, 0.0)
    np.testing.assert_array_equal(filtered, expected_filtered)
    assert mask.dtype == bool
    assert stats["anomaly_pixel_count"] == 1
    assert stats["has_anomalies"] is True


def test_suppression_pandas_constant_matrix_no_anomalies():
    # Constant matrix edge case through Pandas-based filtering.
    delta = np.full((4, 4), 7.0)
    filtered, mask, stats = suppress_background_noise(delta, sigma=3.0)
    assert stats["background_mean"] == 7.0
    assert stats["background_std"] == 0.0
    assert stats["threshold"] == 7.0
    assert not np.any(mask)
    assert stats["anomaly_pixel_count"] == 0
    assert stats["has_anomalies"] is False
    np.testing.assert_array_equal(filtered, np.zeros_like(delta))


def test_suppression_mask_bool_dtype_and_strict_threshold():
    delta = np.array([[0.0, 10.0]])
    _, mask, stats = suppress_background_noise(delta, sigma=1.0)
    # mean=5, std=5, threshold=10; strict > means 10 is NOT flagged.
    assert mask.dtype == bool
    assert stats["threshold"] == 10.0
    assert not np.any(mask)


def test_suppression_non_2d():
    with pytest.raises(ValueError, match="two-dimensional"):
        suppress_background_noise(np.zeros(5))


# ---------------------------------------------------------------------------
# generate_delta_diagnostic_plots
# ---------------------------------------------------------------------------

def test_plots_histogram_and_heatmap():
    delta = np.random.default_rng(0).random((10, 10))
    plots = generate_delta_diagnostic_plots(delta)
    assert "histogram_b64" in plots and "heatmap_b64" in plots
    assert isinstance(plots["histogram_b64"], str) and len(plots["histogram_b64"]) > 0
    assert isinstance(plots["heatmap_b64"], str) and len(plots["heatmap_b64"]) > 0


def test_plots_no_heatmap():
    delta = np.zeros((4, 4))
    plots = generate_delta_diagnostic_plots(delta, include_heatmap=False)
    assert plots["heatmap_b64"] is None
    assert isinstance(plots["histogram_b64"], str)


def test_plots_empty_matrix():
    delta = np.zeros((0, 0))
    plots = generate_delta_diagnostic_plots(delta)
    assert isinstance(plots["histogram_b64"], str)
    assert isinstance(plots["heatmap_b64"], str)


def _decode_png(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw))


def test_plots_base64_decodes_to_valid_png():
    delta = np.random.default_rng(0).random((8, 8))
    plots = generate_delta_diagnostic_plots(delta)
    hist_img = _decode_png(plots["histogram_b64"])
    assert hist_img.format == "PNG"
    heat_img = _decode_png(plots["heatmap_b64"])
    assert heat_img.format == "PNG"


def test_plots_heatmap_output():
    delta = np.random.default_rng(1).random((6, 6))
    mask = np.zeros((6, 6), dtype=bool)
    mask[2, 2] = True
    plots = generate_delta_diagnostic_plots(delta, anomaly_mask=mask, include_heatmap=True)
    assert isinstance(plots["heatmap_b64"], str) and len(plots["heatmap_b64"]) > 0
    img = _decode_png(plots["heatmap_b64"])
    assert img.format == "PNG"


def test_plots_non_2d():
    with pytest.raises(ValueError, match="two-dimensional"):
        generate_delta_diagnostic_plots(np.zeros(5))


# ---------------------------------------------------------------------------
# analyze_delta pipeline + JSON serialization
# ---------------------------------------------------------------------------

def test_analyze_delta_json_serializable():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[1.0, 2.0], [3.0, 9.0]])
    result = analyze_delta(a, b, sigma=3.0, include_plots=True)
    assert set(result.keys()) == {
        "delta_matrix",
        "filtered_matrix",
        "anomaly_mask",
        "stats",
        "plots",
    }
    # Must round-trip through JSON.
    json.dumps(result)
    assert isinstance(result["stats"]["has_anomalies"], bool)
    assert isinstance(result["plots"]["histogram_b64"], str)


def test_analyze_delta_no_plots():
    a = np.ones((3, 3))
    b = np.ones((3, 3))
    result = analyze_delta(a, b, include_plots=False)
    assert result["plots"] == {}
    json.dumps(result)


def test_analyze_delta_nan_and_inf():
    a = np.array([[np.nan, np.inf], [-np.inf, 5.0]])
    b = np.array([[1.0, 1.0], [1.0, 1.0]])
    result = analyze_delta(a, b, sigma=3.0, include_plots=False)
    # No crash; NaN/inf replaced with 0.0 before differencing
    assert set(result.keys()) == {
        "delta_matrix",
        "filtered_matrix",
        "anomaly_mask",
        "stats",
        "plots",
    }
    # Statistics are finite numeric values
    assert np.isfinite(result["stats"]["background_mean"])
    assert np.isfinite(result["stats"]["background_std"])
    assert np.isfinite(result["stats"]["threshold"])
    # Result round-trips through JSON
    json.dumps(result)
