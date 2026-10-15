"""Severity §8.1 — D1 (Cf = observation confidence) and D4 (M-gate) regressions."""

import numpy as np

from ml.severity import (
    calculate_severity,
    gated_severity,
    is_critical,
    is_priority,
    observation_confidence,
    severity_level,
)


def test_formula_matches_srs(config):
    assert calculate_severity(0.5, 0.7, 0.9, config) == (0.6 * 0.5 + 0.4 * 0.7) * 0.9


def test_d1_severity_monotonic_in_change(config):
    """At fixed C and Cf, S must increase with M (it fell before: Cf was match confidence)."""
    values = [calculate_severity(m, 0.7, 0.9, config) for m in np.linspace(0, 1, 11)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_d1_severity_can_exceed_priority_threshold(config):
    s = calculate_severity(1.0, 0.7, 0.9, config)
    assert s > 0.5 and is_priority(s, config)


def test_d1_cf_is_min_of_detector_and_classifier():
    assert observation_confidence(0.8, 0.6) == 0.6
    assert observation_confidence(0.4, 0.95) == 0.4


def test_d4_gate_zeroes_insignificant_change(config):
    threshold = config["change_detection"]["threshold"]
    s, significant = gated_severity(threshold - 0.01, 0.9, 1.0, config)
    assert s == 0.0 and not significant
    s, significant = gated_severity(threshold, 0.9, 1.0, config)
    assert s > 0.0 and significant


def test_levels(config):
    assert severity_level(0.0, config) == "low"
    assert severity_level(0.4, config) == "medium"
    assert severity_level(0.6, config) == "priority"
    assert severity_level(0.85, config) == "critical"
    assert is_critical(calculate_severity(1.0, 0.9, 0.95, config), config)
    assert not is_priority(0.5, config)


def test_extent_separates_small_and_large_defects(config):
    """A confident crack at fixed C scores by size: hairline below priority, long open crack critical."""
    small = calculate_severity(1.0, 0.733, 1.0, config, extent=0.1)
    large = calculate_severity(1.0, 0.733, 1.0, config, extent=0.85)
    assert small < large
    assert severity_level(small, config) == "medium"
    assert is_critical(large, config)
    assert calculate_severity(1.0, 0.733, 1.0, config, extent=None) == calculate_severity(1.0, 0.733, 1.0, config)
    # the gate still uses the unscaled M: a small but certain new defect is not dropped to S = 0
    s, significant = gated_severity(1.0, 0.733, 1.0, config, extent=0.0)
    assert significant and s > 0
