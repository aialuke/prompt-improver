"""Test anomaly_detectors attribute initialization"""

import pytest

from prompt_improver.learning.failure_analyzer import FailureConfig, FailureModeAnalyzer


def test_anomaly_detectors_attribute_exists_post_instantiation():
    """Test that anomaly_detectors attribute exists after instantiation"""
    # Test with robustness validation enabled (default)
    analyzer = FailureModeAnalyzer()

    # The anomaly_detectors attribute should exist
    assert hasattr(analyzer, "anomaly_detectors")
    assert isinstance(analyzer.anomaly_detectors, dict)


def test_anomaly_detectors_attribute_exists_when_robustness_disabled():
    """Test that anomaly_detectors attribute exists even when robustness validation is disabled"""
    # Test with robustness validation explicitly disabled
    config = FailureConfig(enable_robustness_validation=False)
    analyzer = FailureModeAnalyzer(config=config)

    # The anomaly_detectors attribute should still exist (initialized as empty dict)
    assert hasattr(analyzer, "anomaly_detectors")
    assert isinstance(analyzer.anomaly_detectors, dict)
    # When disabled, it should be empty
    assert analyzer.anomaly_detectors == {}


def test_anomaly_detectors_populated_when_robustness_enabled():
    """Test that anomaly_detectors is populated when robustness validation is enabled"""
    # Test with robustness validation enabled
    config = FailureConfig(enable_robustness_validation=True)
    analyzer = FailureModeAnalyzer(config=config)

    # The anomaly_detectors attribute should exist
    assert hasattr(analyzer, "anomaly_detectors")
    assert isinstance(analyzer.anomaly_detectors, dict)

    # Try to import the required libraries to see if they're available
    try:
        from sklearn.covariance import EllipticEnvelope
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM

        # If the libraries are available, anomaly_detectors should be populated
        assert len(analyzer.anomaly_detectors) > 0

        # Check that the expected detectors are present
        expected_detectors = ["isolation_forest", "elliptic_envelope", "one_class_svm"]
        for detector_name in expected_detectors:
            assert detector_name in analyzer.anomaly_detectors

    except ImportError:
        # If the libraries are not available, anomaly_detectors should be empty
        assert analyzer.anomaly_detectors == {}


def test_ensemble_detectors_and_anomaly_detectors_consistency():
    """Test that ensemble_detectors and anomaly_detectors reference the same object when robustness is enabled"""
    config = FailureConfig(enable_robustness_validation=True)
    analyzer = FailureModeAnalyzer(config=config)

    # Both attributes should exist
    assert hasattr(analyzer, "ensemble_detectors")
    assert hasattr(analyzer, "anomaly_detectors")

    try:
        from sklearn.ensemble import IsolationForest

        # If libraries are available, they should reference the same object
        assert analyzer.anomaly_detectors is analyzer.ensemble_detectors
    except ImportError:
        # If libraries are not available, both should be empty
        assert analyzer.anomaly_detectors == {}
        assert analyzer.ensemble_detectors == {}


def test_anomaly_detectors_initialization_default_config():
    """Test anomaly_detectors initialization with default configuration"""
    analyzer = FailureModeAnalyzer()

    # Check that the attribute exists
    assert hasattr(analyzer, "anomaly_detectors")

    # Check that it's a dictionary
    assert isinstance(analyzer.anomaly_detectors, dict)

    # Verify that the attribute is accessible and doesn't raise AttributeError
    detectors = analyzer.anomaly_detectors
    assert detectors is not None
