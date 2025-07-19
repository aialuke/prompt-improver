"""NLTK Resource Manager

This module provides utilities for managing NLTK resources including
automatic downloading with fallback handling for production environments.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

import nltk


class NLTKResourceManager:
    """Manager for NLTK resources with automatic downloading and fallback handling.

    This class handles:
    - Automatic detection of missing NLTK resources
    - Safe downloading in production environments
    - Graceful fallback when resources can't be downloaded
    - Cache management for resource availability
    """

    def __init__(self, download_dir: str | None = None):
        """Initialize the NLTK resource manager.

        Args:
            download_dir: Custom directory for NLTK data. If None, uses NLTK default.
        """
        self.logger = logging.getLogger(__name__)
        self.download_dir = download_dir
        self._resource_cache: dict[str, bool] = {}

        # Set custom download directory if provided
        if download_dir:
            nltk.data.path.insert(0, download_dir)

        # Common required resources
        self.required_resources = {
            "punkt",  # Sentence tokenization
            "averaged_perceptron_tagger",  # POS tagging
            "maxent_ne_chunker",  # Named entity chunking
            "words",  # Word corpus
            "stopwords",  # Stop words
            "wordnet",  # WordNet corpus
            "vader_lexicon",  # Sentiment analysis
            "punkt_tab",  # Updated punkt tokenizer
        }

    def ensure_resources(self, resources: set[str] | None = None) -> dict[str, bool]:
        """Ensure required NLTK resources are available.

        Args:
            resources: Set of resource names to check. If None, uses default required resources.

        Returns:
            Dictionary mapping resource names to availability status
        """
        if resources is None:
            resources = self.required_resources

        results = {}

        for resource in resources:
            try:
                if self._is_resource_available(resource) or self._download_resource(
                    resource
                ):
                    results[resource] = True
                    self._resource_cache[resource] = True
                else:
                    results[resource] = False
                    self._resource_cache[resource] = False
                    self.logger.warning(
                        f"NLTK resource '{resource}' is not available and could not be downloaded"
                    )

            except Exception as e:
                self.logger.error(f"Error checking NLTK resource '{resource}': {e}")
                results[resource] = False
                self._resource_cache[resource] = False

        return results

    def _is_resource_available(self, resource: str) -> bool:
        """Check if an NLTK resource is available.

        Args:
            resource: Name of the NLTK resource

        Returns:
            True if resource is available, False otherwise
        """
        # Check cache first
        if resource in self._resource_cache:
            return self._resource_cache[resource]

        try:
            # Try to find the resource
            nltk.data.find(f"tokenizers/{resource}")
            return True
        except LookupError:
            try:
                # Try other common paths
                nltk.data.find(f"taggers/{resource}")
                return True
            except LookupError:
                try:
                    nltk.data.find(f"chunkers/{resource}")
                    return True
                except LookupError:
                    try:
                        nltk.data.find(f"corpora/{resource}")
                        return True
                    except LookupError:
                        return False

    def _download_resource(self, resource: str) -> bool:
        """Download an NLTK resource.

        Args:
            resource: Name of the NLTK resource to download

        Returns:
            True if download succeeded, False otherwise
        """
        try:
            self.logger.info(f"Downloading NLTK resource: {resource}")

            # Try to download with quiet mode
            success = nltk.download(resource, quiet=True, raise_on_error=False)

            if success:
                self.logger.info(f"Successfully downloaded NLTK resource: {resource}")
                return True
            self.logger.warning(f"Failed to download NLTK resource: {resource}")
            return False

        except Exception as e:
            self.logger.error(f"Error downloading NLTK resource '{resource}': {e}")
            return False

    def get_resource_status(self) -> dict[str, Any]:
        """Get status of all required resources.

        Returns:
            Dictionary with resource status information
        """
        status = {
            "available": [],
            "missing": [],
            "total_required": len(self.required_resources),
            "availability_rate": 0.0,
        }

        available_count = 0

        for resource in self.required_resources:
            if self._is_resource_available(resource):
                status["available"].append(resource)
                available_count += 1
            else:
                status["missing"].append(resource)

        status["availability_rate"] = available_count / len(self.required_resources)

        return status

    def setup_for_production(self) -> bool:
        """Setup NLTK resources for production environment.

        This method ensures all required resources are available and downloads
        them if necessary, with appropriate error handling for production use.

        Returns:
            True if setup succeeded, False if critical resources are missing
        """
        self.logger.info("Setting up NLTK resources for production")

        # First, try to ensure all resources
        results = self.ensure_resources()

        # Check if critical resources are available
        critical_resources = {"punkt", "averaged_perceptron_tagger"}
        critical_available = all(
            results.get(resource, False) for resource in critical_resources
        )

        if not critical_available:
            self.logger.error(
                "Critical NLTK resources are missing and could not be downloaded"
            )
            return False

        # Log status
        status = self.get_resource_status()
        self.logger.info(
            f"NLTK resource availability: {status['availability_rate']:.1%} "
            f"({len(status['available'])}/{status['total_required']})"
        )

        if status["missing"]:
            self.logger.warning(f"Missing NLTK resources: {status['missing']}")

        return True

    def safe_nltk_operation(
        self, operation_func, fallback_func=None, required_resources=None
    ):
        """Execute an NLTK operation with resource checking and fallback.

        Args:
            operation_func: Function to execute that uses NLTK resources
            fallback_func: Fallback function to use if resources are missing
            required_resources: Set of resources required for the operation

        Returns:
            Result of operation_func or fallback_func
        """
        if required_resources:
            resource_status = self.ensure_resources(required_resources)
            if not all(resource_status.values()):
                self.logger.warning(
                    f"Some NLTK resources are missing: {required_resources}"
                )
                if fallback_func:
                    self.logger.info("Using fallback operation")
                    return fallback_func()
                raise RuntimeError(
                    f"Required NLTK resources not available: {required_resources}"
                )

        try:
            return operation_func()
        except Exception as e:
            self.logger.error(f"NLTK operation failed: {e}")
            if fallback_func:
                self.logger.info("Using fallback operation due to error")
                return fallback_func()
            raise


# Global instance for convenience
_global_manager = None


def get_nltk_manager(download_dir: str | None = None) -> NLTKResourceManager:
    """Get the global NLTK resource manager instance.

    Args:
        download_dir: Custom directory for NLTK data

    Returns:
        Global NLTKResourceManager instance
    """
    global _global_manager

    if _global_manager is None:
        _global_manager = NLTKResourceManager(download_dir)

    return _global_manager


def ensure_nltk_resources(resources: set[str] | None = None) -> dict[str, bool]:
    """Convenience function to ensure NLTK resources are available.

    Args:
        resources: Set of resource names to check

    Returns:
        Dictionary mapping resource names to availability status
    """
    manager = get_nltk_manager()
    return manager.ensure_resources(resources)


def setup_nltk_for_production() -> bool:
    """Convenience function to setup NLTK for production use.

    Returns:
        True if setup succeeded, False otherwise
    """
    manager = get_nltk_manager()
    return manager.setup_for_production()
