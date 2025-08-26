"""Prompt Improver Package.

Advanced prompt improvement system with ML-powered optimization and analytics.
"""

# CRITICAL: Configure beartype before any other imports to prevent NumPy contamination
import os

os.environ['BEARTYPE_DISABLE_NUMPY'] = '1'

__version__ = "2025.1.0"
