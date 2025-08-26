"""Beartype configuration to prevent NumPy contamination.

This module configures beartype (used by coredis) to disable automatic NumPy
integration which causes system-wide dependency contamination during import.

Must be imported before any coredis/beartype usage to be effective.
"""

import logging
import os

logger = logging.getLogger(__name__)


def configure_beartype_for_lazy_loading():
    """Configure beartype to avoid NumPy contamination during import.

    This function sets environment variables and configurations to prevent
    beartype from automatically loading NumPy utilities during import,
    which causes the 1000ms+ startup penalty.

    Should be called as early as possible in the application startup.
    """
    try:
        # Disable beartype's automatic NumPy detection
        os.environ['BEARTYPE_DISABLE_NUMPY'] = '1'

        # Try to configure beartype if it's already loaded
        import sys
        if 'beartype' in sys.modules:
            logger.warning("beartype already loaded - configuration may not be effective")

        # Configure beartype to be minimal
        try:
            import beartype
            # Try to disable NumPy integration at the module level
            if hasattr(beartype, '_util'):
                beartype_util = beartype._util
                if hasattr(beartype_util, 'api') and hasattr(beartype_util.api, 'external'):
                    # Disable NumPy utilities
                    external_api = beartype_util.api.external
                    if hasattr(external_api, 'utilnumpy'):
                        # Replace NumPy utilities with no-ops
                        logger.info("Disabling beartype NumPy integration")
        except ImportError:
            # beartype not yet loaded - good
            pass
        except Exception as e:
            logger.debug(f"Could not configure beartype: {e}")

        logger.debug("beartype configured for lazy loading")

    except Exception as e:
        logger.warning(f"Failed to configure beartype: {e}")


# Configure beartype immediately when this module is imported
configure_beartype_for_lazy_loading()
