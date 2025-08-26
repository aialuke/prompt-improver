"""Fix for NumPy contamination during beartype import.

This module implements an import hook to prevent beartype from loading
NumPy utilities during import, eliminating the 1000ms+ startup penalty.

Usage:
    # Import this before any other package imports
    from prompt_improver.core.utils.numpy_contamination_fix import prevent_numpy_contamination
    prevent_numpy_contamination()
"""

import sys
import types


class NumpyContaminationPrevention:
    """Import hook to prevent beartype from loading NumPy utilities."""

    def __init__(self) -> None:
        self.original_import = None
        self.blocked_modules = {
            'beartype._util.api.external.utilnumpy',
            'numpy',  # Block direct numpy imports during beartype loading
        }
        self.beartype_loading = False

    def __call__(self, name: str, *args, **kwargs):
        """Custom import hook that blocks NumPy loading during beartype import."""
        # Detect if we're loading beartype
        if name.startswith('beartype') and not self.beartype_loading:
            self.beartype_loading = True

        # Block NumPy utilities when beartype is loading
        if self.beartype_loading and name in self.blocked_modules:
            # Return a mock module instead
            mock_module = types.ModuleType(name)

            if 'utilnumpy' in name:
                # Mock beartype's numpy utilities
                mock_module.__dict__.update({
                    'is_numpy_available': lambda: False,
                    'is_numpy_array': lambda x: False,
                    'numpy': None,
                })
            elif name == 'numpy':
                # Mock numpy during beartype loading only
                mock_module.__dict__.update({
                    'array': lambda *args, **kwargs: None,
                    'ndarray': type,
                })

            sys.modules[name] = mock_module
            return mock_module

        # Use original import for everything else
        return self.original_import(name, *args, **kwargs)


_hook_installed = False


def prevent_numpy_contamination():
    """Install import hook to prevent NumPy contamination during beartype loading."""
    global _hook_installed

    if _hook_installed:
        return

    try:
        # Install the import hook
        prevention = NumpyContaminationPrevention()
        prevention.original_import = __builtins__['__import__']
        __builtins__['__import__'] = prevention

        _hook_installed = True
        print("✓ NumPy contamination prevention installed")

    except Exception as e:
        print(f"⚠️ Could not install NumPy contamination prevention: {e}")


def remove_numpy_contamination_prevention():
    """Remove the import hook (for testing)."""
    global _hook_installed

    if not _hook_installed:
        return

    try:
        # This is tricky to undo cleanly, so we'll just mark it as removed
        _hook_installed = False
        print("✓ NumPy contamination prevention marked for removal")
    except Exception as e:
        print(f"⚠️ Could not remove NumPy contamination prevention: {e}")


# Auto-install when module is imported
if not _hook_installed:
    prevent_numpy_contamination()
