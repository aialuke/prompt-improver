"""Sphinx configuration for APES documentation.

Following 2025 best practices for automated documentation generation.
"""

import os
import sys
from pathlib import Path

# Add the source directory to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# Project information
project = "APES - Adaptive Prompt Enhancement System"
copyright = "2025, APES Development Team"
author = "APES Development Team"
version = "2025.1.0"
release = "2025.1.0"

# Extensions for 2025 documentation standards
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# AutoDoc configuration for 2025
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Type hints configuration
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True
typehints_use_signature = True
typehints_use_signature_return = True

# Autosummary configuration
autosummary_generate = True
autosummary_imported_members = True

# Templates path
templates_path = ["_templates"]

# Source file patterns
source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
}

# Master document
master_doc = "index"

# Language
language = "en"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML theme for 2025
html_theme = "furo"
html_title = f"{project} v{version}"

# HTML theme options
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#2563eb",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#60a5fa",
    },
}

# Static files
html_static_path = ["_static"]

# Custom CSS
html_css_files = [
    "custom.css",
]

# HTML options
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Intersphinx mapping for external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
    "fastapi": ("https://fastapi.tiangolo.com/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "asyncio": ("https://docs.python.org/3/library/asyncio.html", None),
}

# Todo extension
todo_include_todos = True

# Coverage extension
coverage_show_missing_items = True

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Custom roles and directives
rst_prolog = """
.. |project| replace:: APES
.. |version| replace:: 2025.1.0
"""

# Autodoc configuration for async functions
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Custom processing for async functions
def setup(app):
    """Sphinx setup function for custom processing."""
    app.add_css_file("custom.css")
