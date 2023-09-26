import sys
import os

sys.path.insert(0, os.path.abspath("../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Galactic"
copyright = "Taylor AI, Inc., 2023"
author = "Benjamin Anderson & Brian Kim"

favicons = [
    "logo-dark.ico",
    "logo-light.png",
]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Automatic documentation from docstrings
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
    "sphinx_copybutton",  # Add copy button to code blocks
    "sphinx_favicon",  # Add favicon
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autodoc_typehints = "both"
autodoc_typehints_format = "fully-qualified"

highlight_language = "python"


html_theme_options = {
    "show_navbar_depth": 2,
    "repository_url": "https://github.com/taylorai/galactic",
    "use_issues_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/taylorai/galactic",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Slack",
            "url": "https://galactic-ai.slack.com/",
            "icon": "fab fa-slack",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/TryTaylor_AI",
            "icon": "fab fa-twitter",
        },
        {
            "name": "LinkedIn",
            "url": "https://www.linkedin.com/company/taylor-ai/",
            "icon": "fab fa-linkedin",
        },
        {
            "name": "ProductHunt",
            "url": "https://www.producthunt.com/posts/taylor-ai",
            "icon": "fab fa-product-hunt",
        },
    ],
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
