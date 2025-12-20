"""PDF template utilities for generating professional document structure.

This package contains utilities for PDF generation:
- TemplateEngine: Abstract base class and default implementation for PDF templates
"""

from src.template.template_engine import (
    DefaultPDFTemplateEngine,
    PDFTemplateEngine,
)

__all__ = [
    "PDFTemplateEngine",
    "DefaultPDFTemplateEngine",
]

