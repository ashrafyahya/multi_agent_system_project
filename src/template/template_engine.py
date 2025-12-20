"""PDF template engine for generating professional document structure.

This module implements the PDFTemplateEngine abstract base class and concrete
implementations for generating cover pages, headers, and footers for PDF documents.

Example:
    ```python
    from src.template.template_engine import DefaultPDFTemplateEngine
    from src.models.pdf_branding_config import PDFBrandingConfig
    from src.models.pdf_layout_config import PDFLayoutConfig
    
    branding = PDFBrandingConfig(company_name="Acme Corp")
    layout = PDFLayoutConfig(page_size="A4")
    
    engine = DefaultPDFTemplateEngine(
        branding_config=branding,
        layout_config=layout,
        report_title="Competitor Analysis Report"
    )
    
    cover_page = engine.create_cover_page()
    ```
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.models.pdf_branding_config import PDFBrandingConfig
from src.models.pdf_layout_config import PDFLayoutConfig
from src.template.cover_page import create_cover_page
from src.template.header_footer import create_footer, create_header

logger = logging.getLogger(__name__)


class PDFTemplateEngine(ABC):
    """Abstract base class for PDF template engines.
    
    This class follows the Template Method pattern, defining the algorithm
    structure for building PDF document elements while allowing subclasses
    to implement specific steps. This follows the Open/Closed Principle -
    the base class is closed for modification but open for extension.
    
    Attributes:
        branding_config: PDF branding configuration (optional)
        layout_config: PDF layout configuration (optional)
        report_title: Title of the report
        report_date: Date of the report (defaults to current date)
    
    Example:
        ```python
        class CustomTemplateEngine(PDFTemplateEngine):
            def create_cover_page(self) -> list[Any]:
                # Custom implementation
                return []
            
            def create_header(self, page_num: int, total_pages: int) -> Any:
                # Custom implementation
                return None
            
            def create_footer(self, page_num: int, total_pages: int) -> Any:
                # Custom implementation
                return None
        ```
    """
    
    def __init__(
        self,
        branding_config: PDFBrandingConfig | None = None,
        layout_config: PDFLayoutConfig | None = None,
        report_title: str = "Competitor Analysis Report",
        report_date: datetime | None = None,
    ) -> None:
        """Initialize template engine with configuration.
        
        Args:
            branding_config: Optional PDF branding configuration
            layout_config: Optional PDF layout configuration
            report_title: Title of the report
            report_date: Date of the report (defaults to current date if None)
        """
        self.branding_config = branding_config
        self.layout_config = layout_config
        self.report_title = report_title
        self.report_date = report_date if report_date else datetime.now()
    
    @abstractmethod
    def create_cover_page(self) -> list[Any]:
        """Create cover page flowables for PDF.
        
        This method should generate a list of ReportLab Flowable objects
        that represent the cover page, including logo, title, date, and
        other metadata.
        
        Returns:
            List of ReportLab Flowable objects for the cover page
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement create_cover_page")
    
    @abstractmethod
    def create_header(
        self, page_num: int, total_pages: int
    ) -> Any:
        """Create header flowable for PDF pages.
        
        This method should generate a header that appears on each page,
        typically including company name/logo and report title.
        
        Args:
            page_num: Current page number (1-indexed)
            total_pages: Total number of pages in document
            
        Returns:
            ReportLab Flowable object for the header, or None if no header
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement create_header")
    
    @abstractmethod
    def create_footer(
        self, page_num: int, total_pages: int
    ) -> Any:
        """Create footer flowable for PDF pages.
        
        This method should generate a footer that appears on each page,
        typically including page numbers, date, and optional footer text.
        
        Args:
            page_num: Current page number (1-indexed)
            total_pages: Total number of pages in document
            
        Returns:
            ReportLab Flowable object for the footer, or None if no footer
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement create_footer")
    
    def build_document_structure(
        self, report_content: str
    ) -> dict[str, list[Any]]:
        """Build complete document structure using template method pattern.
        
        This is the template method that defines the algorithm for building
        the document structure. It calls the abstract methods in the correct
        order and returns a dictionary with all document elements.
        
        Args:
            report_content: Markdown-formatted report content
            
        Returns:
            Dictionary containing:
                - "cover_page": List of cover page flowables
                - "header": Header flowable (or None)
                - "footer": Footer flowable (or None)
        """
        cover_page = self.create_cover_page()
        
        # Headers and footers are created per-page, so we return the factory methods
        # The actual header/footer creation will happen during PDF building
        return {
            "cover_page": cover_page,
            "header_factory": self.create_header,
            "footer_factory": self.create_footer,
        }


class DefaultPDFTemplateEngine(PDFTemplateEngine):
    """Default implementation of PDF template engine.
    
    This class provides a concrete implementation of PDFTemplateEngine with
    professional cover page, header, and footer generation.
    It supports multiple template styles and branding customization.
    
    This implementation delegates to specialized modules for each component,
    following the Single Responsibility Principle.
    
    Example:
        ```python
        from src.template.template_engine import DefaultPDFTemplateEngine
        from src.models.pdf_branding_config import PDFBrandingConfig
        
        branding = PDFBrandingConfig(
            company_name="Acme Corp",
            company_logo_path=Path("./logo.png"),
            cover_page_template="executive"
        )
        
        engine = DefaultPDFTemplateEngine(
            branding_config=branding,
            report_title="Competitor Analysis Report"
        )
        
        cover_page = engine.create_cover_page()
        ```
    """
    
    def create_cover_page(self) -> list[Any]:
        """Create cover page flowables for PDF.
        
        Delegates to the cover_page module for cover page generation.
        
        Returns:
            List of ReportLab Flowable objects for the cover page
        """
        return create_cover_page(
            branding_config=self.branding_config,
            layout_config=self.layout_config,
            report_title=self.report_title,
            report_date=self.report_date,
        )
    
    def create_header(
        self, page_num: int, total_pages: int
    ) -> Any:
        """Create header callback function for PDF pages.
        
        Delegates to the header_footer module for header generation.
        
        Args:
            page_num: Current page number (1-indexed)
            total_pages: Total number of pages in document
            
        Returns:
            Callable function(canvas, doc) for drawing header, or None if no header
        """
        return create_header(
            branding_config=self.branding_config,
            report_title=self.report_title,
            page_num=page_num,
            total_pages=total_pages,
        )
    
    def create_footer(
        self, page_num: int, total_pages: int
    ) -> Any:
        """Create footer callback function for PDF pages.
        
        Delegates to the header_footer module for footer generation.
        
        Args:
            page_num: Current page number (1-indexed)
            total_pages: Total number of pages in document
            
        Returns:
            Callable function(canvas, doc) for drawing footer, or None if no footer
        """
        return create_footer(
            branding_config=self.branding_config,
            report_date=self.report_date,
            page_num=page_num,
            total_pages=total_pages,
        )
