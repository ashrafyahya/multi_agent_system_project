"""Source quality classification enum.

This module defines the SourceQuality enum used to classify the reliability
and credibility of data sources in competitor analysis.
"""

from enum import Enum


class SourceQuality(str, Enum):
    """Source quality classification enum.
    
    Used to classify the reliability and credibility of data sources.
    Higher quality sources are more reliable and should be prioritized
    in analysis and reporting.
    
    Values:
        PRIMARY: Official sources (government, financial reports, scientific publications)
        SECONDARY_HIGH: Renowned market research institutes (Gartner, Forrester, IDC)
        SECONDARY_MEDIUM: Tech news sites, trade journals
        SECONDARY_LOW: Marketing blogs, company blogs
        COMMUNITY: Community discussions (Reddit, forums)
    """
    
    PRIMARY = "primary"
    SECONDARY_HIGH = "secondary_high"
    SECONDARY_MEDIUM = "secondary_medium"
    SECONDARY_LOW = "secondary_low"
    COMMUNITY = "community"
