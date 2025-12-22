"""Data collection helper functions.

This module contains utility functions for extracting and processing
competitor information from search results.
"""

import logging
import re
from typing import Any
from urllib.parse import urlparse

from src.utils.input_validator import validate_url

logger = logging.getLogger(__name__)


def extract_competitor_name(title: str, url: str, snippet: str) -> str | None:
    """Extract competitor name from title, URL, or snippet.
    
    Args:
        title: Page title
        url: Page URL
        snippet: Page snippet
    
    Returns:
        Competitor name string or None if extraction fails
    """
    # Try to extract from title first (most reliable)
    if title:
        # Remove common suffixes and prefixes
        name = re.sub(
            r'\s*-\s*(Home|Official|Website|About|Company|Inc|LLC|Ltd).*$',
            '',
            title,
            flags=re.IGNORECASE
        )
        name = re.sub(r'^(About|Welcome to|Home -)\s*', '', name, flags=re.IGNORECASE)
        name = name.strip()
        # More lenient: allow longer names and shorter minimum
        if name and len(name) >= 2 and len(name) < 150:
            # Clean up common patterns
            name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
            # Remove trailing punctuation
            name = name.rstrip('.,;:!?')
            if name:
                logger.debug(f"Extracted name from title: '{name}'")
                return name
    
    # Try to extract from URL domain
    if url:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain:
                # Remove www. and extract main domain part
                domain_parts = domain.replace("www.", "").split(".")
                if len(domain_parts) >= 2:
                    # Use the main domain name (second-to-last part)
                    name = (
                        domain_parts[-2]
                        if len(domain_parts) > 1
                        else domain_parts[0]
                    )
                    if name and len(name) >= 2:
                        return name.capitalize()
        except Exception:
            pass
    
    # Try to extract from snippet (look for company/product names)
    if snippet:
        # Look for capitalized words at the start (likely company names)
        words = snippet.split()
        # Take first 3-5 words if they start with capital letters
        potential_words = []
        for word in words[:6]:
            # Remove punctuation for checking
            clean_word = word.strip('.,;:!?()[]{}')
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                potential_words.append(clean_word)
            elif potential_words:  # Stop if we hit a non-capitalized word after finding some
                break
        
        if potential_words:
            potential_name = " ".join(potential_words[:3])  # Limit to 3 words
            if len(potential_name) >= 2 and len(potential_name) < 80:
                logger.debug(
                    f"Extracted name from snippet (capitalized): '{potential_name}'"
                )
                return potential_name
        
        # Fallback: use first few words (even if not all capitalized)
        words = snippet.split()[:4]
        # Clean up words
        clean_words = [
            w.strip('.,;:!?()[]{}')
            for w in words
            if w.strip('.,;:!?()[]{}')
        ]
        if clean_words:
            potential_name = " ".join(clean_words)
            if len(potential_name) >= 2 and len(potential_name) < 60:
                logger.debug(
                    f"Extracted name from snippet (fallback): '{potential_name}'"
                )
                return potential_name
    
    logger.debug("All name extraction methods failed")
    return None


def extract_website_url(url: str, snippet: str) -> str | None:
    """Extract competitor website URL.
    
    Args:
        url: Source URL
        snippet: Page snippet (unused, kept for API compatibility)
    
    Returns:
        Website URL string or None
    """
    # Use the source URL's domain as website
    try:
        parsed = urlparse(url)
        website = f"{parsed.scheme}://{parsed.netloc}"
        # Validate the constructed website URL
        is_valid, sanitized = validate_url(website, allow_localhost=False)
        if is_valid and sanitized:
            return sanitized
        return None
    except Exception:
        return None


def extract_products(snippet: str, title: str) -> list[str]:
    """Extract product names from snippet and title.
    
    Args:
        snippet: Page snippet
        title: Page title
    
    Returns:
        List of product names
    """
    products: list[str] = []
    text = f"{title} {snippet}".lower()
    
    # Look for product mentions
    product_patterns = [
        r'products?:\s*([^\.]+)',
        r'offers?\s+([^\.]+)',
        r'solutions?:\s*([^\.]+)',
    ]
    
    for pattern in product_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Split by common separators
            product_list = re.split(r'[,;]|\sand\s', match)
            for product in product_list:
                product = product.strip()
                if product and len(product) > 2 and len(product) < 100:
                    products.append(product)
            if products:
                break
        if products:
            break
    
    # Limit to 5 products
    return products[:5]


def extract_quantitative_metrics(
    snippet: str,
    title: str
) -> dict[str, Any]:
    """Extract quantitative metrics from snippet and title.
    
    Attempts to extract structured quantitative data from text using
    pattern matching. Extracts market share, revenue, user count,
    founded year, headquarters, and key features.
    
    Args:
        snippet: Page snippet text
        title: Page title
    
    Returns:
        Dictionary containing extracted metrics:
        - market_share: float | None
        - revenue: float | str | None
        - user_count: int | str | None
        - founded_year: int | None
        - headquarters: str | None
        - key_features: list[str]
    """
    metrics: dict[str, Any] = {
        "market_share": None,
        "revenue": None,
        "user_count": None,
        "founded_year": None,
        "headquarters": None,
        "key_features": []
    }
    
    text = f"{title} {snippet}".lower()
    full_text = f"{title} {snippet}"
    
    # Extract market share
    metrics["market_share"] = _extract_market_share(text)
    
    # Extract revenue
    metrics["revenue"] = _extract_revenue(text)
    
    # Extract user count
    metrics["user_count"] = _extract_user_count(text)
    
    # Extract founded year
    metrics["founded_year"] = _extract_founded_year(text)
    
    # Extract headquarters
    metrics["headquarters"] = _extract_headquarters(full_text)
    
    # Extract key features
    metrics["key_features"] = _extract_key_features(text)
    
    return metrics


def _extract_market_share(text: str) -> float | None:
    """Extract market share percentage from text.
    
    Args:
        text: Text to search
    
    Returns:
        Market share as float or None
    """
    market_share_patterns = [
        r'(\d+\.?\d*)\s*%\s*market\s*share',
        r'(\d+\.?\d*)\s*%\s*of\s*the\s*market',
        r'market\s*share\s*of\s*(\d+\.?\d*)\s*%',
    ]
    for pattern in market_share_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                share = float(match.group(1))
                if 0 <= share <= 100:
                    return share
            except (ValueError, TypeError):
                continue
    return None


def _extract_revenue(text: str) -> float | None:
    """Extract revenue from text.
    
    Args:
        text: Text to search
    
    Returns:
        Revenue as float (in dollars) or None
    """
    revenue_patterns = [
        r'\$(\d+\.?\d*)\s*([BMK]|billion|million|thousand)',
        r'revenue\s*(?:of|is|:)?\s*\$?(\d+\.?\d*)\s*([BMK]|billion|million)',
    ]
    for pattern in revenue_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                unit = match.group(2).upper()
                if unit in ['B', 'BILLION']:
                    return value * 1e9
                elif unit in ['M', 'MILLION']:
                    return value * 1e6
                elif unit in ['K', 'THOUSAND']:
                    return value * 1e3
                else:
                    return value
            except (ValueError, TypeError, IndexError):
                continue
    return None


def _extract_user_count(text: str) -> int | None:
    """Extract user count from text.
    
    Args:
        text: Text to search
    
    Returns:
        User count as int or None
    """
    user_count_patterns = [
        r'(\d+\.?\d*)\s*([BMK]|million|thousand)\s*(?:users?|customers?|subscribers?)',
        r'(?:users?|customers?|subscribers?):?\s*(\d+\.?\d*)\s*([BMK]|million|thousand)',
    ]
    for pattern in user_count_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                unit = match.group(2).upper()
                if unit in ['M', 'MILLION']:
                    return int(value * 1e6)
                elif unit in ['K', 'THOUSAND']:
                    return int(value * 1e3)
                elif unit in ['B', 'BILLION']:
                    return int(value * 1e9)
                else:
                    return int(value)
            except (ValueError, TypeError, IndexError):
                continue
    return None


def _extract_founded_year(text: str) -> int | None:
    """Extract founded year from text.
    
    Args:
        text: Text to search
    
    Returns:
        Founded year as int or None
    """
    founded_patterns = [
        r'founded\s+(?:in\s+)?(\d{4})',
        r'established\s+(?:in\s+)?(\d{4})',
        r'(\d{4})\s*\(founded\)',
    ]
    for pattern in founded_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                year = int(match.group(1))
                if 1800 <= year <= 2100:
                    return year
            except (ValueError, TypeError):
                continue
    return None


def _extract_headquarters(full_text: str) -> str | None:
    """Extract headquarters location from text.
    
    Args:
        full_text: Text to search (preserve case)
    
    Returns:
        Headquarters location string or None
    """
    hq_patterns = [
        r'headquartered\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'based\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'hq:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    ]
    for pattern in hq_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            hq = match.group(1).strip()
            if len(hq) > 2 and len(hq) < 100:
                return hq
    return None


def _extract_key_features(text: str) -> list[str]:
    """Extract key features from text.
    
    Args:
        text: Text to search
    
    Returns:
        List of key features
    """
    features: list[str] = []
    feature_patterns = [
        r'features?:\s*([^\.]+)',
        r'key\s+features?:\s*([^\.]+)',
        r'offers?\s+([^\.]+)',
    ]
    for pattern in feature_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Split by common separators
            feature_list = re.split(r'[,;]|\sand\s', match)
            for feature in feature_list:
                feature = feature.strip()
                if feature and len(feature) > 3 and len(feature) < 100:
                    features.append(feature)
            if features:
                break
        if features:
            break
    
    # Limit features
    return features[:10]

