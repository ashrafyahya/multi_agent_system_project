"""Data consistency validator for competitor data.

This module implements the DataConsistencyValidator that validates
quantitative data consistency across competitor profiles, checking for
logical inconsistencies, conflicting data points, and unrealistic values.

Example:
    ```python
    from src.graph.validators.data_consistency_validator import DataConsistencyValidator
    
    validator = DataConsistencyValidator()
    data = {
        "competitors": [
            {"name": "Comp1", "market_share": 35.0, "revenue": 2000000000},
            {"name": "Comp2", "market_share": 20.0, "revenue": 1500000000},
        ]
    }
    result = validator.validate(data)
    ```
"""

import logging
from typing import Any

from src.graph.validators.base_validator import BaseValidator, ValidationResult

logger = logging.getLogger(__name__)


class DataConsistencyValidator(BaseValidator):
    """Validator that checks data consistency across competitor profiles.
    
    This validator performs consistency checks on quantitative data:
    - Market share percentages should sum to reasonable ranges (70-120%)
    - Revenue figures should be consistent (no conflicting data for same competitor)
    - No unrealistic numbers (negative values, percentages > 100%)
    - Detect conflicting data points for the same competitor
    
    The validator returns warnings (not errors) for inconsistencies, as
    these are data quality issues that should be noted but don't prevent
    workflow continuation.
    
    Example:
        ```python
        validator = DataConsistencyValidator()
        data = {"competitors": [...]}
        result = validator.validate(data)
        if result.has_warnings():
            for warning in result.warnings:
                print(f"Warning: {warning}")
        ```
    """
    
    # Reasonable range for market share sum (allows for incomplete data)
    MIN_MARKET_SHARE_SUM = 70.0
    MAX_MARKET_SHARE_SUM = 120.0
    
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate data consistency across competitor profiles.
        
        Performs multiple consistency checks:
        1. Market share sum validation
        2. Revenue consistency checks
        3. Unrealistic value detection
        4. Conflicting data point detection
        
        Args:
            data: Dictionary containing competitor data. Expected structure:
                {
                    "competitors": [
                        {
                            "name": str,
                            "market_share": float | None,
                            "revenue": float | str | None,
                            "user_count": int | str | None,
                            ...
                        },
                        ...
                    ]
                }
        
        Returns:
            ValidationResult with warnings for any inconsistencies found.
            Validation always passes (is_valid=True) as inconsistencies
            are warnings, not blocking errors.
        """
        result = ValidationResult.success()
        
        if not isinstance(data, dict):
            result.add_error("Data must be a dictionary")
            return result
        
        competitors = data.get("competitors", [])
        if not isinstance(competitors, list):
            result.add_error("'competitors' field must be a list")
            return result
        
        if not competitors:
            # Empty list is valid, just return success
            return result
        
        # Perform consistency checks
        self._validate_market_share_sum(competitors, result)
        self._validate_revenue_consistency(competitors, result)
        self._validate_unrealistic_values(competitors, result)
        self._validate_conflicting_data(competitors, result)
        
        return result
    
    def _validate_market_share_sum(
        self,
        competitors: list[dict[str, Any]],
        result: ValidationResult
    ) -> None:
        """Validate that market share percentages sum to reasonable range.
        
        Checks if the sum of all market share percentages falls within
        a reasonable range (70-120%). Values outside this range may indicate
        incomplete data or data quality issues.
        
        Args:
            competitors: List of competitor dictionaries
            result: ValidationResult to add warnings to
        """
        market_shares: list[float] = []
        
        for comp in competitors:
            if not isinstance(comp, dict):
                continue
            
            market_share = comp.get("market_share")
            if market_share is not None:
                try:
                    share_value = float(market_share)
                    if share_value >= 0:  # Only include non-negative values
                        market_shares.append(share_value)
                except (ValueError, TypeError):
                    # Skip invalid market share values
                    continue
        
        if not market_shares:
            # No market share data available, skip check
            return
        
        total_share = sum(market_shares)
        
        if total_share < self.MIN_MARKET_SHARE_SUM:
            result.add_warning(
                f"Market share percentages sum to {total_share:.1f}%, "
                f"which is below the expected minimum ({self.MIN_MARKET_SHARE_SUM}%). "
                f"This may indicate incomplete competitor data or missing market players."
            )
        elif total_share > self.MAX_MARKET_SHARE_SUM:
            result.add_warning(
                f"Market share percentages sum to {total_share:.1f}%, "
                f"which exceeds the expected maximum ({self.MAX_MARKET_SHARE_SUM}%). "
                f"This may indicate overlapping market definitions or data inconsistencies."
            )
    
    def _validate_revenue_consistency(
        self,
        competitors: list[dict[str, Any]],
        result: ValidationResult
    ) -> None:
        """Check revenue figures for consistency across competitors.
        
        Validates that revenue figures are reasonable relative to each other
        and flags potential inconsistencies (e.g., competitor with higher
        market share but much lower revenue).
        
        Args:
            competitors: List of competitor dictionaries
            result: ValidationResult to add warnings to
        """
        revenue_data: list[tuple[str, float, float | None]] = []
        
        for comp in competitors:
            if not isinstance(comp, dict):
                continue
            
            name = comp.get("name", "Unknown")
            market_share = comp.get("market_share")
            revenue = comp.get("revenue")
            
            # Try to extract numeric revenue value
            revenue_value: float | None = None
            if revenue is not None:
                if isinstance(revenue, (int, float)):
                    revenue_value = float(revenue)
                elif isinstance(revenue, str):
                    # Try to extract number from string (e.g., "$1B" -> 1000000000)
                    revenue_value = self._parse_revenue_string(revenue)
            
            if market_share is not None and revenue_value is not None:
                try:
                    share_value = float(market_share)
                    revenue_data.append((name, share_value, revenue_value))
                except (ValueError, TypeError):
                    continue
        
        if len(revenue_data) < 2:
            # Need at least 2 competitors for comparison
            return
        
        # Check for revenue/market share inconsistencies
        # Sort by market share
        revenue_data_sorted = sorted(revenue_data, key=lambda x: x[1], reverse=True)
        
        for i in range(len(revenue_data_sorted) - 1):
            name1, share1, rev1 = revenue_data_sorted[i]
            name2, share2, rev2 = revenue_data_sorted[i + 1]
            
            # If competitor 1 has significantly higher market share but lower revenue,
            # this might indicate a data inconsistency
            if share1 > share2 * 1.5 and rev1 < rev2 * 0.5:
                result.add_warning(
                    f"Potential revenue inconsistency: {name1} has {share1:.1f}% "
                    f"market share but revenue (${rev1/1e9:.2f}B) is lower than "
                    f"{name2} ({share2:.1f}% share, ${rev2/1e9:.2f}B revenue). "
                    f"This may indicate data quality issues or different revenue definitions."
                )
    
    def _validate_unrealistic_values(
        self,
        competitors: list[dict[str, Any]],
        result: ValidationResult
    ) -> None:
        """Flag unrealistic numbers (negative values, percentages > 100%).
        
        Args:
            competitors: List of competitor dictionaries
            result: ValidationResult to add warnings to
        """
        for comp in competitors:
            if not isinstance(comp, dict):
                continue
            
            name = comp.get("name", "Unknown")
            
            # Check market share
            market_share = comp.get("market_share")
            if market_share is not None:
                try:
                    share_value = float(market_share)
                    if share_value < 0:
                        result.add_warning(
                            f"{name}: Market share cannot be negative ({share_value}%)"
                        )
                    elif share_value > 100:
                        result.add_warning(
                            f"{name}: Market share exceeds 100% ({share_value}%)"
                        )
                except (ValueError, TypeError):
                    continue
            
            # Check revenue (if numeric)
            revenue = comp.get("revenue")
            if revenue is not None and isinstance(revenue, (int, float)):
                if revenue < 0:
                    result.add_warning(
                        f"{name}: Revenue cannot be negative (${revenue:,.0f})"
                    )
            
            # Check user count (if numeric)
            user_count = comp.get("user_count")
            if user_count is not None and isinstance(user_count, int):
                if user_count < 0:
                    result.add_warning(
                        f"{name}: User count cannot be negative ({user_count:,})"
                    )
            
            # Check founded year
            founded_year = comp.get("founded_year")
            if founded_year is not None:
                try:
                    year = int(founded_year)
                    if year < 1800 or year > 2100:
                        result.add_warning(
                            f"{name}: Founded year seems unrealistic ({year})"
                        )
                except (ValueError, TypeError):
                    continue
    
    def _validate_conflicting_data(
        self,
        competitors: list[dict[str, Any]],
        result: ValidationResult
    ) -> None:
        """Detect conflicting data points for the same competitor.
        
        Checks if the same competitor appears multiple times with
        conflicting quantitative data (e.g., different market shares).
        
        Args:
            competitors: List of competitor dictionaries
            result: ValidationResult to add warnings to
        """
        competitor_data: dict[str, list[dict[str, Any]]] = {}
        
        # Group competitors by name (case-insensitive)
        for comp in competitors:
            if not isinstance(comp, dict):
                continue
            
            name = comp.get("name", "").strip().lower()
            if not name:
                continue
            
            if name not in competitor_data:
                competitor_data[name] = []
            competitor_data[name].append(comp)
        
        # Check for conflicts in grouped data
        for name_lower, comp_list in competitor_data.items():
            if len(comp_list) < 2:
                continue
            
            # Check for conflicting market shares
            market_shares = [
                comp.get("market_share")
                for comp in comp_list
                if comp.get("market_share") is not None
            ]
            
            if len(market_shares) >= 2:
                unique_shares = set(market_shares)
                if len(unique_shares) > 1:
                    name_display = comp_list[0].get("name", name_lower)
                    result.add_warning(
                        f"Conflicting market share data for {name_display}: "
                        f"found values {sorted(unique_shares)}. "
                        f"This may indicate duplicate entries or data from different sources."
                    )
            
            # Check for conflicting revenue
            revenues = [
                comp.get("revenue")
                for comp in comp_list
                if comp.get("revenue") is not None
            ]
            
            if len(revenues) >= 2:
                # Normalize revenue values for comparison
                revenue_values: list[float] = []
                for rev in revenues:
                    if isinstance(rev, (int, float)):
                        revenue_values.append(float(rev))
                    elif isinstance(rev, str):
                        parsed = self._parse_revenue_string(rev)
                        if parsed is not None:
                            revenue_values.append(parsed)
                
                if len(revenue_values) >= 2:
                    unique_revenues = set(revenue_values)
                    if len(unique_revenues) > 1:
                        name_display = comp_list[0].get("name", name_lower)
                        # Check if difference is significant (>20%)
                        revenue_list = sorted(revenue_values)
                        max_rev = revenue_list[-1]
                        min_rev = revenue_list[0]
                        if (max_rev - min_rev) / max_rev > 0.2:
                            result.add_warning(
                                f"Conflicting revenue data for {name_display}: "
                                f"found significantly different values. "
                                f"This may indicate duplicate entries or data from different time periods."
                            )
    
    def _parse_revenue_string(self, revenue_str: str) -> float | None:
        """Parse revenue string to numeric value.
        
        Attempts to extract numeric value from revenue strings like:
        - "$1B" -> 1000000000
        - "$500M" -> 500000000
        - "$1.5B" -> 1500000000
        - "1B-2B" -> 1500000000 (average)
        
        Args:
            revenue_str: Revenue string to parse
        
        Returns:
            Parsed revenue value in dollars, or None if parsing fails
        """
        if not isinstance(revenue_str, str):
            return None
        
        revenue_str = revenue_str.strip().upper()
        
        # Remove currency symbols and spaces
        revenue_str = revenue_str.replace("$", "").replace(",", "").strip()
        
        # Handle ranges (e.g., "1B-2B" -> use average)
        if "-" in revenue_str:
            parts = revenue_str.split("-")
            if len(parts) == 2:
                val1 = self._parse_single_revenue(parts[0].strip())
                val2 = self._parse_single_revenue(parts[1].strip())
                if val1 is not None and val2 is not None:
                    return (val1 + val2) / 2
            return None
        
        return self._parse_single_revenue(revenue_str)
    
    def _parse_single_revenue(self, revenue_str: str) -> float | None:
        """Parse a single revenue value string.
        
        Args:
            revenue_str: Revenue string (e.g., "1B", "500M")
        
        Returns:
            Parsed value in dollars, or None if parsing fails
        """
        try:
            # Extract number and multiplier
            multiplier = 1.0
            if revenue_str.endswith("B"):
                multiplier = 1e9
                revenue_str = revenue_str[:-1]
            elif revenue_str.endswith("M"):
                multiplier = 1e6
                revenue_str = revenue_str[:-1]
            elif revenue_str.endswith("K"):
                multiplier = 1e3
                revenue_str = revenue_str[:-1]
            
            value = float(revenue_str)
            return value * multiplier
        except (ValueError, TypeError):
            return None
    
    @property
    def name(self) -> str:
        """Return validator name.
        
        Returns:
            String identifier for this validator
        """
        return "data_consistency_validator"

