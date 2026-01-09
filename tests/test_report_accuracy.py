"""Tests for report accuracy and data integrity.

This module contains tests to ensure reports contain accurate data,
proper source citations, and maintain data integrity throughout
the generation process.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from src.agents.report_agent import ReportAgent
from src.graph.state import WorkflowState
from src.models.insight_model import Insight
from src.models.source_quality import SourceQuality
from src.utils.data_verification_service import DataVerificationService


class TestReportAccuracy:
    """Test suite for report accuracy and data integrity."""
    
    def test_report_includes_user_query(self):
        """Test that reports include the original user query for transparency."""
        # Setup
        llm = Mock(spec=BaseChatModel)
        agent = ReportAgent(llm=llm, config={})
        state = WorkflowState(
            user_query="Analyze competitors in the AI market",
            insights={
                "swot": {
                    "strengths": ["Strong AI capabilities"],
                    "weaknesses": ["High costs"],
                    "opportunities": ["Growing market"],
                    "threats": ["Competition"]
                },
                "positioning": "Leading AI platform with strong market presence and innovative technology solutions that dominate the competitive landscape in enterprise applications.",
                "trends": ["AI adoption increasing"],
                "opportunities": ["Market expansion"]
            }
        )
        
        # Execute
        summary = agent._prepare_insights_summary(Insight(**state["insights"]), state)
        
        # Assert
        assert "Original User Query: Analyze competitors in the AI market" in summary
    
    def test_source_citations_in_report(self):
        """Test that reports include proper source citations."""
        # Setup
        llm = Mock(spec=BaseChatModel)
        agent = ReportAgent(llm=llm, config={})
        state = WorkflowState(
            collected_data={
                "competitors": [
                    {
                        "name": "Company A",
                        "source_url": "https://company-a.com",
                        "source_quality": SourceQuality.PRIMARY
                    }
                ]
            },
            insights={
                "swot": {
                    "strengths": ["Strong market position"],
                    "weaknesses": [],
                    "opportunities": [],
                    "threats": []
                },
                "positioning": "Strong market position with verified data sources and comprehensive competitive analysis covering pricing, features, and market positioning strategies.",
            }
        )
        
        # Execute
        summary = agent._prepare_insights_summary(Insight(**state["insights"]), state)
        
        # Assert
        assert "Source Quality Statistics" in summary
        assert "Primary sources" in summary
    
    def test_data_verification_statistics(self):
        """Test that data verification statistics are included."""
        # Setup
        llm = Mock(spec=BaseChatModel)
        agent = ReportAgent(llm=llm, config={})
        state = WorkflowState(
            collected_data={
                "competitors": [
                    {
                        "name": "Company A",
                        "source_quality": SourceQuality.PRIMARY
                    }
                ]
            },
            insights={
                "swot": {
                    "strengths": ["Verified market share"],
                    "weaknesses": [],
                    "opportunities": [],
                    "threats": []
                },
                "positioning": "Data verification shows strong market presence with verified quantitative metrics and comprehensive source validation across multiple quality levels.",
            }
        )
        
        # Execute
        summary = agent._prepare_insights_summary(Insight(**state["insights"]), state)
        
        # Assert
        assert "Source Quality Statistics" in summary
    
    def test_source_quality_warnings(self):
        """Test that source quality warnings are properly formatted."""
        # Setup
        llm = Mock(spec=BaseChatModel)
        agent = ReportAgent(llm=llm, config={})
        state = WorkflowState(
            source_quality_warnings=["Low quality source detected"],
            insights={
                "swot": {
                    "strengths": ["Good data"],
                    "weaknesses": [],
                    "opportunities": [],
                    "threats": []
                },
                "positioning": "Comprehensive analysis with source quality warnings and validation statistics to ensure data integrity and transparency in competitive intelligence reporting.",
            }
        )
        
        # Execute
        summary = agent._prepare_insights_summary(Insight(**state["insights"]), state)
        
        # Assert
        assert "Source Quality Warnings" in summary
        assert "Low quality source detected" in summary