#!/usr/bin/env python3
"""
Heart Protocol Funding Transparency Script

Generates transparent funding reports and tracks ethical funding compliance.
"""

import json
import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class FundingSource:
    """Represents a funding source"""
    name: str
    type: str  # 'individual', 'organization', 'enterprise', 'grant'
    amount: float
    frequency: str  # 'one-time', 'monthly', 'annual'
    purpose: str
    ethical_approval: bool
    transparency_level: str  # 'public', 'anonymous', 'aggregate'
    start_date: str
    end_date: str = None

@dataclass
class FundingExpense:
    """Represents a funding expense"""
    category: str
    amount: float
    description: str
    date: str
    healing_impact: str
    community_benefit: bool

@dataclass
class FundingReport:
    """Quarterly funding transparency report"""
    quarter: str
    year: int
    total_revenue: float
    total_expenses: float
    sources: List[FundingSource]
    expenses: List[FundingExpense]
    ethical_compliance_score: float
    community_feedback: str
    goals_next_quarter: List[str]

class EthicalFundingTracker:
    """
    Transparent funding tracker for Heart Protocol.
    
    Principles:
    - Complete transparency on fund sources and usage
    - Community oversight on ethical compliance
    - Prioritize healing outcomes over profit
    - Never exploit user vulnerability for revenue
    """
    
    def __init__(self):
        self.funding_sources = []
        self.expenses = []
        self.ethical_requirements = self._load_ethical_requirements()
        self.transparency_ledger = []
    
    def _load_ethical_requirements(self) -> Dict[str, Any]:
        """Load ethical funding requirements"""
        return {
            'prohibited_sources': [
                'surveillance_companies',
                'data_brokers',
                'predatory_lenders',
                'gambling_companies',
                'tobacco_companies',
                'companies_with_poor_labor_practices'
            ],
            'required_commitments': [
                'user_wellbeing_over_profit',
                'transparent_data_usage',
                'community_governance_participation',
                'privacy_protection',
                'ethical_ai_practices'
            ],
            'maximum_profit_margin': 15.0,  # Maximum 15% profit margin
            'minimum_community_reinvestment': 25.0,  # Minimum 25% back to community
            'salary_equity_ratio': 3.0  # No executive salary > 3x median
        }
    
    def add_funding_source(self, source: FundingSource) -> bool:
        """Add a funding source after ethical review"""
        if self._passes_ethical_review(source):
            self.funding_sources.append(source)
            self._log_funding_event('source_added', source)
            return True
        else:
            self._log_funding_event('source_rejected', source)
            return False
    
    def _passes_ethical_review(self, source: FundingSource) -> bool:
        """Check if funding source passes ethical review"""
        # Check against prohibited sources
        if any(prohibited in source.name.lower() 
               for prohibited in self.ethical_requirements['prohibited_sources']):
            return False
        
        # Enterprise sources need special review
        if source.type == 'enterprise':
            return self._review_enterprise_source(source)
        
        # Individual and community sources generally approved
        return True
    
    def _review_enterprise_source(self, source: FundingSource) -> bool:
        """Special review process for enterprise sources"""
        # Would integrate with community governance system
        # For now, basic checks
        required_commitments = self.ethical_requirements['required_commitments']
        
        # Check if purpose aligns with healing mission
        healing_keywords = ['wellbeing', 'mental health', 'community care', 'healing']
        purpose_aligned = any(keyword in source.purpose.lower() 
                            for keyword in healing_keywords)
        
        return purpose_aligned and source.ethical_approval
    
    def record_expense(self, expense: FundingExpense):
        """Record a funding expense with transparency"""
        self.expenses.append(expense)
        self._log_funding_event('expense_recorded', expense)
    
    def generate_quarterly_report(self, quarter: str, year: int) -> FundingReport:
        """Generate quarterly transparency report"""
        # Filter for quarter
        quarter_sources = self._filter_by_quarter(self.funding_sources, quarter, year)
        quarter_expenses = self._filter_by_quarter(self.expenses, quarter, year)
        
        total_revenue = sum(source.amount for source in quarter_sources)
        total_expenses = sum(expense.amount for expense in quarter_expenses)
        
        ethical_score = self._calculate_ethical_compliance_score()
        
        return FundingReport(
            quarter=quarter,
            year=year,
            total_revenue=total_revenue,
            total_expenses=total_expenses,
            sources=quarter_sources,
            expenses=quarter_expenses,
            ethical_compliance_score=ethical_score,
            community_feedback="Quarterly community survey results here",
            goals_next_quarter=[
                "Maintain 100% ethical funding compliance",
                "Increase community funding percentage",
                "Launch transparency dashboard improvements"
            ]
        )
    
    def _filter_by_quarter(self, items: List, quarter: str, year: int) -> List:
        """Filter items by quarter and year"""
        # Implementation would filter based on date ranges
        return items  # Simplified for example
    
    def _calculate_ethical_compliance_score(self) -> float:
        """Calculate ethical compliance score (0-100)"""
        score = 100.0
        
        # Check funding source diversity
        total_funding = sum(source.amount for source in self.funding_sources)
        if total_funding > 0:
            enterprise_percentage = sum(
                source.amount for source in self.funding_sources 
                if source.type == 'enterprise'
            ) / total_funding * 100
            
            # Penalize over-reliance on enterprise funding
            if enterprise_percentage > 60:
                score -= (enterprise_percentage - 60) * 0.5
        
        # Check expense allocation
        total_expenses = sum(expense.amount for expense in self.expenses)
        if total_expenses > 0:
            community_expenses = sum(
                expense.amount for expense in self.expenses 
                if expense.community_benefit
            ) / total_expenses * 100
            
            # Reward community investment
            if community_expenses >= 25:
                score += min(10, (community_expenses - 25) * 0.2)
            else:
                score -= (25 - community_expenses) * 0.3
        
        return max(0, min(100, score))
    
    def _log_funding_event(self, event_type: str, data: Any):
        """Log funding events for transparency"""
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'event_type': event_type,
            'data': asdict(data) if hasattr(data, '__dict__') else str(data)
        }
        self.transparency_ledger.append(event)
    
    def export_transparency_report(self, output_path: str):
        """Export full transparency report"""
        report_data = {
            'heart_protocol_funding_transparency': {
                'report_date': datetime.datetime.now().isoformat(),
                'funding_sources': [asdict(source) for source in self.funding_sources],
                'expenses': [asdict(expense) for expense in self.expenses],
                'ethical_compliance_score': self._calculate_ethical_compliance_score(),
                'ethical_requirements': self.ethical_requirements,
                'transparency_ledger': self.transparency_ledger[-100:],  # Last 100 events
                'community_commitments': {
                    'open_source_forever': True,
                    'user_data_never_sold': True,
                    'algorithmic_transparency': True,
                    'community_governance': True,
                    'crisis_support_always_free': True
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def check_funding_health(self) -> Dict[str, Any]:
        """Check overall funding health and sustainability"""
        total_monthly_revenue = sum(
            source.amount for source in self.funding_sources 
            if source.frequency == 'monthly'
        )
        
        total_monthly_expenses = sum(
            expense.amount for expense in self.expenses 
            if 'monthly' in expense.description.lower()
        )
        
        sustainability_months = (
            total_monthly_revenue / total_monthly_expenses 
            if total_monthly_expenses > 0 else float('inf')
        )
        
        return {
            'monthly_revenue': total_monthly_revenue,
            'monthly_expenses': total_monthly_expenses,
            'sustainability_months': sustainability_months,
            'ethical_compliance': self._calculate_ethical_compliance_score(),
            'funding_diversity': self._calculate_funding_diversity(),
            'community_support_percentage': self._calculate_community_support_percentage(),
            'recommendations': self._generate_funding_recommendations()
        }
    
    def _calculate_funding_diversity(self) -> float:
        """Calculate funding source diversity (higher is better)"""
        if not self.funding_sources:
            return 0.0
        
        source_types = {}
        total_amount = sum(source.amount for source in self.funding_sources)
        
        for source in self.funding_sources:
            source_types[source.type] = source_types.get(source.type, 0) + source.amount
        
        # Calculate Shannon diversity index
        diversity = 0.0
        for amount in source_types.values():
            proportion = amount / total_amount
            if proportion > 0:
                diversity -= proportion * math.log2(proportion)
        
        return diversity
    
    def _calculate_community_support_percentage(self) -> float:
        """Calculate percentage of funding from community sources"""
        total_funding = sum(source.amount for source in self.funding_sources)
        if total_funding == 0:
            return 0.0
        
        community_funding = sum(
            source.amount for source in self.funding_sources 
            if source.type in ['individual', 'community']
        )
        
        return (community_funding / total_funding) * 100
    
    def _generate_funding_recommendations(self) -> List[str]:
        """Generate recommendations for funding health"""
        recommendations = []
        
        ethical_score = self._calculate_ethical_compliance_score()
        if ethical_score < 90:
            recommendations.append("Review funding sources for ethical compliance")
        
        community_percentage = self._calculate_community_support_percentage()
        if community_percentage < 30:
            recommendations.append("Increase community funding through GitHub Sponsors and Ko-fi")
        
        diversity = self._calculate_funding_diversity()
        if diversity < 1.5:
            recommendations.append("Diversify funding sources to reduce dependency risk")
        
        return recommendations

def main():
    """Example usage of funding transparency system"""
    tracker = EthicalFundingTracker()
    
    # Add example funding sources
    github_sponsors = FundingSource(
        name="GitHub Sponsors Community",
        type="individual",
        amount=2500.0,
        frequency="monthly",
        purpose="Support caring algorithm development",
        ethical_approval=True,
        transparency_level="aggregate",
        start_date="2024-01-01"
    )
    
    ko_fi_support = FundingSource(
        name="Ko-fi Community Support",
        type="individual", 
        amount=800.0,
        frequency="monthly",
        purpose="One-time community contributions",
        ethical_approval=True,
        transparency_level="anonymous",
        start_date="2024-01-01"
    )
    
    ethical_enterprise = FundingSource(
        name="Mental Health Nonprofit Partnership",
        type="enterprise",
        amount=5000.0,
        frequency="monthly",
        purpose="Healing-focused algorithm research collaboration",
        ethical_approval=True,
        transparency_level="public",
        start_date="2024-02-01"
    )
    
    # Add sources
    tracker.add_funding_source(github_sponsors)
    tracker.add_funding_source(ko_fi_support)
    tracker.add_funding_source(ethical_enterprise)
    
    # Add example expenses
    infrastructure_expense = FundingExpense(
        category="Infrastructure",
        amount=1200.0,
        description="Monthly server costs for caring algorithms",
        date="2024-01-01",
        healing_impact="Enables 24/7 crisis intervention and care delivery",
        community_benefit=True
    )
    
    development_expense = FundingExpense(
        category="Development",
        amount=4000.0,
        description="Core team salaries (living wages)",
        date="2024-01-01", 
        healing_impact="Maintains and improves caring algorithm quality",
        community_benefit=True
    )
    
    community_expense = FundingExpense(
        category="Community",
        amount=800.0,
        description="Community events and support programs",
        date="2024-01-01",
        healing_impact="Builds healing-focused community connections",
        community_benefit=True
    )
    
    tracker.record_expense(infrastructure_expense)
    tracker.record_expense(development_expense)
    tracker.record_expense(community_expense)
    
    # Generate reports
    health_report = tracker.check_funding_health()
    print("Heart Protocol Funding Health Report:")
    print(json.dumps(health_report, indent=2))
    
    quarterly_report = tracker.generate_quarterly_report("Q1", 2024)
    print(f"\nQuarterly Report Q1 2024:")
    print(f"Total Revenue: ${quarterly_report.total_revenue:,.2f}")
    print(f"Total Expenses: ${quarterly_report.total_expenses:,.2f}")
    print(f"Ethical Compliance Score: {quarterly_report.ethical_compliance_score:.1f}/100")
    
    # Export full transparency report
    tracker.export_transparency_report("funding_transparency_report.json")
    print("\nTransparency report exported to funding_transparency_report.json")

if __name__ == "__main__":
    import math
    main()