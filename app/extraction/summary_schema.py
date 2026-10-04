"""Pydantic schemas for the IPO Prospectus Structured Summary Card."""

from typing import List, Optional
from pydantic import BaseModel, Field


class IssueDetails(BaseModel):
    """Core issuance details and pricing structure."""
    issuer_name: str = Field(description="Name of the issuer company")
    issue_type: str = Field(default="100% Book Built", description="e.g. 100% Book Built Issue")
    price_band_floor: Optional[float] = Field(default=None, description="Lower price band in ₹")
    price_band_cap: Optional[float] = Field(default=None, description="Upper price band in ₹")
    face_value: Optional[float] = Field(default=None, description="Face value per equity share in ₹")
    lot_size: Optional[int] = Field(default=None, description="Minimum bid lot / number of equity shares")
    fresh_issue_amount: Optional[str] = Field(default=None, description="Fresh issue size with units (e.g. ₹ 90,000 million)")
    ofs_amount: Optional[str] = Field(default=None, description="Offer for sale size with units (e.g. ₹ 3,750 million)")
    total_issue_size: Optional[str] = Field(default=None, description="Aggregated offer size with units (e.g. ₹ 93,750 million)")
    listing_exchanges: List[str] = Field(default_factory=lambda: ["BSE", "NSE"], description="Proposed listing bourses")
    page_reference: Optional[int] = Field(default=None, description="Physical page number in prospectus")


class ObjectOfIssueItem(BaseModel):
    """Itemized capital deployment objective."""
    description: str = Field(description="Detailed purpose of fund utilization")
    estimated_amount: str = Field(description="Amount allocated with currency units (e.g. ₹ 67,500 million)")
    deployment_schedule: Optional[str] = Field(default=None, description="e.g. FY22: 20,000, FY23: 30,000, FY24: 17,500")
    page_reference: Optional[int] = Field(default=None, description="Physical page number in prospectus")


class FinancialSnapshot(BaseModel):
    """Fiscal year financial summary metrics."""
    fiscal_year: str = Field(description="e.g. Fiscal 2021, Fiscal 2020")
    revenue: Optional[str] = Field(default=None, description="Total Revenue from Operations with units")
    ebitda: Optional[str] = Field(default=None, description="EBITDA with units")
    pat: Optional[str] = Field(default=None, description="Profit / (Loss) After Tax with units")
    diluted_eps: Optional[str] = Field(default=None, description="Diluted Earnings Per Share in ₹")
    ronw: Optional[str] = Field(default=None, description="Return on Net Worth percentage (e.g. -14.2%)")
    nav_per_share: Optional[str] = Field(default=None, description="Net Asset Value per share in ₹")
    page_reference: Optional[int] = Field(default=None, description="Physical page number in prospectus")


class PromoterInfo(BaseModel):
    """Promoter or selling shareholder structure."""
    name: str = Field(description="Name of promoter or selling entity")
    is_selling_shareholder: bool = Field(default=False, description="Whether selling shares in OFS")
    pre_offer_equity_shares: Optional[int] = Field(default=None, description="Number of shares held prior to offer")
    pre_offer_percentage: Optional[str] = Field(default=None, description="Percentage holding prior to offer (e.g. 74.25%)")
    shares_offered: Optional[int] = Field(default=None, description="Number of shares offered for sale")
    page_reference: Optional[int] = Field(default=None, description="Physical page number in prospectus")


class IPOSummaryCard(BaseModel):
    """Comprehensive structured summary card for an IPO filing."""
    ipo_id: str
    issue_details: IssueDetails
    objects_of_issue: List[ObjectOfIssueItem] = Field(default_factory=list)
    financials: List[FinancialSnapshot] = Field(default_factory=list)
    promoters: List[PromoterInfo] = Field(default_factory=list)
    key_risks_summary: List[str] = Field(default_factory=list, description="Top 3-5 bulleted risk factors")
