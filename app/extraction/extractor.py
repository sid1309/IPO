"""Structured Summary Card and Proceeds Extractor."""

import json
import logging
from pathlib import Path
from typing import List, Optional
from app.core.config import settings
from app.extraction.summary_schema import (
    IPOSummaryCard,
    IssueDetails,
    ObjectOfIssueItem,
    FinancialSnapshot,
    PromoterInfo,
)
from app.ingestion.chunker import DocumentChunk
from app.llm.client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_EXTRACTION = """You are an expert financial analyst extracting structured IPO information from prospectus excerpts.
You must return ONLY a valid JSON object matching the requested schema.
Do NOT fabricate numbers. If a field cannot be determined from the excerpts, set it to null.
Preserve exact currency units and formatting (e.g. "₹ 90,000 million").
"""

EXTRACTION_USER_PROMPT = """Extract the structured IPO Summary Card from the provided prospectus excerpts.

Excerpts:
================================================================================
{context_text}
================================================================================

JSON Schema format:
{{
  "issuer_name": "Company Name",
  "issue_type": "100% Book Built",
  "price_band_floor": 72.0,
  "price_band_cap": 76.0,
  "face_value": 1.0,
  "lot_size": 195,
  "fresh_issue_amount": "₹ 90,000 million",
  "ofs_amount": "₹ 3,750 million",
  "total_issue_size": "₹ 93,750 million",
  "listing_exchanges": ["BSE", "NSE"],
  "objects_of_issue": [
    {{
      "object_category": "Funding organic and inorganic growth initiatives",
      "description": "Customer acquisition and branding initiatives",
      "estimated_amount": "₹ 67,500 million",
      "deployment_schedule": "FY22: 20,000, FY23: 30,000, FY24: 17,500"
    }}
  ],
  "financials": [
    {{
      "fiscal_year": "Fiscal 2021",
      "revenue": "₹ 19,937.89 million",
      "ebitda": "₹ (4,672.00) million",
      "pat": "₹ (8,164.20) million",
      "diluted_eps": "(₹ 1.08)",
      "ronw": "-14.2%",
      "nav_per_share": "₹ 8.60"
    }}
  ],
  "promoters": [
    {{
      "name": "Info Edge (India) Limited",
      "is_selling_shareholder": true,
      "pre_offer_equity_shares": 145000000,
      "pre_offer_percentage": "18.4%",
      "shares_offered": 49200000
    }}
  ],
  "key_risks_summary": [
    "History of net losses and anticipation of increased expenses in future periods."
  ]
}}

Return ONLY the raw JSON object.
"""


class SummaryCardExtractor:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.cache_dir = Path(settings.PROCESSED_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, ipo_id: str) -> Path:
        return self.cache_dir / f"{ipo_id}_summary.json"

    def extract_summary_card(
        self,
        ipo_id: str,
        chunks: List[DocumentChunk],
        force_refresh: bool = False,
    ) -> IPOSummaryCard:
        cache_path = self.get_cache_path(ipo_id)
        if cache_path.exists() and not force_refresh:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return IPOSummaryCard.model_validate_json(f.read())
            except Exception as e:
                logger.warning(f"Failed to load cached summary card from {cache_path}: {e}")

        key_sections = {
            "summary_of_offer",
            "objects_of_issue",
            "financial_information",
            "our_promoters",
            "risk_factors",
            "general_and_summary",
        }

        targeted = [c for c in chunks if any(sec in c.section.lower() for sec in key_sections)]
        if not targeted:
            targeted = chunks[:20]

        targeted.sort(key=lambda c: (c.chunk_type != "table", c.page_start))

        context_parts = []
        char_count = 0
        for c in targeted:
            if char_count + len(c.text) > 12000:
                break
            header = f"\n--- [Section: {c.section}, Page {c.page_start}, Type: {c.chunk_type.upper()}] ---"
            context_parts.append(header)
            context_parts.append(c.text)
            char_count += len(c.text) + len(header)

        context_text = "\n\n".join(context_parts)
        prompt = EXTRACTION_USER_PROMPT.format(context_text=context_text)

        raw_response = self.llm_client.generate(
            prompt=prompt,
            system_instruction=SYSTEM_PROMPT_EXTRACTION,
            temperature=0.0,
            use_cache=False,
        )

        data = self._clean_and_parse_json(raw_response)

        issue_details = IssueDetails(
            issuer_name=data.get("issuer_name") or ipo_id.replace("-", " ").title(),
            issue_type=data.get("issue_type") or "100% Book Built",
            price_band_floor=data.get("price_band_floor"),
            price_band_cap=data.get("price_band_cap"),
            face_value=data.get("face_value"),
            lot_size=data.get("lot_size"),
            fresh_issue_amount=data.get("fresh_issue_amount"),
            ofs_amount=data.get("ofs_amount"),
            total_issue_size=data.get("total_issue_size"),
            listing_exchanges=data.get("listing_exchanges") or ["BSE", "NSE"],
        )

        objects_of_issue = [
            ObjectOfIssueItem(
                description=item.get("description", ""),
                estimated_amount=item.get("estimated_amount", "N/A"),
                deployment_schedule=item.get("deployment_schedule"),
                page_reference=item.get("page_reference"),
            )
            for item in data.get("objects_of_issue", [])
        ]

        financials = [
            FinancialSnapshot(
                fiscal_year=item.get("fiscal_year", ""),
                revenue=item.get("revenue"),
                ebitda=item.get("ebitda"),
                pat=item.get("pat"),
                diluted_eps=item.get("diluted_eps"),
                ronw=item.get("ronw"),
                nav_per_share=item.get("nav_per_share"),
                page_reference=item.get("page_reference"),
            )
            for item in data.get("financials", [])
        ]

        promoters = [
            PromoterInfo(
                name=item.get("name", ""),
                is_selling_shareholder=item.get("is_selling_shareholder", False),
                pre_offer_equity_shares=item.get("pre_offer_equity_shares"),
                pre_offer_percentage=item.get("pre_offer_percentage"),
                shares_offered=item.get("shares_offered"),
                page_reference=item.get("page_reference"),
            )
            for item in data.get("promoters", [])
        ]

        key_risks = data.get("key_risks_summary", [])

        summary_card = IPOSummaryCard(
            ipo_id=ipo_id,
            issue_details=issue_details,
            objects_of_issue=objects_of_issue,
            financials=financials,
            promoters=promoters,
            key_risks_summary=key_risks,
        )

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(summary_card.model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"Failed to cache summary card to {cache_path}: {e}")

        return summary_card

    def _clean_and_parse_json(self, raw_text: str) -> dict:
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()

        try:
            return json.loads(clean)
        except Exception as e:
            logger.error(f"Failed to parse summary JSON: {e}")
            return {}
