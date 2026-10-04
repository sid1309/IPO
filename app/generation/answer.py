"""Answer generator orchestrating prompt construction, LLM generation, citation extraction, and numeric verification."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from app.generation.prompts import (
    SYSTEM_PROMPT_RAG,
    SYSTEM_PROMPT_LISTING_ANALYSIS,
    USER_PROMPT_TEMPLATE,
)
from app.generation.guardrails import InputGuardrail, SEBI_DISCLAIMER
from app.generation.citations import CitationValidator, CitationValidationReport
from app.generation.numeric_check import NumericVerifier, NumericVerificationReport
from app.retrieval.parent_expand import AssembledContext
from app.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """Complete structured answer response returned to the user or API."""
    question: str
    answer: str
    is_refusal: bool
    is_advice_refusal: bool
    is_listing_analysis: bool = False
    citations: CitationValidationReport = field(
        default_factory=lambda: CitationValidationReport([], 0, 0, 0, 1.0, False)
    )
    numeric_verification: NumericVerificationReport = field(
        default_factory=lambda: NumericVerificationReport(0, 0, 0, 1.0, True, [])
    )
    cited_pages: List[int] = field(default_factory=list)
    sections_covered: List[str] = field(default_factory=list)
    disclaimer: str = SEBI_DISCLAIMER
    model_used: str = "gemini-3.8-flash"


class AnswerGenerator:
    """End-to-end grounded answer generator with guardrails and verification."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate_answer(
        self,
        question: str,
        context: AssembledContext,
        company_name: str,
        doc_type: str = "RHP",
        use_cache: bool = True,
    ) -> AnswerResult:
        guard_res = InputGuardrail.check_input(question)
        if not guard_res.is_safe:
            return AnswerResult(
                question=question,
                answer=guard_res.refusal_message or "Request refused by guardrail.",
                is_refusal=True,
                is_advice_refusal=guard_res.reason == "advice_refusal",
                citations=CitationValidationReport([], 0, 0, 0, 1.0, False),
                cited_pages=[],
                sections_covered=[],
            )

        if not context.context_text.strip():
            return AnswerResult(
                question=question,
                answer="The provided prospectus excerpts do not contain information regarding this topic.",
                is_refusal=True,
                is_advice_refusal=False,
                citations=CitationValidationReport([], 0, 0, 0, 1.0, False),
                cited_pages=[],
                sections_covered=[],
            )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            company_name=company_name,
            doc_type=doc_type,
            context_text=context.context_text,
            question=question,
        )

        if guard_res.analysis_mode == "listing_analysis":
            system_instruction = SYSTEM_PROMPT_LISTING_ANALYSIS
        else:
            system_instruction = SYSTEM_PROMPT_RAG

        raw_answer = self.llm_client.generate(
            prompt=user_prompt,
            system_instruction=system_instruction,
            temperature=0.0,
            use_cache=use_cache,
        )

        citation_report = CitationValidator.validate_citations(
            answer_text=raw_answer,
            retrieved_pages=context.cited_pages,
        )

        numeric_report = NumericVerifier.verify(
            answer=raw_answer,
            context_text=context.context_text,
        )

        lower_ans = raw_answer.lower()
        is_refusal = (
            "does not contain" in lower_ans
            or "do not contain" in lower_ans
            or "not found in the provided" in lower_ans
        )

        final_answer = raw_answer.strip()
        if "disclaimer" not in final_answer.lower() and "regulatory notice" not in final_answer.lower():
            final_answer += f"\n\n---\n*{SEBI_DISCLAIMER}*"

        return AnswerResult(
            question=question,
            answer=final_answer,
            is_refusal=is_refusal,
            is_advice_refusal=False,
            is_listing_analysis=guard_res.analysis_mode == "listing_analysis",
            citations=citation_report,
            numeric_verification=numeric_report,
            cited_pages=context.cited_pages,
            sections_covered=context.sections_covered,
        )
