import re
from dataclasses import dataclass
from typing import Optional

SEBI_DISCLAIMER = (
    "Disclaimer: This is an educational document-research system. "
    "Under SEBI (Research Analysts) Regulations, providing investment recommendations requires registration. "
    "This tool reports only what regulatory filings disclose and does not recommend to buy, sell, apply, or avoid any security."
)

DIRECT_ADVICE_PATTERNS = (
    r"\bshould\s+i\s+(?:apply|buy|invest|subscribe|avoid|sell)\b",
    r"\bis\s+this\s+(?:ipo|stock)\s+(?:good|bad|worth|profitable)\s+to\s+(?:buy|apply|invest)\b",
    r"\bwould\s+you\s+recommend\b",
    r"\brecommend\s+(?:buying|selling|investing|subscribing|applying)\b",
    r"\btarget\s+price\b",
    r"\bbuy\s+or\s+sell\b",
    r"\bworth\s+applying\b",
    r"\bmultibagger\b",
)

LISTING_ANALYSIS_PATTERNS = (
    r"\blisting\s+(?:gain|loss|losses|day|performance)\b",
    r"\bpotential\s+(?:listing|gain|loss|losses)\b",
    r"\b(?:gain|lose|loss)\s+(?:possible|potential|expected)\b",
    r"\bexpected\s+(?:listing|return|gain)\b",
    r"\bwill\s+(?:i|this|the)\s+(?:stock|ipo|company)?\s*(?:give|make|get|yield|generate|have)\s+.*?(?:profit|gain|return|loss|money)\b",
)

INJECTION_PATTERNS = (
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
    r"you\s+are\s+now\s+a\b",
    r"system\s*prompt\s*override",
    r"disregard\s+the\s+rules",
)


@dataclass
class GuardrailResult:
    is_safe: bool
    refusal_message: Optional[str] = None
    reason: Optional[str] = None
    analysis_mode: str = "standard"
    disclaimer: str = SEBI_DISCLAIMER


class InputGuardrail:
    """
    Enforces SEBI regulatory compliance by refusing direct investment advice queries,
    detecting prompt injection attempts, and routing listing performance questions
    to an objective document-backed valuation & risk analysis mode.
    """

    @classmethod
    def check_input(cls, question: str) -> GuardrailResult:
        clean_q = question.strip().lower()

        for pat in INJECTION_PATTERNS:
            if re.search(pat, clean_q):
                return GuardrailResult(
                    is_safe=False,
                    refusal_message="Request refused: System security guardrail detected an instruction override attempt.",
                    reason="prompt_injection",
                )

        for pat in DIRECT_ADVICE_PATTERNS:
            if re.search(pat, clean_q):
                return GuardrailResult(
                    is_safe=False,
                    refusal_message=(
                        f"I cannot provide an opinion on whether to apply, buy, or avoid this IPO. "
                        f"{SEBI_DISCLAIMER} However, I can extract and analyze factual details from the prospectus, "
                        f"such as the Use of Proceeds, financial metrics, debt levels, peer valuations, and disclosed risk factors. "
                        f"Please ask a factual question about the filing."
                    ),
                    reason="advice_refusal",
                )

        for pat in LISTING_ANALYSIS_PATTERNS:
            if re.search(pat, clean_q):
                return GuardrailResult(
                    is_safe=True,
                    analysis_mode="listing_analysis",
                    disclaimer=SEBI_DISCLAIMER,
                )

        return GuardrailResult(is_safe=True, analysis_mode="standard")
