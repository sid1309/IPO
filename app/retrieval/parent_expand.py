"""Parent chunk retrieval expander and context assembler for generative LLM."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from app.ingestion.chunker import DocumentChunk
from app.retrieval.rerank import RerankedChunk


@dataclass
class AssembledContext:
    context_text: str
    source_chunks: List[RerankedChunk]
    cited_pages: List[int]
    sections_covered: List[str]


class ParentContextManager:
    """Manages mapping between child chunks and their full parent sections."""

    def __init__(self, parent_chunks: Optional[List[DocumentChunk]] = None):
        self._parent_map: Dict[str, DocumentChunk] = {}
        if parent_chunks:
            self.register_parents(parent_chunks)

    def register_parents(self, parent_chunks: List[DocumentChunk]) -> None:
        for chunk in parent_chunks:
            self._parent_map[chunk.chunk_id] = chunk

    def assemble_context(
        self,
        reranked_chunks: List[RerankedChunk],
        max_context_chars: int = 14000,
    ) -> AssembledContext:
        seen_parent_ids: Set[str] = set()
        formatted_blocks: List[str] = []
        cited_pages: Set[int] = set()
        sections_covered: Set[str] = set()
        total_chars = 0

        for chunk in reranked_chunks:
            parent_id = chunk.parent_id
            page_start = chunk.page_start
            page_end = chunk.page_end
            section = chunk.section

            cited_pages.add(page_start)
            if page_end != page_start:
                cited_pages.add(page_end)
            sections_covered.add(section)

            if parent_id and parent_id in self._parent_map:
                if parent_id in seen_parent_ids:
                    continue
                seen_parent_ids.add(parent_id)
                parent_chunk = self._parent_map[parent_id]
                content = parent_chunk.text
                page_info = f"Pages {parent_chunk.page_start}-{parent_chunk.page_end}"
            else:
                content = chunk.text
                page_info = f"Page {page_start}" if page_start == page_end else f"Pages {page_start}-{page_end}"

            block = (
                f"--- [DOCUMENT EXCERPT {len(formatted_blocks) + 1}] ---\n"
                f"Section: {section.replace('_', ' ').title()}\n"
                f"Physical Location: {page_info}\n"
                f"Type: {chunk.chunk_type.upper()}\n\n"
                f"{content}\n"
            )

            if total_chars + len(block) > max_context_chars:
                break

            formatted_blocks.append(block)
            total_chars += len(block)

        combined_text = "\n".join(formatted_blocks)

        return AssembledContext(
            context_text=combined_text,
            source_chunks=reranked_chunks,
            cited_pages=sorted(list(cited_pages)),
            sections_covered=sorted(list(sections_covered)),
        )
