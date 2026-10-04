"""Pydantic schemas for the Neo4j Corporate Knowledge Graph."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

NodeType = Literal["Company", "Promoter", "Director", "GroupCompany", "Litigation", "ObjectOfIssue"]
RelationType = Literal["PROMOTER_OF", "DIRECTOR_OF", "HAS_GROUP_COMPANY", "PARTY_TO", "ALLOCATES_PROCEEDS_TO"]


class GraphNode(BaseModel):
    """Represents an entity node in the knowledge graph."""
    node_id: str = Field(description="Unique canonical identifier for the node (e.g. 'promoter_deepinder_goyal')")
    label: NodeType = Field(description="Primary entity label")
    name: str = Field(description="Human-readable entity name")
    ipo_id: str = Field(description="Associated IPO identifier")
    page: Optional[int] = Field(default=None, description="Physical page where entity is introduced")
    chunk_id: Optional[str] = Field(default=None, description="Source chunk ID for retrieval grounding")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties (e.g. role, amount, country)")


class GraphRelationship(BaseModel):
    """Represents a directed edge between two entity nodes."""
    source_id: str = Field(description="Source node_id")
    target_id: str = Field(description="Target node_id")
    rel_type: RelationType = Field(description="Relationship type")
    page: Optional[int] = Field(default=None, description="Physical page proving this relationship")
    chunk_id: Optional[str] = Field(default=None, description="Source chunk ID")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Edge properties (e.g. status, shareholding_pct)")


class ExtractedGraph(BaseModel):
    """Graph representation extracted from one or more prospectus chapters."""
    ipo_id: str
    nodes: List[GraphNode] = Field(default_factory=list)
    relationships: List[GraphRelationship] = Field(default_factory=list)
