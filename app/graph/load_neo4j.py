"""Neo4j knowledge graph loader and local in-memory fallback graph store."""

import logging
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.graph.schema import ExtractedGraph, GraphNode, GraphRelationship

logger = logging.getLogger(__name__)


class InMemoryGraphStore:
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.relationships: List[GraphRelationship] = []

    def insert_graph(self, graph: ExtractedGraph) -> int:
        for node in graph.nodes:
            self.nodes[node.node_id] = node
        for rel in graph.relationships:
            self.relationships.append(rel)
        return len(graph.nodes) + len(graph.relationships)

    def query_promoter_litigations(self, ipo_id: str) -> List[Dict[str, Any]]:
        results = []
        promoters = [
            n for n in self.nodes.values()
            if n.label == "Promoter" and n.ipo_id == ipo_id
        ]

        for p in promoters:
            group_rels = [
                r for r in self.relationships
                if r.source_id == p.node_id and r.rel_type in ("DIRECTOR_OF", "PROMOTER_OF")
            ]
            for gr in group_rels:
                group_node = self.nodes.get(gr.target_id)
                if not group_node or group_node.label != "GroupCompany":
                    continue
                lit_rels = [
                    r for r in self.relationships
                    if r.source_id == group_node.node_id and r.rel_type == "PARTY_TO"
                ]
                for lr in lit_rels:
                    lit_node = self.nodes.get(lr.target_id)
                    if not lit_node or lit_node.label != "Litigation":
                        continue
                    results.append({
                        "promoter": p.name,
                        "promoter_page": p.page,
                        "group_company": group_node.name,
                        "group_page": group_node.page,
                        "litigation": lit_node.name,
                        "litigation_amount": lit_node.properties.get("amount", "Unspecified"),
                        "litigation_status": lit_node.properties.get("status", "Pending"),
                        "litigation_page": lit_node.page,
                    })

        return results

    def query_common_directorships(self, ipo_id: str) -> List[Dict[str, Any]]:
        results = []
        issuer_directors = set()

        for r in self.relationships:
            if r.rel_type in ("DIRECTOR_OF", "PROMOTER_OF"):
                target = self.nodes.get(r.target_id)
                if target and target.label == "Company" and target.ipo_id == ipo_id:
                    issuer_directors.add(r.source_id)

        for r in self.relationships:
            if r.rel_type == "DIRECTOR_OF" and r.source_id in issuer_directors:
                target = self.nodes.get(r.target_id)
                source = self.nodes.get(r.source_id)
                if target and target.label == "GroupCompany":
                    results.append({
                        "person_name": source.name if source else r.source_id,
                        "group_company": target.name,
                        "page": r.page or (target.page if target else None),
                    })

        return results

    def query_connected_pages(
        self,
        ipo_id: str,
        query_entities: List[str],
        max_depth: int = 2,
    ) -> List[int]:
        pages = set()
        current_node_ids = set()

        for name in query_entities:
            clean = name.lower().strip()
            for n in self.nodes.values():
                if n.ipo_id == ipo_id and clean in n.name.lower():
                    current_node_ids.add(n.node_id)
                    if n.page:
                        pages.add(n.page)

        visited = set(current_node_ids)
        frontier = set(current_node_ids)

        for _ in range(max_depth):
            next_frontier = set()
            for r in self.relationships:
                if r.source_id in frontier and r.target_id not in visited:
                    if r.page:
                        pages.add(r.page)
                    tgt = self.nodes.get(r.target_id)
                    if tgt:
                        visited.add(tgt.node_id)
                        next_frontier.add(tgt.node_id)
                        if tgt.page:
                            pages.add(tgt.page)
                elif r.target_id in frontier and r.source_id not in visited:
                    if r.page:
                        pages.add(r.page)
                    src = self.nodes.get(r.source_id)
                    if src:
                        visited.add(src.node_id)
                        next_frontier.add(src.node_id)
                        if src.page:
                            pages.add(src.page)

            frontier = next_frontier
            if not frontier:
                break

        return sorted(list(pages))


class Neo4jLoader:
    def __init__(self, uri: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
        self._uri = uri or settings.NEO4J_URI
        self._username = username or settings.NEO4J_USERNAME
        self._password = password or settings.NEO4J_PASSWORD
        self._driver = None
        self._memory_store = InMemoryGraphStore()
        self._connect()

    def _connect(self):
        if not self._uri:
            logger.info("No Neo4j URI configured; using InMemoryGraphStore.")
            return

        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self._uri, auth=(self._username, self._password))
            self._driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j instance.")
            self._ensure_constraints()
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j ({e}); falling back to InMemoryGraphStore.")
            self._driver = None

    def _ensure_constraints(self):
        if not self._driver:
            return
        with self._driver.session() as session:
            for label in ("Company", "Promoter", "Director", "GroupCompany", "Litigation", "ObjectOfIssue"):
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.node_id IS UNIQUE")

    def insert_graph(self, graph: ExtractedGraph) -> int:
        self._memory_store.insert_graph(graph)

        if not self._driver:
            return len(graph.nodes) + len(graph.relationships)

        try:
            with self._driver.session() as session:
                for node in graph.nodes:
                    cypher = f"""
                    MERGE (n:{node.label} {{node_id: $node_id}})
                    ON CREATE SET n.name = $name, n.ipo_id = $ipo_id, n.page = $page
                    ON MATCH SET n.name = $name, n.page = $page
                    """
                    session.run(cypher, node_id=node.node_id, name=node.name, ipo_id=node.ipo_id, page=node.page)

                for rel in graph.relationships:
                    cypher = f"""
                    MATCH (s {{node_id: $source_id}})
                    MATCH (t {{node_id: $target_id}})
                    MERGE (s)-[r:{rel.rel_type}]->(t)
                    ON CREATE SET r.page = $page
                    """
                    session.run(cypher, source_id=rel.source_id, target_id=rel.target_id, page=rel.page)

            return len(graph.nodes) + len(graph.relationships)
        except Exception as e:
            logger.error(f"Failed to upsert to Neo4j: {e}")
            return len(graph.nodes) + len(graph.relationships)

    def store(self) -> InMemoryGraphStore:
        return self._memory_store

    def close(self):
        if self._driver:
            self._driver.close()
