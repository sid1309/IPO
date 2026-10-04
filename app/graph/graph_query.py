"""GraphRAG query engine answering multi-hop corporate network questions."""

import logging
from typing import Any, Dict, List, Optional
from app.graph.load_neo4j import Neo4jLoader

logger = logging.getLogger(__name__)


class GraphRAGQueryEngine:
    def __init__(self, loader: Optional[Neo4jLoader] = None):
        self.loader = loader or Neo4jLoader()

    def get_promoter_litigations(self, ipo_id: str) -> List[Dict[str, Any]]:
        if getattr(self.loader, "_driver", None):
            cypher = """
            MATCH (p:Promoter {ipo_id: $ipo_id})-[:DIRECTOR_OF]->(g:GroupCompany)-[:PARTY_TO]->(l:Litigation)
            RETURN p.name AS promoter,
                   p.page AS promoter_page,
                   g.name AS group_company,
                   g.page AS group_page,
                   l.name AS litigation,
                   l.amount AS litigation_amount,
                   l.status AS litigation_status,
                   l.page AS litigation_page
            """
            try:
                with self.loader._driver.session() as session:
                    res = session.run(cypher, ipo_id=ipo_id)
                    return [dict(record) for record in res]
            except Exception as e:
                logger.error(f"Live Cypher query failed ({e}), falling back to in-memory store.")

        return self.loader.store().query_promoter_litigations(ipo_id)

    def get_common_directorships(self, ipo_id: str) -> List[Dict[str, Any]]:
        if getattr(self.loader, "_driver", None):
            cypher = """
            MATCH (p {ipo_id: $ipo_id})-[:DIRECTOR_OF]->(c:Company {ipo_id: $ipo_id})
            MATCH (p)-[:DIRECTOR_OF]->(g:GroupCompany)
            RETURN p.name AS person_name, g.name AS group_company, g.page AS page
            """
            try:
                with self.loader._driver.session() as session:
                    res = session.run(cypher, ipo_id=ipo_id)
                    return [dict(record) for record in res]
            except Exception as e:
                logger.error(f"Live directorship query failed: {e}")

        return self.loader.store().query_common_directorships(ipo_id)

    def get_graph_augmented_pages(self, query: str, ipo_id: str) -> List[int]:
        known_nodes = [
            n for n in self.loader.store().nodes.values()
            if n.ipo_id == ipo_id
        ]
        matched_names = []
        clean_q = query.lower()

        for node in known_nodes:
            if len(node.name) > 3 and node.name.lower() in clean_q:
                matched_names.append(node.name)

        if not matched_names:
            return []

        return self.loader.store().query_connected_pages(
            query_entities=matched_names,
            ipo_id=ipo_id,
        )
