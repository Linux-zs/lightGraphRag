from pathlib import Path

import networkx as nx

from src.lightrag_service import LightRAGService


def make_service(graphml_path: Path) -> LightRAGService:
    service = LightRAGService.__new__(LightRAGService)
    service.working_dir = graphml_path.parent.parent
    service.workspace = graphml_path.parent.name
    return service


def test_find_graph_references_detects_deleted_doc_residuals(tmp_path):
    path = tmp_path / "kb" / "graph_chunk_entity_relation.graphml"
    path.parent.mkdir()
    graph = nx.Graph()
    graph.add_node("A", entity_id="A", source_id="doc_abcd-chunk-000", file_path="")
    graph.add_node("B", entity_id="B", source_id="", file_path="uploads/example.txt")
    graph.add_edge("A", "B", source_id="doc_abcd-chunk-001", file_path="")
    nx.write_graphml(graph, path)

    result = make_service(path).find_graph_references(
        doc_id="doc_abcd",
        doc_name="example.txt",
    )

    assert result["checked"] is True
    assert result["has_residuals"] is True
    assert result["node_count"] == 2
    assert result["edge_count"] == 1
    assert result["nodes"][0]["id"] == "A"


def test_find_graph_references_handles_missing_graph(tmp_path):
    result = make_service(tmp_path / "kb" / "graph_chunk_entity_relation.graphml").find_graph_references(
        doc_id="doc_missing",
        doc_name="missing.txt",
    )

    assert result["checked"] is True
    assert result["has_residuals"] is False
    assert result["graph_exists"] is False
