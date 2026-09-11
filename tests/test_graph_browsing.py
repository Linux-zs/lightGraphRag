import networkx as nx

from src.lightrag_service import LightRAGService


def _service_with_graph(tmp_path):
    service = LightRAGService(
        {
            "paths": {
                "data_dir": str(tmp_path),
                "lightrag_dir": str(tmp_path / "lightrag"),
            }
        },
        workspace="kb",
    )
    graph = nx.Graph()
    graph.add_node("Hub", entity_id="Hub", entity_type="concept", description="central")
    for index in range(4):
        node_id = f"Neighbor-{index}"
        graph.add_node(
            node_id,
            entity_id=node_id,
            entity_type="service",
            description=f"neighbor {index}",
        )
        graph.add_edge("Hub", node_id, description="contains", weight=1.0)
    graph.add_node(
        "HiddenAtlas",
        entity_id="Hidden Atlas",
        entity_type="database",
        description="full-corpus-only search target",
    )
    service.graphml_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, service.graphml_path)
    return service


def test_full_graph_search_finds_node_omitted_from_ranked_overview(tmp_path):
    service = _service_with_graph(tmp_path)

    overview = service.read_graph(limit=2)
    result = service.search_graph_nodes("hidden atlas")

    assert "HiddenAtlas" not in {node["id"] for node in overview["nodes"]}
    assert result["total_matches"] == 1
    assert result["nodes"][0]["id"] == "HiddenAtlas"
    assert result["nodes"][0]["category"] == "database"


def test_focused_graph_returns_center_and_only_its_one_hop_neighborhood(tmp_path):
    service = _service_with_graph(tmp_path)

    result = service.read_graph(limit=3, focus_node_id="Hub")

    assert result["nodes"][0]["id"] == "Hub"
    assert len(result["nodes"]) == 3
    assert all(edge["source"] == "Hub" or edge["target"] == "Hub" for edge in result["edges"])
    assert result["metadata"]["view"] == "neighborhood"
    assert result["metadata"]["focus_node_id"] == "Hub"
    assert result["metadata"]["focus_found"] is True


def test_focused_graph_reports_missing_entity_without_falling_back_to_overview(tmp_path):
    service = _service_with_graph(tmp_path)

    result = service.read_graph(limit=3, focus_node_id="Missing")

    assert result["nodes"] == []
    assert result["edges"] == []
    assert result["metadata"]["focus_found"] is False
