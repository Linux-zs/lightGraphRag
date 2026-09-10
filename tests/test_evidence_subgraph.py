from src.api.server import _build_evidence_chain


def test_capped_evidence_never_has_dangling_edges():
    chain = _build_evidence_chain({"relationships": [
        {"src_id": "hub", "tgt_id": f"node-{i}", "description": "related"}
        for i in range(50)
    ]}, [])
    ids = {node.id for node in chain.nodes}
    assert len(ids) == 24
    assert chain.edges
    assert all(edge.source in ids and edge.target in ids for edge in chain.edges)
    for node in chain.nodes:
        degree = sum(edge.source == node.id or edge.target == node.id for edge in chain.edges)
        assert node.critical == (degree >= 3)


def test_relation_only_evidence_supplies_both_endpoint_nodes():
    chain = _build_evidence_chain({"relationships": [
        {"src_id": "A", "tgt_id": "B", "description": "depends on"},
    ]}, [])
    assert {node.id for node in chain.nodes} == {"A", "B"}
    assert len(chain.edges) == 1
