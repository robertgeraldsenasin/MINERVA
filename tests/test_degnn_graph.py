"""Unit tests for scripts/minerva_degnn.py — the DE-GNN graph module.

Covers:
- _l2_normalize correctness
- build_knn_graph_edges symmetry and shape
- GraphSAGE forward shape
- GraphSAGE training-mode vs eval-mode determinism

These tests verify the graph-construction half of BATB §1.4 SO 1.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEGNN_PATH = REPO_ROOT / "scripts" / "minerva_degnn.py"


@pytest.fixture(scope="module")
def degmod():
    """Load minerva_degnn as a module for testing."""
    spec = importlib.util.spec_from_file_location("minerva_degnn", DEGNN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["minerva_degnn"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# L2 normalize
# --------------------------------------------------------------------------

class TestL2Normalize:
    def test_l2_unit_norm(self, degmod):
        x = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]])
        out = degmod._l2_normalize(x)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0, 1.0])

    def test_l2_preserves_direction(self, degmod):
        x = np.array([[3.0, 4.0]])
        out = degmod._l2_normalize(x)
        np.testing.assert_array_almost_equal(out, [[0.6, 0.8]])

    def test_l2_zero_vector_safe(self, degmod):
        # eps clamps the denominator so zero vectors don't crash
        x = np.array([[0.0, 0.0], [1.0, 0.0]])
        out = degmod._l2_normalize(x)
        assert np.all(np.isfinite(out))
        # Second row should still be unit norm
        np.testing.assert_array_almost_equal(out[1], [1.0, 0.0])


# --------------------------------------------------------------------------
# kNN graph construction
# --------------------------------------------------------------------------

class TestBuildKnnGraphEdges:
    def test_basic_shape(self, degmod):
        np.random.seed(0)
        x = np.random.randn(10, 5)
        edge_index = degmod.build_knn_graph_edges(x, k=3)
        assert edge_index.shape[0] == 2  # [src, dst]
        # k=3 nearest neighbors, both directions stored, no self-loops
        # 10 nodes × 3 neighbors × 2 directions = 60 directed edges
        assert edge_index.shape[1] == 60

    def test_symmetry(self, degmod):
        # Both (a,b) and (b,a) should appear since we add both directions
        np.random.seed(0)
        x = np.random.randn(20, 8)
        edge_index = degmod.build_knn_graph_edges(x, k=2)
        edges = set(map(tuple, edge_index.T.tolist()))
        # For every (a,b), check (b,a) also exists
        for src, dst in list(edges)[:30]:  # sample first 30
            assert (dst, src) in edges, f"Edge ({src},{dst}) lacks reverse"

    def test_no_self_loops(self, degmod):
        # k+1 neighbors are queried, slot 0 is self, [1:] used.
        # So no self-loops should appear in the output.
        np.random.seed(0)
        x = np.random.randn(15, 6)
        edge_index = degmod.build_knn_graph_edges(x, k=2)
        for src, dst in edge_index.T.tolist():
            assert src != dst, f"Self-loop found at node {src}"

    def test_k_zero_raises(self, degmod):
        x = np.random.randn(5, 3)
        with pytest.raises(ValueError):
            degmod.build_knn_graph_edges(x, k=0)

    def test_k_negative_raises(self, degmod):
        x = np.random.randn(5, 3)
        with pytest.raises(ValueError):
            degmod.build_knn_graph_edges(x, k=-1)

    def test_deterministic(self, degmod):
        # Same input → same edge set
        np.random.seed(42)
        x = np.random.randn(8, 4)
        e1 = degmod.build_knn_graph_edges(x, k=2)
        e2 = degmod.build_knn_graph_edges(x, k=2)
        np.testing.assert_array_equal(e1, e2)


# --------------------------------------------------------------------------
# GraphSAGE forward
# --------------------------------------------------------------------------

class TestGraphSAGE:
    def test_forward_shape(self, degmod):
        torch.manual_seed(0)
        model = degmod.GraphSAGE(in_dim=8, hidden=16, out_dim=2)
        x = torch.randn(10, 8)
        # 5 directed edges
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long
        )
        out = model(x, edge_index)
        assert out.shape == (10, 2)  # 10 nodes × 2 classes

    def test_encode_shape(self, degmod):
        # encode() returns hidden representation, not class logits
        torch.manual_seed(0)
        model = degmod.GraphSAGE(in_dim=4, hidden=12, out_dim=2)
        x = torch.randn(6, 4)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        h = model.encode(x, edge_index)
        assert h.shape == (6, 12)  # 6 nodes × hidden

    def test_eval_mode_deterministic(self, degmod):
        # In eval mode, dropout is off → forward is deterministic
        torch.manual_seed(0)
        model = degmod.GraphSAGE(in_dim=4, hidden=8, out_dim=2)
        model.eval()
        x = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        with torch.no_grad():
            out1 = model(x, edge_index)
            out2 = model(x, edge_index)
        torch.testing.assert_close(out1, out2)

    def test_isolated_node_handled(self, degmod):
        # A node with NO incoming edges should still get a finite output
        # (the lin_self contributes; neigh_mean is zeroed by deg.clamp(min=1))
        torch.manual_seed(0)
        model = degmod.GraphSAGE(in_dim=4, hidden=8, out_dim=2)
        model.eval()
        x = torch.randn(4, 4)
        # Edges only connect nodes 0,1,2 — node 3 is isolated
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        with torch.no_grad():
            out = model(x, edge_index)
        assert torch.all(torch.isfinite(out))
        assert out.shape == (4, 2)

    def test_dropout_active_in_train_mode(self, degmod):
        # In train mode with dropout > 0, two forward passes should differ
        torch.manual_seed(0)
        model = degmod.GraphSAGE(in_dim=4, hidden=16, out_dim=2, dropout=0.5)
        model.train()
        x = torch.randn(8, 4)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 0]], dtype=torch.long
        )
        out1 = model(x, edge_index)
        out2 = model(x, edge_index)
        # With dropout, the two outputs should NOT be identical
        assert not torch.allclose(out1, out2)
