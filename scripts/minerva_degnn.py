from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import dump, load
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Model


class GraphSAGE(nn.Module):
    """A small, dependency-free GraphSAGE implementation (mean aggregator)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.lin_self1 = nn.Linear(in_dim, hidden)
        self.lin_neigh1 = nn.Linear(in_dim, hidden)
        self.lin_self2 = nn.Linear(hidden, hidden)
        self.lin_neigh2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.dropout = float(dropout)

    def _sage_layer(self, x: torch.Tensor, edge_index: torch.Tensor, lin_self: nn.Linear, lin_neigh: nn.Linear) -> torch.Tensor:
        """Compute one GraphSAGE layer with a mean neighbor aggregator.

        edge_index is shaped [2, E] where src -> dst.
        """

        src, dst = edge_index

        # Sum of neighbor features for each dst
        neigh_sum = torch.zeros_like(x)
        neigh_sum.index_add_(0, dst, x[src])

        # Degree for mean
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=x.device))

        neigh_mean = neigh_sum / deg.clamp(min=1).unsqueeze(1)
        h = lin_self(x) + lin_neigh(neigh_mean)
        return F.relu(h)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return the latent node embeddings (after the 2nd SAGE layer)."""
        h = self._sage_layer(x, edge_index, self.lin_self1, self.lin_neigh1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self._sage_layer(h, edge_index, self.lin_self2, self.lin_neigh2)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        return self.out(h)


# -----------------------------------------------------------------------------
# Artifacts


@dataclass
class DegnnArtifacts:
    """Everything needed to re-run inference for new samples."""

    # Schema
    feature_cols: List[str]
    edge_cols: List[str]

    # Preprocessing
    scaler_node: StandardScaler
    scaler_edge: StandardScaler

    # Graph reference set (the nodes used during training)
    ids: np.ndarray  # shape [N]
    splits: np.ndarray  # shape [N]
    y_true: np.ndarray  # shape [N]
    x_node_scaled: np.ndarray  # shape [N, F]
    x_edge_norm: np.ndarray  # shape [N, E]
    edge_index: np.ndarray  # shape [2, num_edges]
    knn_k: int

    # Model config
    hidden_dim: int
    dropout: float


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def build_knn_graph_edges(x_edge_norm: np.ndarray, k: int) -> np.ndarray:
    """Build a symmetric kNN graph (stored as directed edges src->dst)."""
    if k <= 0:
        raise ValueError("k must be > 0")

    nn_model = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn_model.fit(x_edge_norm)
    neigh = nn_model.kneighbors(x_edge_norm, return_distance=False)

    src: List[int] = []
    dst: List[int] = []
    for i in range(neigh.shape[0]):
        for j in neigh[i, 1:]:
            # Add both directions to approximate an undirected graph
            src.append(i)
            dst.append(int(j))
            src.append(int(j))
            dst.append(i)

    return np.asarray([src, dst], dtype=np.int64)


def save_artifacts(path: Path, art: DegnnArtifacts) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(art, path)


def load_artifacts(path: Path) -> DegnnArtifacts:
    return load(path)


# Backwards/patch compatibility: some scripts refer to this name.
def load_degnn_artifacts(path: Path) -> DegnnArtifacts:
    """Alias for :func:`load_artifacts` (kept for script compatibility)."""

    return load_artifacts(path)


def load_degnn_model(
    state_path: Path,
    artifacts: DegnnArtifacts,
    device: Optional[torch.device] = None,
) -> GraphSAGE:
    """Instantiate GraphSAGE and load weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not state_path.exists():
        raise FileNotFoundError(state_path)

    model = GraphSAGE(
        in_dim=int(artifacts.x_node_scaled.shape[1]),
        hidden=int(artifacts.hidden_dim),
        out_dim=2,
        dropout=float(artifacts.dropout),
    ).to(device)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_p_fake_for_new_nodes(
    new_df,
    *,
    artifacts: DegnnArtifacts,
    model: GraphSAGE,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Predict p(fake) for **new** rows using the trained DE‑GNN.

    Parameters
    ----------
    new_df:
        A pandas.DataFrame-like object (must support __getitem__ with
        column names) containing the same feature columns used at training.

    Inference strategy
    ------------------
    We keep the original training graph edges intact.
    For each new node, we add directed edges (train_node -> new_node) from its
    k nearest neighbors in the *edge feature* space.
    This allows the new node to aggregate messages from the existing graph
    without the new node perturbing training node neighborhoods.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build new node feature matrices (node features + edge features)
    missing = [c for c in artifacts.feature_cols if c not in new_df.columns]
    if missing:
        raise KeyError(
            "New samples are missing required DE-GNN feature columns. "
            f"Missing: {missing}\n"
            "Fix: ensure your generation/extraction script outputs the same "
            "feature schema used by Script 09."
        )

    x_new_node = np.asarray(new_df[artifacts.feature_cols], dtype=np.float32)
    x_new_node_scaled = artifacts.scaler_node.transform(
        x_new_node).astype(np.float32)

    x_new_edge = np.asarray(new_df[artifacts.edge_cols], dtype=np.float32)
    x_new_edge_scaled = artifacts.scaler_edge.transform(
        x_new_edge).astype(np.float32)
    x_new_edge_norm = _l2_normalize(x_new_edge_scaled).astype(np.float32)

    # --- kNN from new nodes into the existing reference graph
    nn = NearestNeighbors(n_neighbors=artifacts.knn_k, metric="cosine")
    nn.fit(artifacts.x_edge_norm)
    neigh = nn.kneighbors(x_new_edge_norm, return_distance=False)

    n_ref = int(artifacts.x_node_scaled.shape[0])
    n_new = int(x_new_node_scaled.shape[0])
    total = n_ref + n_new

    # Combined node features
    x_all = np.vstack(
        [artifacts.x_node_scaled, x_new_node_scaled]).astype(np.float32)

    # Combined edges: keep original edges, plus directed edges ref->new
    base_ei = artifacts.edge_index
    src_extra: List[int] = []
    dst_extra: List[int] = []
    for i_new in range(n_new):
        dst_node = n_ref + i_new
        for j_ref in neigh[i_new]:
            src_extra.append(int(j_ref))
            dst_extra.append(int(dst_node))

    if src_extra:
        ei_extra = np.asarray([src_extra, dst_extra], dtype=np.int64)
        ei_all = np.concatenate([base_ei, ei_extra], axis=1)
    else:
        ei_all = base_ei

    # Torch tensors
    x_t = torch.tensor(x_all, dtype=torch.float32, device=device)
    ei_t = torch.tensor(ei_all, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(x_t, ei_t)
        p_fake = torch.softmax(logits, dim=-1)[:, 1]
        p_new = p_fake[n_ref:].detach().cpu().numpy().astype(np.float32)

    return p_new
