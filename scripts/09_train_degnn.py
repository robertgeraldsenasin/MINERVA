from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

FEAT_DIR = Path("data/features")
TRAIN = FEAT_DIR / "train_tabular.csv"
VAL = FEAT_DIR / "val_tabular.csv"
TEST = FEAT_DIR / "test_tabular.csv"

MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

OUT_MODEL = MODEL_DIR / "degnn.pt"
OUT_PREDS = FEAT_DIR / "degnn_preds.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KNN_K = 10
EPOCHS = 60
LR = 1e-3
HIDDEN = 64
DROPOUT = 0.2

# Safety: graph building on very large corpora is expensive.
MAX_NODES = 20000  # set None to disable


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.lin_self1 = nn.Linear(in_dim, hidden)
        self.lin_neigh1 = nn.Linear(in_dim, hidden)
        self.lin_self2 = nn.Linear(hidden, hidden)
        self.lin_neigh2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def sage_layer(self, x, edge_index, lin_self, lin_neigh):
        src, dst = edge_index  # src -> dst
        neigh_sum = torch.zeros_like(x)
        neigh_sum.index_add_(0, dst, x[src])
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        neigh_mean = neigh_sum / deg.clamp(min=1).unsqueeze(1)
        h = lin_self(x) + lin_neigh(neigh_mean)
        return F.relu(h)

    def forward(self, x, edge_index):
        h = self.sage_layer(x, edge_index, self.lin_self1, self.lin_neigh1)
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = self.sage_layer(h, edge_index, self.lin_self2, self.lin_neigh2)
        h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.out(h)


def f1_binary(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary")
    return p, r, f1


def main():
    for p in [TRAIN, VAL, TEST]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run 06_extract_features.py first.")

    train_df = pd.read_csv(TRAIN)
    val_df = pd.read_csv(VAL)
    test_df = pd.read_csv(TEST)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # subsample for graph practicality
    if MAX_NODES is not None and len(all_df) > MAX_NODES:
        all_df = all_df.sample(
            n=MAX_NODES, random_state=42).reset_index(drop=True)
        print(f"[INFO] Subsampled to {len(all_df)} nodes for DE-GNN.")

    feature_cols = [c for c in all_df.columns if c not in {
        "id", "label", "dataset", "lang", "split"}]
    X = all_df[feature_cols].values.astype(np.float32)
    y = all_df["label"].values.astype(np.int64)

    # masks
    split = all_df["split"].values
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    # Build kNN graph on features
    nn_model = NearestNeighbors(n_neighbors=KNN_K + 1, metric="cosine")
    nn_model.fit(X)
    neigh = nn_model.kneighbors(X, return_distance=False)

    # Build directed edges i -> neighbor
    src = []
    dst = []
    for i in range(neigh.shape[0]):
        for j in neigh[i, 1:]:  # skip self
            src.append(i)
            dst.append(j)
            src.append(j)
            dst.append(i)  # make undirected by adding both

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=DEVICE)

    x_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
    train_idx = torch.tensor(np.where(train_mask)[
                             0], dtype=torch.long, device=DEVICE)
    val_idx = torch.tensor(
        np.where(val_mask)[0], dtype=torch.long, device=DEVICE)
    test_idx = torch.tensor(
        np.where(test_mask)[0], dtype=torch.long, device=DEVICE)

    model = GraphSAGE(in_dim=x_t.size(1), hidden=HIDDEN, out_dim=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        logits = model(x_t, edge_index)
        loss = F.cross_entropy(logits[train_idx], y_t[train_idx])

        opt.zero_grad()
        loss.backward()
        opt.step()

        # eval
        model.eval()
        with torch.no_grad():
            logits = model(x_t, edge_index)
            val_pred = logits[val_idx].argmax(dim=-1).cpu().numpy()
            val_true = y_t[val_idx].cpu().numpy()
            vp, vr, vf1 = f1_binary(val_true, val_pred)

        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} loss={loss.item():.4f} val_f1={vf1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, OUT_MODEL)
        print(f"[OK] Saved best DE-GNN -> {OUT_MODEL}")

    # final test preds (only for nodes that exist in this subsample)
    model.eval()
    with torch.no_grad():
        logits = model(x_t, edge_index)
        pred = logits.argmax(dim=-1).cpu().numpy()

    out = pd.DataFrame({"id": all_df["id"].astype(
        str), "split": all_df["split"], "y_true": y, "y_pred": pred})
    out.to_csv(OUT_PREDS, index=False)
    print(f"[OK] Saved DE-GNN preds -> {OUT_PREDS}")

    # report test metrics
    test_rows = out[out["split"] == "test"]
    if len(test_rows) > 0:
        acc = accuracy_score(test_rows["y_true"], test_rows["y_pred"])
        p, r, f1 = f1_binary(test_rows["y_true"], test_rows["y_pred"])
        print(
            f"[TEST] acc={acc:.4f} precision={p:.4f} recall={r:.4f} f1={f1:.4f}")
    else:
        print("[WARN] No test rows in the current graph subsample.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)
