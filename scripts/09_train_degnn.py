from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

# Allow importing repo-root modules when running `python scripts/...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minerva_degnn import (  # noqa: E402
    GraphSAGE,
    DegnnArtifacts,
    build_knn_graph_edges,
    save_artifacts,
    _l2_normalize,
)

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
OUT_ARTIFACTS = MODEL_DIR / "degnn_artifacts.joblib"
OUT_NODE_NPZ = FEAT_DIR / "degnn_node_repr.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KNN_K = 10
EPOCHS = 60
LR = 1e-3
HIDDEN = 64
DROPOUT = 0.2

# Safety: graph building on very large corpora is expensive.
MAX_NODES = 20000  # set None to disable


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

    # -------------------------
    # Feature schema
    # -------------------------
    exclude = {"id", "label", "dataset", "lang", "split"}
    feature_cols = [c for c in all_df.columns if c not in exclude]
    if not feature_cols:
        raise RuntimeError("No feature columns found in tabular files.")

    # Build graph edges using *semantic-ish* features only (avoid char_len dominating cosine).
    # This approximates "contextual similarity" graphs when explicit repost/reply graphs aren't available.
    edge_cols = [
        c for c in feature_cols
        if c.startswith("r_pca_")
        or c.startswith("d_pca_")
        or c in {"p_roberta_fake", "p_distil_fake"}
    ]
    if len(edge_cols) < 4:
        # Fallback: at least use whatever we have.
        edge_cols = feature_cols.copy()
        print("[WARN] Could not identify PCA/prob columns for edge construction; using ALL features for kNN graph.")

    X_node = all_df[feature_cols].values.astype(np.float32)
    X_edge = all_df[edge_cols].values.astype(np.float32)
    y = all_df["label"].values.astype(np.int64)

    # masks
    split = all_df["split"].values
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    # -------------------------
    # Scaling (critical for cosine kNN)
    # -------------------------
    scaler_node = StandardScaler()
    X_node_scaled = scaler_node.fit_transform(X_node).astype(np.float32)

    scaler_edge = StandardScaler()
    X_edge_scaled = scaler_edge.fit_transform(X_edge).astype(np.float32)
    X_edge_norm = _l2_normalize(X_edge_scaled).astype(np.float32)

    # -------------------------
    # Build kNN graph
    # -------------------------
    edge_index_np = build_knn_graph_edges(X_edge_norm, k=KNN_K)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=DEVICE)

    x_t = torch.tensor(X_node_scaled, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
    train_idx = torch.tensor(np.where(train_mask)[
                             0], dtype=torch.long, device=DEVICE)
    val_idx = torch.tensor(
        np.where(val_mask)[0], dtype=torch.long, device=DEVICE)
    test_idx = torch.tensor(
        np.where(test_mask)[0], dtype=torch.long, device=DEVICE)

    torch.manual_seed(42)
    np.random.seed(42)

    model = GraphSAGE(in_dim=x_t.size(1), hidden=HIDDEN,
                      out_dim=2, dropout=DROPOUT).to(DEVICE)
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

        # Save inference artifacts
        art = DegnnArtifacts(
            feature_cols=feature_cols,
            edge_cols=edge_cols,
            scaler_node=scaler_node,
            scaler_edge=scaler_edge,
            ids=all_df["id"].astype(str).values,
            splits=all_df["split"].astype(str).values,
            y_true=y,
            x_node_scaled=X_node_scaled,
            x_edge_norm=X_edge_norm,
            edge_index=edge_index_np,
            knn_k=int(KNN_K),
            hidden_dim=int(HIDDEN),
            dropout=float(DROPOUT),
        )
        save_artifacts(OUT_ARTIFACTS, art)
        print(f"[OK] Saved DE-GNN artifacts -> {OUT_ARTIFACTS}")

    # final test preds (only for nodes that exist in this subsample)
    model.eval()
    with torch.no_grad():
        logits = model(x_t, edge_index)
        prob_fake = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        pred = logits.argmax(dim=-1).cpu().numpy()

        # Latent node embeddings (for downstream RF / analysis)
        emb = model.encode(x_t, edge_index).cpu().numpy().astype(np.float32)

    out = pd.DataFrame({
        "id": all_df["id"].astype(str),
        "split": all_df["split"],
        "y_true": y,
        "y_pred": pred,
        "p_degnn_fake": prob_fake.astype(float),
    })
    out.to_csv(OUT_PREDS, index=False)
    print(f"[OK] Saved DE-GNN preds -> {OUT_PREDS}")

    # Compact NPZ for embeddings
    np.savez_compressed(
        OUT_NODE_NPZ,
        id=all_df["id"].astype(str).values,
        split=all_df["split"].astype(str).values,
        y=y,
        p_fake=prob_fake.astype(np.float32),
        emb=emb,
        feature_cols=np.array(feature_cols, dtype=object),
        edge_cols=np.array(edge_cols, dtype=object),
    )
    print(f"[OK] Saved DE-GNN node repr -> {OUT_NODE_NPZ}")

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
