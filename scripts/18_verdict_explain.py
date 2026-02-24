import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Captum for Integrated Gradients
from captum.attr import LayerIntegratedGradients

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_seed_models(model_dirs: List[str]) -> Tuple[AutoTokenizer, List[AutoModelForSequenceClassification]]:
    # Assumes all seeds use same tokenizer config; load tokenizer from first
    tok = AutoTokenizer.from_pretrained(model_dirs[0], use_fast=True)
    models = []
    for d in model_dirs:
        m = AutoModelForSequenceClassification.from_pretrained(d)
        m.eval()
        models.append(m)
    return tok, models


@torch.no_grad()
def _predict_proba_ensemble(models: List[AutoModelForSequenceClassification], inputs: Dict[str, torch.Tensor]) -> float:
    # returns P(fake) averaged across models
    probs = []
    for m in models:
        m.to(inputs["input_ids"].device)
        out = m(**inputs)
        p = torch.softmax(out.logits, dim=-1)[0, 1].item()
        probs.append(p)
    return float(np.mean(probs)), [float(x) for x in probs]


def _heuristics(text: str) -> Dict[str, float]:
    urls = len(URL_RE.findall(text))
    excls = text.count("!")
    qmarks = text.count("?")
    letters = [c for c in text if c.isalpha()]
    caps = [c for c in letters if c.isupper()]
    caps_ratio = (len(caps) / len(letters)) if letters else 0.0
    length = len(text)
    return {
        "url_count": float(urls),
        "exclamation_count": float(excls),
        "question_count": float(qmarks),
        "caps_ratio": float(caps_ratio),
        "char_len": float(length),
    }


def _ig_token_attributions(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    target_class: int = 1,
    max_len: int = 256,
    n_steps: int = 24,
) -> List[Dict[str, float]]:
    """
    Layer Integrated Gradients on the embedding layer.
    Output: top tokens with attribution toward target_class.
    """
    dev = _device()
    model = model.to(dev)
    model.eval()

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    enc = {k: v.to(dev) for k, v in enc.items()}

    input_ids = enc["input_ids"]
    attn_mask = enc.get("attention_mask", None)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # fall back: set pad token to eos (common for GPT-like, but OK here)
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    baseline_ids = torch.full_like(input_ids, fill_value=pad_id)

    def forward_func(ids, mask):
        out = model(input_ids=ids, attention_mask=mask)
        probs = torch.softmax(out.logits, dim=-1)
        return probs[:, target_class]

    lig = LayerIntegratedGradients(forward_func, model.get_input_embeddings())

    attributions, _delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attn_mask,),
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    # aggregate attribution over embedding dims
    scores = attributions.sum(dim=-1).squeeze(0)  # [seq_len]
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

    pairs = []
    for t, s in zip(tokens, scores.detach().cpu().numpy().tolist()):
        # skip special tokens
        if t in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
            continue
        pairs.append((t, float(s)))

    # sort by contribution toward "fake" (positive)
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:20]
    return [{"token": t, "score": float(s)} for t, s in top]


@dataclass
class VerdictResult:
    text: str
    label: str
    fake_prob: float
    credibility_percent: float
    roberta: Dict
    distilbert: Dict
    heuristics: Dict
    top_tokens: List[Dict[str, float]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--seed_list", default="0,1,2,3,4")
    ap.add_argument("--models_base", default="models")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--use_ig_from",
                    choices=["roberta", "distilbert"], default="roberta")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seed_list.split(",") if x.strip()]

    roberta_dirs = [
        os.path.join(args.models_base, "roberta", f"run_{args.run_id}", f"seed_{s}") for s in seeds
    ]
    distilbert_dirs = [
        os.path.join(args.models_base, "distilbert", f"run_{args.run_id}", f"seed_{s}") for s in seeds
    ]

    for d in roberta_dirs + distilbert_dirs:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Missing trained model dir: {d}")

    roberta_tok, roberta_models = _load_seed_models(roberta_dirs)
    dist_tok, dist_models = _load_seed_models(distilbert_dirs)

    dev = _device()
    enc_r = roberta_tok(args.text, truncation=True,
                        max_length=args.max_len, return_tensors="pt")
    enc_r = {k: v.to(dev) for k, v in enc_r.items()}

    enc_d = dist_tok(args.text, truncation=True,
                     max_length=args.max_len, return_tensors="pt")
    enc_d = {k: v.to(dev) for k, v in enc_d.items()}

    roberta_fake, roberta_seed_probs = _predict_proba_ensemble(
        roberta_models, enc_r)
    dist_fake, dist_seed_probs = _predict_proba_ensemble(dist_models, enc_d)

    # Default fusion (until you plug Qlattice fusion here): average detector probs
    fake_prob = float((roberta_fake + dist_fake) / 2.0)
    label = "fake" if fake_prob >= 0.5 else "real"
    credibility = float((1.0 - fake_prob) * 100.0)

    heur = _heuristics(args.text)

    # IG explanation from chosen model: use best seed proxy = seed_0
    if args.use_ig_from == "roberta":
        top_tokens = _ig_token_attributions(
            roberta_models[0], roberta_tok, args.text, target_class=1, max_len=args.max_len)
    else:
        top_tokens = _ig_token_attributions(
            dist_models[0], dist_tok, args.text, target_class=1, max_len=args.max_len)

    out = VerdictResult(
        text=args.text,
        label=label,
        fake_prob=fake_prob,
        credibility_percent=credibility,
        roberta={"fake_prob": roberta_fake, "seed_probs": roberta_seed_probs},
        distilbert={"fake_prob": dist_fake, "seed_probs": dist_seed_probs},
        heuristics=heur,
        top_tokens=top_tokens,
    )

    payload = asdict(out)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print("[OK] Wrote", args.out_json)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
