from __future__ import annotations

from pathlib import Path
import sys
import json

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

GEN_DIR = Path("models/gpt2_tagalog_finetuned")
DETECTOR_DIR = Path("models/roberta_finetuned")

OUT_DIR = Path("generated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "gpt2_synthetic_samples.jsonl"

REAL_TOKEN = "<|label=real|>"
FAKE_TOKEN = "<|label=fake|>"


@torch.no_grad()
def prob_fake(det_model, det_tok, text: str, max_len: int = 256) -> float:
    enc = det_tok(text, truncation=True,
                  max_length=max_len, return_tensors="pt")
    enc = {k: v.to(det_model.device) for k, v in enc.items()}
    logits = det_model(**enc).logits
    p = F.softmax(logits, dim=-1)[0, 1].item()
    return float(p)


def main() -> None:
    if not GEN_DIR.exists():
        raise FileNotFoundError(
            f"Missing {GEN_DIR}. Run 11_train_gpt2MINERVA.py first.")
    if not DETECTOR_DIR.exists():
        raise FileNotFoundError(
            f"Missing {DETECTOR_DIR}. Run 04_train_robertaMINERVA.py first.")

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    target = sys.argv[2].strip().lower() if len(sys.argv) > 2 else "fake"
    min_conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.70
    max_new_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 120

    if target not in {"fake", "real"}:
        raise ValueError("target must be 'fake' or 'real'")

    prompt = FAKE_TOKEN if target == "fake" else REAL_TOKEN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_tok = AutoTokenizer.from_pretrained(str(GEN_DIR), use_fast=True)
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(
        str(GEN_DIR)).to(device).eval()

    det_tok = AutoTokenizer.from_pretrained(str(DETECTOR_DIR), use_fast=True)
    det_model = AutoModelForSequenceClassification.from_pretrained(
        str(DETECTOR_DIR)).to(device).eval()

    kept = 0
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for _ in range(n):
            inp = gen_tok(prompt, return_tensors="pt").to(device)
            out_ids = gen_model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=gen_tok.eos_token_id,
            )
            text = gen_tok.decode(out_ids[0], skip_special_tokens=True).strip()

            pf = prob_fake(det_model, det_tok, text)

            if target == "fake" and pf < min_conf:
                continue
            if target == "real" and pf > (1.0 - min_conf):
                continue

            kept += 1
            f.write(json.dumps({
                "id": kept,
                "tag": "SYNTHETIC_FOR_SIMULATION",
                "target_label": target,
                "detector_prob_fake": pf,
                "text": text,
            }, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote {kept} validated samples -> {OUT_FILE.resolve()}")
    print("Note: file is intentionally under generated/ (gitignored).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)
