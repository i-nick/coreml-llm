#!/usr/bin/env python3
"""Convert a Gemma 4 E2B checkpoint from GQA (num_kv>1) to MQA (num_kv=1).

Motivation: full-attention layers dominate the 8K decode path (memory-
bandwidth-bound). Halving the KV heads halves the KV read per step for those
layers, for an expected +40% on 8K tok/s once the chunks are rebuilt.

Scope (by default):
  - ONLY modify `full_attention` layers (identified via config.layer_types).
  - The 20 KV-shared layers (L15-34) don't own K/V projections — they read
    from whichever earlier layer owns KV. When we shrink L14's K/V, all
    its readers automatically see the shrunken cache. No separate surgery
    needed for KV-shared layers.

Weight collapse rule: for each (layer, kv_head), reshape K/V proj weight
  (num_kv * head_dim, hidden) → (num_kv, head_dim, hidden)
and take the simple mean across the num_kv axis → (1 * head_dim, hidden).
Mean is chosen because it preserves expectation under random grouping; this
is the standard initialization for MQA distillation recovery.

Output:
  - modified HF-compatible safetensors
  - modified config.json (num_key_value_heads updated for affected layer-types)
  - a small report (JSON) summarizing what changed

Usage:
    python conversion/convert_to_mqa.py \\
        --model-id google/gemma-4-E2B-it \\
        --output ./gemma-4-E2B-it-mqa \\
        --apply-to full_attention

    # Follow up: QLoRA-finetune to recover quality (see
    # conversion/finetune_mqa_recovery.ipynb).

Sanity-check after conversion:
    python - <<'PY'
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained("./gemma-4-E2B-it-mqa", torch_dtype="float16")
    t = AutoTokenizer.from_pretrained("./gemma-4-E2B-it-mqa")
    ids = t.encode("Hello", return_tensors="pt")
    print(m.generate(ids.to(m.device), max_new_tokens=10))
    PY
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch


def infer_layer_kv_widths(cfg):
    """Return dict layer_type -> num_kv_heads_original (int)."""
    # Gemma4TextConfig keeps num_key_value_heads as a single int for all layer types.
    # We mutate it to 1 in the output config.
    return {
        "full_attention":    cfg.num_key_value_heads,
        "sliding_attention": cfg.num_key_value_heads,
    }


def collapse_kv(weight: torch.Tensor, num_kv: int, head_dim: int, target_kv: int = 1) -> torch.Tensor:
    """Shrink K or V projection weight from num_kv heads to target_kv heads.

    weight shape: (num_kv * head_dim, hidden). Returns (target_kv * head_dim, hidden).
    Assumes num_kv is divisible by target_kv (standard case).
    """
    if num_kv == target_kv:
        return weight
    assert num_kv % target_kv == 0, f"num_kv={num_kv} not divisible by target_kv={target_kv}"
    H = weight.shape[1]
    w = weight.view(num_kv, head_dim, H)               # (num_kv, head_dim, H)
    group = num_kv // target_kv                         # heads per output group
    w = w.view(target_kv, group, head_dim, H).mean(1)   # average within group
    return w.reshape(target_kv * head_dim, H).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--apply-to", type=str, default="full_attention",
                    help="Comma-separated layer_types to convert to MQA. "
                         "Use 'all' to convert every layer.")
    ap.add_argument("--target-kv", type=int, default=1,
                    help="Target num_kv_heads for converted layers (default 1 = MQA).")
    args = ap.parse_args()

    print(f"Loading {args.model_id}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    cfg = AutoConfig.from_pretrained(args.model_id)
    tcfg = cfg.text_config if hasattr(cfg, "text_config") else cfg

    # Inspect layer_types
    layer_types = getattr(tcfg, "layer_types", None)
    if layer_types is None:
        raise RuntimeError("model has no `layer_types` — not a Gemma 4-style hybrid model")
    num_layers = len(layer_types)
    head_dim = getattr(tcfg, "head_dim", tcfg.hidden_size // tcfg.num_attention_heads)
    num_kv = tcfg.num_key_value_heads
    first_kv_shared = num_layers - getattr(tcfg, "num_kv_shared_layers", 0)
    print(f"  num_layers={num_layers}, head_dim={head_dim}, num_kv_heads={num_kv}")
    print(f"  layer_types sample: {layer_types[:10]} ...")
    print(f"  kv-shared start:   L{first_kv_shared} (layers >= this read from earlier)")

    if args.target_kv >= num_kv:
        raise ValueError(f"target_kv={args.target_kv} must be < current num_kv={num_kv}")

    # Selection
    apply_set = set(t.strip() for t in args.apply_to.split(",")) if args.apply_to != "all" else set(layer_types)
    print(f"  applying MQA ({num_kv}→{args.target_kv}) to layer types: {apply_set}")

    # Load model
    print(f"\nLoading weights (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16,
                                                 device_map="cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Walk layers, modify K/V projections on matching layer_types (and non-KV-shared)
    tm = model.model
    if hasattr(tm, "language_model"): tm = tm.language_model
    if hasattr(tm, "model"): tm = tm.model
    layers = tm.layers

    modified = []
    for idx, layer in enumerate(layers):
        if layer_types[idx] not in apply_set: continue
        if idx >= first_kv_shared:
            # KV-shared: no own k_proj/v_proj, nothing to collapse here
            continue
        attn = layer.self_attn
        if not hasattr(attn, "k_proj") or not hasattr(attn, "v_proj"):
            # This layer doesn't own its K/V
            continue

        with torch.no_grad():
            new_k = collapse_kv(attn.k_proj.weight.data, num_kv, head_dim, args.target_kv)
            new_v = collapse_kv(attn.v_proj.weight.data, num_kv, head_dim, args.target_kv)

        new_k_proj = torch.nn.Linear(new_k.shape[1], new_k.shape[0], bias=False, dtype=torch.float16)
        new_v_proj = torch.nn.Linear(new_v.shape[1], new_v.shape[0], bias=False, dtype=torch.float16)
        new_k_proj.weight.data.copy_(new_k)
        new_v_proj.weight.data.copy_(new_v)
        attn.k_proj = new_k_proj
        attn.v_proj = new_v_proj
        modified.append({"layer_idx": idx, "layer_type": layer_types[idx],
                         "kv_before": num_kv, "kv_after": args.target_kv})

    # Update config — note this globally changes num_key_value_heads which affects
    # ALL layers including the ones we did NOT modify. To avoid breaking them,
    # we only flip the global config if the user asked for `all` layer-types.
    # Otherwise we need a per-layer-type override which Gemma4 config doesn't support
    # directly. Practical workaround: require --apply-to=all, OR accept that the
    # transformers runtime will error if non-modified layers disagree with config.
    if args.apply_to == "all":
        tcfg.num_key_value_heads = args.target_kv
    else:
        # Cannot globally mutate. Emit a warning and keep original config — the
        # resulting checkpoint will be self-inconsistent for transformers runtime.
        # This is OK for our downstream: the conversion pipeline into mlpackages
        # reads per-layer weight shapes, not config.num_key_value_heads.
        print("\nWARN: partial MQA (--apply-to != 'all'). config.num_key_value_heads "
              "is NOT updated to preserve compatibility of unmodified layers. "
              "The output cannot be loaded by vanilla transformers; it is intended "
              "only for the CoreML conversion pipeline that reads per-layer "
              "weight shapes directly.")

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {out_dir}...")
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    # Report
    report = {
        "source_model": args.model_id,
        "apply_to":     args.apply_to,
        "target_kv":    args.target_kv,
        "num_kv_before": num_kv,
        "head_dim":      head_dim,
        "num_layers":    num_layers,
        "modified_layers": modified,
        "kv_shared_start": first_kv_shared,
        "notes": [
            "Weight collapse: mean across grouped KV heads.",
            "KV-shared layers (L >= kv_shared_start) are not modified directly; they "
            "inherit the shrunken KV from their anchor layer (e.g., L14).",
        ],
    }
    with open(out_dir / "mqa_conversion_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  modified {len(modified)} layer(s)")
    for m in modified:
        print(f"    L{m['layer_idx']:2d} ({m['layer_type']}): kv {m['kv_before']}->{m['kv_after']}")
    print(f"  report: {out_dir / 'mqa_conversion_report.json'}")

    print("\nNext steps:")
    print("  1. (optional but recommended) QLoRA-recover quality:")
    print("     conversion/finetune_mqa_recovery.ipynb")
    print("  2. Re-run the target chunk conversion pipeline on the new checkpoint")
    print("     (bench session's build_speculative.py path).")
    print("  3. Benchmark on iPhone to measure actual 8K tok/s uplift.")


if __name__ == "__main__":
    main()
