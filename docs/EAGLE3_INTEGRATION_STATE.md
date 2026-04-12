# EAGLE-3 Integration — Resumable Session State

Working document for the ongoing EAGLE-3 integration on this MacBook Air (M3 16GB). Written so a new session can pick up from here without re-deriving context.

**Last updated:** 2026-04-12. Branch: `feature/audio-support`.

---

## Training artifacts (source of truth)

| File | Path | Notes |
|---|---|---|
| `eagle3_draft_best.pt` | `/Users/daisukemajima/Downloads/eagle3_draft/` | 188MB, 47.2M params |
| `eagle3_config.json` | same dir | `fusion_layers=[8,17,34]`, `hidden=1536`, `num_heads=8`, `num_kv_heads=1`, `head_dim=256`, `ffn=6144`, `embed_scale=39.1918...`, `ttt_k=3`, `model_id=google/gemma-4-E2B-it` |
| `eagle3_eval.json` | same dir | **acc[0]=74.94%, acc[1]=40.6%, acc[2]=23.9%, expL=2.13** — well above §3.1 gates (≥55% / ≥2.0). Projection 30→63.8 tok/s. |
| `eagle3_training.log` | same dir | 2 epochs × ~30k samples on Colab |

Use `best.pt`, not `step4000.pt` or `final.pt` — best has highest acc[0].

---

## Converted CoreML artifacts (ready for iPhone)

All under `/Users/daisukemajima/Downloads/CoreML-LLM/output/`:

| File | Size | Outputs (incl. EAGLE-3 additions) | Built by |
|---|---:|---|---|
| `eagle3_draft.mlpackage` | 210MB | `h_out`, `token` (int32 scalar), `logit` (fp16) | `build_eagle3.py` |
| `eagle3_fusion.mlpackage` | 14MB | `h_fused` | `build_eagle3.py` |
| `eagle3-chunks/chunk1.mlpackage` | 149MB | (unchanged, no fusion layer) | `build_eagle3_chunks.py` |
| `eagle3-chunks/chunk2.mlpackage` | 128MB | existing + **`hidden_at_L8` (1,1,1536) fp16** | `build_eagle3_chunks.py` |
| `eagle3-chunks/chunk3.mlpackage` | 311MB | existing + **`hidden_at_L17` (1,1,1536) fp16** | `build_eagle3_chunks.py` |
| `eagle3-chunks/chunk4.mlpackage` | 503MB | `token_id`, `token_logit`, **`hidden_at_L34`** (pre-norm, 1,1,1536 fp16) | `build_eagle3_chunks.py` |

All INT4 palettized (group_size=32). Output dtype 65552 = `MLMultiArrayDataType.float16`.

Smoke-tested with `/tmp/smoke_eagle3_chunks.py`: all output names / shapes verified.

---

## Environment gotchas (things that will bite if ignored)

### Python version
- System Python 3.9.6 does **not** work with coremltools 8+/9+. Need 3.10-3.12.
- Installed: `brew install python@3.11` → `/opt/homebrew/bin/python3.11` (Python 3.11.15).

### Venv and pinned deps
- `conversion/.venv/` is the active venv. Activate with `source conversion/.venv/bin/activate` from repo root.
- `requirements.txt` pins `torch==2.11.0` but the monolithic path bug surfaces with `torch==2.7.0` too — the bug is not a torch version issue (see below).
- `accelerate` is NOT in requirements.txt but is required by `transformers==5.5.0` when using `device_map`. Installed ad hoc: `pip install accelerate`.

### Current installed torch: 2.7.0
Downgraded from 2.11 to try to fix convert.py (didn't help — bug is elsewhere). If you bump torch back up, verify `build_eagle3_chunks.py` still traces cleanly.

### HF access
- `google/gemma-4-E2B-it` is **not** gated (at time of writing) — anonymous DL works even without `HF_TOKEN`. The "unauthenticated requests" warning is cosmetic.
- Gemma 4 model already cached at `~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/` (~5.5GB) and copied to `output/gemma4-e2b/hf_model/` for `Gemma4Model.from_pretrained(HF_DIR)` to use.

### convert.py is broken for Gemma 4 monolithic — **do not use**
- Error: `gemma4_wrapper.py:107` passes (1,1,1536) hidden directly to a Conv2d(1536, 8960, 1) without the NCHW permute (`.permute(0,2,1).unsqueeze(2)`). Fails trace with "expected input[1, 1, 1, 1536] to have 1536 channels, but got 1".
- **Not needed for EAGLE-3 work.** Use `build_eagle3_chunks.py` (bypasses the broken wrapper, uses `SWAChunk1-4` directly which have correct NCHW handling).
- If someone wants to fix it: wrap hidden_states in `.permute(0,2,1).unsqueeze(2)` before the Conv2d, then `.squeeze(2).permute(0,2,1)` after.

### test_eagle3_infer.py Mac-compat patches (already applied)
1. `apply_rope` now casts cos/sin to `x.dtype` (was fp32, would break fp16 q/k).
2. Draft cast to fp16 after loading (was fp32, target forward is fp16 → dtype mismatch on fusion).
3. Pass `--device cpu` or `--device mps`. **MPS OOMs on M3 16GB** (9.54 GiB single alloc). Use `cpu` — slow but works for sanity.

### Sanity check result (for reference)
```
[1/2] target-only greedy generation ... 1.11 tok/s
[2/2] EAGLE-3 speculative (K=3) ....... 0.97 tok/s
  draft accept rate: 33.3% of 3 proposals per step
  outputs match: True    ← THIS IS THE GATE, not speedup (CPU is apples-to-oranges vs ANE)
```

---

## Remaining work

### Phase 2A — Verify chunks (N=3) — NOT STARTED
Blocker: `_run_layer_swa` is hardcoded for T=1 (line 70 of `gemma4_swa_chunks.py`: `q.view(1, num_heads, hd, 1)` assumes T=1). Verify chunks need T=3 to batch-verify K=3 candidates.

Two approaches:
1. **Rewrite _run_layer_swa to handle variable T.** Cleanest but touches the core decode path — risk of regressing the shipped T=1 code.
2. **Build verify chunks on top of `_run_layer_prefill`** (in `gemma4_prefill_chunks.py`) which already handles T>1 (used for batched prefill, N=64/512). Build with sample shapes N=3, keep K/V I/O explicit so Swift can discard rejected updates.

Recommend **approach 2** (less risk).

Files to create: `conversion/build_eagle3_verify.py`. Outputs:
- `verify_chunk1.mlpackage` — SWAChunk1 at T=3
- `verify_chunk2.mlpackage` — + `hidden_at_L8` at last position (or all 3 positions)
- `verify_chunk3.mlpackage` — + `hidden_at_L17`
- `verify_chunk4.mlpackage` — + `hidden_at_L34` (pre-norm), + `token_ids` per position (shape (1,3) int32)

### Phase 2B — Swift integration in ChunkedEngine
`SpeculativeLoop.swift` is already written (see file in Sources/CoreMLLLM/). It expects a `SpeculativeTarget`-conforming object. ChunkedEngine needs:

1. **Store `hidden_at_L{8,17,34}`** after each decode step. Modify `decodeStep()` (ChunkedEngine.swift around line 340-386) to fetch these outputs from chunk2/3/4 and stash in 3 ivars.

2. **Conform to `SpeculativeTarget`**:
   ```swift
   func lastHiddenMulti(at indices: [Int]) throws -> [MLMultiArray] {
       // Match indices to lastHiddenAtL8 / L17 / L34
   }
   func commitAccepted(_ tokens: [Int32]) throws {
       // For each token, run decodeStep(tokenID:) — advances position + updates KV
       // Last iteration naturally refreshes the lastHiddenAtL* ivars
   }
   func verifyCandidates(_ candidates: [Int32], K: Int) throws -> [Int32] {
       // Needs verify_chunk1..4 (Phase 2A).
       // Until Phase 2A lands: throw SpeculativeError.verifyFailed("verify chunks not built")
   }
   ```

3. **Make CoreMLLLM public API** decide when to use speculative (based on `SpeculativeLoop.shouldSpeculate` + rolling acceptance). See `SpeculativeLoop.swift:194`.

### Phase 3 — iPhone deployment + bench
1. Compile `.mlpackage` → `.mlmodelc` (either via Xcode "Add to target" or at runtime via `MLModel.compileModel(at:)`).
2. Replace existing iPhone `chunk1/2/3/4.mlmodelc` with EAGLE-3 versions. Existing chunks lack `hidden_at_L*` outputs → would crash Swift that expects them, so this is an all-or-nothing swap.
3. Bench on iPhone 17 Pro, thermal-stable 10-min, K=1 (baseline) vs K=3 (EAGLE-3). Target per docs/SPEED_8K.md §3 P1: ctx=2048 at 55-70 tok/s, ctx=8192 at ~30 tok/s.

---

## Command cheat sheet (Mac)

```bash
# Always start from repo root:
cd /Users/daisukemajima/Downloads/CoreML-LLM
source conversion/.venv/bin/activate

# Sanity check (CPU, slow but validates match)
python conversion/test_eagle3_infer.py \
    --ckpt /Users/daisukemajima/Downloads/eagle3_draft/eagle3_draft_best.pt \
    --prompt "The capital of Japan is" --max-new 16 --K 3 --device cpu

# Rebuild fusion + draft mlpackages (≈3 min)
python conversion/build_eagle3.py \
    --ckpt /Users/daisukemajima/Downloads/eagle3_draft/eagle3_draft_best.pt \
    --output ./output/eagle3_draft.mlpackage \
    --fusion-output ./output/eagle3_fusion.mlpackage \
    --palettize-int4

# Rebuild all 4 decode chunks (≈20 min total)
python conversion/build_eagle3_chunks.py --output ./output/eagle3-chunks
# Or one at a time: --only chunk2

# Smoke-test output contracts
python /tmp/smoke_eagle3_chunks.py
```

---

## Files I own in this work

| File | What |
|---|---|
| `conversion/build_eagle3_chunks.py` | New — builds decode chunks with hidden taps |
| `conversion/test_eagle3_infer.py` | Patched for Mac (apply_rope dtype, draft fp16 cast, HF_DIR fallback) |
| `conversion/build_speculative.py` | Patched `HF_DIR` to env var / `../output/gemma4-e2b/hf_model` fallback |
| `docs/EAGLE3_INTEGRATION_STATE.md` | This file |

`SpeculativeLoop.swift` was already in place; unchanged in this session.

---

## Quick validation to run first in a new session

```bash
cd /Users/daisukemajima/Downloads/CoreML-LLM
source conversion/.venv/bin/activate
ls -la output/eagle3_*.mlpackage output/eagle3-chunks/*.mlpackage
# Should show 2 + 4 mlpackages, total ≈1.4GB
python /tmp/smoke_eagle3_chunks.py  # should print PASS
```

If that passes, the Mac-side conversion work is intact and you can move to Phase 2A (verify chunks) or 2B (Swift).
