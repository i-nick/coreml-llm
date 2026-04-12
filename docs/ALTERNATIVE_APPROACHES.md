# Alternative Approaches — Beyond Speculative Decoding

This is a decision log for alternative speedup directions we considered beyond the
current EAGLE-3 + W8A8 path. Kept for posterity so future iteration has the full
option set on record.

Status legend: **S** = picked, **A** = viable next, **B** = parked, **C** = deferred, **D** = rejected.

---

## #1. Distill Gemma 4 E2B (2.7B) → 1B
Ship a smaller "Turbo" SKU trained to mimic Gemma 4 E2B outputs.

| Axis | Value |
|---|---|
| Status | **A** — parked, revisit after current stack lands |
| Cost | H100 × 1-2 weeks, 5-10 B tokens self-distillation |
| Expected decode cost | ~3× faster per token |
| Projected tok/s (iPhone 17 Pro) | 2K: 90-100, 8K: 45-55 (solo) |
| Compound with EAGLE-3 | 2K: 180+, 8K: 90+ |
| Risk | Quality drop (especially multilingual + code) |
| Differentiation | "1B at Gemma-4 quality" — strong vs Apple Foundation (3B) |
| Why parked | Locks us to a specific distilled model; less reusable as a library recipe |

---

## #2. Sliding-only — drop all 7 full-attention layers
Replace full-attention with sliding window in L4/L9/L14/L19/L24/L29/L34. All 35 layers become W=512 (Gemma 3 architecture).

| Axis | Value |
|---|---|
| Status | **B** — fallback if MQA under-performs |
| Cost | QLoRA 1-2 days on A100 |
| Expected decode cost | ctx-invariant at sliding-layer speed |
| Projected tok/s | 2K: 31 (unchanged), 8K: **31** (from 15) |
| Compound with EAGLE-3 | 2K: 60, 8K: 60 (both equal, the point) |
| Risk | Long-context retrieval quality degrades (needle-in-haystack) |
| Differentiation | Matches Apple's own 3B design direction (5 local + 1 global) |
| Why parked | WFA quality-regression evidence suggests sliding-only will also regress; QLoRA may not fully recover |

---

## #3. MQA for full-attention layers (num_kv 2 → 1) — **SELECTED**
Collapse the 2 KV heads in full-attention layers to 1 via weight averaging, then QLoRA-recover.

| Axis | Value |
|---|---|
| Status | **S** — executing now |
| Cost | Weight surgery (minutes) + optional QLoRA (hours on A100) |
| Expected decode cost | 8K full-attn bandwidth halved → +40% on 8K path |
| Projected tok/s | 2K: 32 (small), 8K: **20-22** (from 15) |
| Compound with EAGLE-3 | 8K: 44-48, +DuoAttention: 60+, +W8A8: 80+ |
| Risk | Quality loss without QLoRA; QLoRA recovery well-documented for GQA→MQA |
| ANE compat | ✅ Static shape change only (num_kv in chunks) |
| Why picked | Smallest-touch change with meaningful wins; fully composable with everything else |

---

## #4. Cascade router (small model for easy tokens, big for hard)
Run a 300M token predictor first; on high-confidence tokens, commit from small model; otherwise fall to Gemma 4.

| Axis | Value |
|---|---|
| Status | **C** — deferred to v0.7+ |
| Cost | 300M model training + routing threshold tuning |
| Expected tok/s | 3-5× average speedup depending on prompt difficulty |
| Risk | Lossy (small model's tokens accepted directly) — hard to QA |
| ANE compat | ✅ Two models co-resident on ANE is fine |
| Why deferred | Breaks "lossless" contract; LongBench gates harder to reason about |

---

## #5. From-scratch ANE-native model
Design architecture from ANE constraints upward: head_dim=128, all sliding, Conv2d projections, no QK-norm (cat-trick RMSNorm native). Train 100B tokens over 2-3 weeks.

| Axis | Value |
|---|---|
| Status | **C** — v1.0+ ambition; gate: after all Gemma-4 work is exhausted |
| Cost | H100 cluster × 2-3 weeks, ~100B tokens |
| Expected tok/s | 2K: 80+, 8K: 50+ (solo, no spec) |
| Compound with EAGLE-3 | 2K: 150+, 8K: 100+ |
| Risk | Highest-investment path, quality takes months to approach Gemma |
| Differentiation | World's first truly ANE-native LLM — unique moat |
| Why deferred | User directive: finish Gemma-4 ceiling first |

---

## Decision rationale (2026-04-12)

Current stack ranked by composable speedup at pure ANE execution:

```
baseline            15 tok/s @ 8K
+ pre-alloc         16   (trivial, from Swift)
+ Q-batch KV-share  17   (~40 LoC)
+ MQA (#3)          22   (QLoRA hours)           ← NEW
+ W8A8              31   (calibration, in flight)
+ DuoAttention      46   (head identification + chunk surgery)
+ EAGLE-3           92   (in training, verify lossless)
```

Realistic ANE overhead correction (×0.65) → **~60 tok/s @ 8K** under pure ANE.

**Logic**:
- #3 (MQA) is picked because it is the smallest-touch change that compounds with everything we have in flight. No speculative-decoding dependence, no W8A8 dependence.
- #1 and #5 are left for later because they commit to a specific model or training pipeline — less reusable as a general CoreML-LLM library recipe.
- #2 is in reserve if MQA quality holds but LongBench still suffers on long contexts.
- #4 is architecturally attractive but lossy — deferred until lossless options are exhausted.

---

## What "Gemma-4 ceiling" means (the gate before #5)

Before we entertain a from-scratch model, we should have landed:

- [ ] #3 MQA + QLoRA recovery
- [ ] W8A8 proper calibration (in flight, bench session)
- [ ] DuoAttention head cap (Tier-2)
- [ ] EAGLE-3 deployed on iPhone (in training + conversion pipeline ready)
- [ ] Pre-alloc masks + KV-share Q-batch in ChunkedEngine (bench session)
- [ ] Prefill on A19 Pro GPU tensor cores (independent TTFT win)
- [ ] StreamingLLM+QLoRA or MLA retrofit evaluated (true 8K quality recovery)

When every item above is decided (shipped or rejected with data), `#5 from-scratch
ANE-native` becomes the next frontier.
