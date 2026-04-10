# CoreML-LLM

**On-device LLMs on Apple Neural Engine** — Run **Gemma 4** on iPhone with CoreML, ANE-optimized, no server.

CoreML-LLM targets the **Apple Neural Engine** rather than the GPU, making it a good fit for always-on, battery-friendly inference. [MLX Swift](https://github.com/ml-explore/mlx-swift) is the best choice when you want maximum throughput from the GPU; CoreML-LLM is the answer when you want the LLM to live on the ANE so the GPU stays free.

> **v0.5.0** — 31 tok/s decode (+11%), 154 tok/s prefill (+60%), IOSurface KV cache, vectorized embeddings. See [What's new](#whats-new-in-v050).

| Text | Multimodal |
|------|------------|
| ![text](https://github.com/user-attachments/assets/67584300-ce34-4aa5-b3bd-5521cfe8855a) | ![multimodal](https://github.com/user-attachments/assets/2a869bf5-8315-422d-8b06-a4a7edecd173) |

## Performance (Gemma 4 E2B, iPhone 17 Pro)

| | v0.1.0 | v0.2.0 | v0.3.0 | v0.4.0 | **v0.5.0** |
|---|---:|---:|---:|---:|---:|
| Context length | 512 | 2048 | 2048 | 2048 | **2048** |
| Decode speed | ~11 tok/s | ~11 tok/s | ~28 tok/s | ~28 tok/s | **~31 tok/s** |
| Prefill | ~11 tok/s | ~175 tok/s | ~96 tok/s | ~96 tok/s | **~154 tok/s** |
| Multimodal (image) | — | — | broken | working | **working** |
| ANE placement | — | — | 99.78% | 99.78% | **99.78%** |
| Memory (`phys_footprint`) | — | — | — | ~1 GB | **~1 GB** |

Ground-truth ANE placement measured via `MLComputePlan` (7,294 / 7,310 dispatched LLM ops on ANE). Vision encoder runs on GPU by design.

> **Memory:** ~1 GB `phys_footprint` (the iOS jetsam basis), measured via `task_vm_info`. Previous versions of this README quoted ~250 MB from Xcode's memory gauge, which underreports when CoreML loads INT4 palettized weights. The actual number is ~873 MB after load, ~981 MB during inference. `os_proc_available` remains ~5 GB on iPhone 17 Pro (8 GB RAM).

## Pre-converted Models

| Model | Size | Multimodal | Download |
|-------|------|------------|----------|
| **Gemma 4 E2B** | 2.7 GB | Image + Text | [HuggingFace](https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml) |
| Qwen2.5-0.5B | 302 MB | Text only | [HuggingFace](https://huggingface.co/mlboydaisuke/qwen2.5-0.5b-coreml) |

The iOS sample app downloads models automatically. You can also convert your own.

## Quick Start

### iOS App

```bash
open Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj
```

Set your development team → build to an iOS 18+ device → **Get Model** → Download → Chat.

The app uses `.cpuAndNeuralEngine` to force ANE execution.

### Swift Package

```swift
dependencies: [
    .package(url: "https://github.com/john-rocky/CoreML-LLM", from: "0.5.0"),
]
```

```swift
import CoreMLLLM

let llm = try await CoreMLLLM.load(from: modelDirectory)
let answer = try await llm.generate("What is the capital of France?")
// → "The capital of France is **Paris**."

// Streaming
for await token in try await llm.stream("Tell me a story") {
    print(token, terminator: "")
}

// Multimodal (Gemma 4)
let caption = try await llm.generate("Describe this image", image: cgImage)
```

Auto-detects model layout: chunked SWA (Gemma 4 E2B) or monolithic (Qwen2.5).

### Convert a Model

```bash
cd conversion
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install scikit-learn

# Qwen2.5-0.5B (~2 min)
python convert.py --model qwen2.5-0.5b --output ./output/qwen2.5-0.5b

# Gemma 4 E2B (~15 min, 10 GB download)
python convert.py --model gemma4-e2b --output ./output/gemma4-e2b

# List available models
python convert.py --list
```

## What's new in v0.5.0

### Decode +11%, Prefill +60%
Two optimizations that required no model changes — pure Swift-side improvements:

1. **Vectorized embedding lookup** — Replaced scalar INT8→FP16 dequantization loop (10,496 iterations per token) with Accelerate SIMD pipeline: `vDSP.convertElements` → `vDSP.multiply` → `vImageConvert_PlanarFtoPlanar16F`. Embedding time dropped from 2.0ms to 0.4ms per token. Prefill benefits even more since it processes hundreds of embeddings per call.

2. **IOSurface-backed KV cache** — SWA KV cache buffers use IOSurface-backed `CVPixelBuffer` + `MLMultiArray(pixelBuffer:shape:)` for zero-copy CPU↔ANE data transfer.

### Parallel chunk loading
All 8 model chunks (4 decode + 4 prefill) load concurrently via `withThrowingTaskGroup`. First-run ANE compilation is pipelined across chunks.

### Research documentation
New [docs/RESEARCH.md](docs/RESEARCH.md) with comprehensive mobile LLM competitive analysis, ANE hardware internals, and optimization findings.

## What's new in v0.4.0

### Multimodal image understanding
Gemma 4 E2B can now describe images on iPhone. The vision encoder runs on GPU (`.cpuAndGPU`), produces 256 soft tokens projected to the LLM's hidden space, and the features are injected at `<|image|>` placeholder positions during prefill/decode.

Two bugs were preventing this from working:

1. **PLE corruption at image positions** — Per-Layer Embedding was being looked up from the PAD/IMAGE token IDs (norm ~94 each) instead of being zeroed for image positions. All 256 image positions received garbage PLE that corrupted the model's internal state. Fix: set `per_layer_raw = zeros` for any position where the hidden state comes from the vision encoder.

2. **Multi-turn prompt duplication** — `buildPrompt()` inserted 256 image placeholders into every user message when `hasImage` was true. On the second turn, the first message's 256 placeholders would consume all vision features, leaving the second message's placeholders with no features. Fix: image tokens are now inserted only for the last user message.

Image features are cached across conversation turns so follow-up questions about a previously sent image work without re-attaching it. Cache clears on "Clear".

See [docs/MULTIMODAL.md](docs/MULTIMODAL.md) for the full architecture and debugging notes.

### Memory diagnostics
New "Mem" button in the sample app reports `task_vm_info` (`phys_footprint`, `resident_size`, `compressed`) and `os_proc_available_memory()` instantly, without the heavy `MLComputePlan` load that the "ANE?" button requires.

### Memory correction
Previous versions quoted ~250 MB memory usage from Xcode's gauge. Actual `phys_footprint` (iOS jetsam basis) is **~1 GB** — INT4 palettized model weights are counted in `phys_footprint` but may not appear in Xcode's gauge. Corrected in this release. Thanks to community feedback for flagging this.

## What's new in v0.3.0

- **Prefill fp16 overflow fix** — `q_norm` pre-scaling overflowed `Q @ K^T` in fp16. Reverted to manual attention with scale=1.0.
- **Decode ~2.5× faster** — 11 → 28 tok/s (side effect of the rebuild).
- **Prefill window 64 → 512** — multimodal prompts fit in a single prefill pass.
- **ANE placement verified** — 99.78% via `MLComputePlan` on iPhone.
- **Swift Package works** — `CoreMLLLM.load(from:)` auto-detects chunked vs monolithic models.
- **Battery benchmark** — "Bench" menu for sustained generation + SoC drain tracking.

## Carried over from v0.2.0

- **Sliding Window Attention** — 28 sliding (W=512) + 7 full-attention layers.
- **Per-Layer Embedding on ANE** — 8960×1536 projection inside the CoreML graph.
- **Context length 2048** — stateless KV cache with explicit I/O.

## Architecture

```
      Prompt  ─┐
               ▼
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │ Prefill ch1  │─► │ Prefill ch2  │─► │ Prefill ch3  │─► │ Prefill ch4  │─► first token
        │ L0-7  + PLE  │   │ L8-14, kv13/ │   │ L15-24 shared│   │ L25-34 + LM  │
        └──────────────┘   │  kv14 out    │   └──────────────┘   └──────────────┘
               │           └──────────────┘          ▲                  ▲
               │                  │                  │                  │
               │                  └────────────┬─────┴──────────────────┘
               │                                          kv13_k/v, kv14_k/v
               ▼                                              (shared)
          writes K/V to persistent SWA caches
                      │
                      ▼  (decode loop, 1 token per step)
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │  Decode ch1  │─► │  Decode ch2  │─► │  Decode ch3  │─► │  Decode ch4  │─► next token
        └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

### ANE Optimizations

| Technique | What | Why |
|-----------|------|-----|
| ANERMSNorm | `cat([x,-x])` → LayerNorm → slice | ANE has optimized LayerNorm; bare RMSNorm is slow |
| Conv2d Linear | `nn.Linear` → `nn.Conv2d(1)` | ANE executes Conv2d ~3× faster than MatMul |
| In-Model Argmax | Argmax inside the CoreML graph | Avoids shipping 256K logits from ANE to CPU |
| Manual softmax | `max/sub/exp/sum/div` with explicit fp16 casts | Prevents PyTorch's silent fp16→fp32 upcast in `torch.exp` |
| Pre-computed RoPE | cos/sin as model inputs, looked up in Swift | Eliminates `gather`/`greater_equal` (int ops → CPU) |
| Explicit KV I/O | Plain tensor inputs/outputs, no `MLState` | Avoids int64 state indices that break ANE placement |
| Sliding Window | Shift-based cache for 28/35 layers | O(W) per step instead of O(ctx) |
| Batched Prefill | One CoreML call for up to 512 tokens | Order-of-magnitude faster TTFT vs per-token |
| PLE in graph | Conv2d projection + per-layer norm | 8 ms → 1.8 ms/token |

### Why not MLX?

MLX Swift targets the Apple GPU (Metal) and is excellent when you want maximum throughput on a plugged-in Mac. CoreML-LLM targets the Apple Neural Engine instead — which matters when:

- You want the GPU to stay free for rendering, games, or other ML work
- You want the LLM to coexist with foreground apps without competing for the same silicon
- You want the model placed on the most power-efficient compute unit available on Apple devices

The two are complementary, not competing. If you're on a Mac and want to burn through a 70B model as fast as possible, use MLX. If you want a 2B model quietly running on ANE inside an iPhone app, this library is aimed at that case.

## Adding New Models

See [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) for a step-by-step guide, [docs/CONVERSION.md](docs/CONVERSION.md) for the full conversion reference, and [docs/MULTIMODAL.md](docs/MULTIMODAL.md) for the vision pipeline architecture.

## Project Structure

```
CoreML-LLM/
├── Package.swift
├── Sources/CoreMLLLM/
│   ├── CoreMLLLM.swift                  # Public API (auto-detects model layout)
│   ├── ChunkedEngine.swift              # SWA 4-chunk decode + prefill engine
│   ├── EmbeddingLookup.swift            # INT8 quantized embedding table
│   ├── ImageProcessor.swift             # Vision encoder preprocessing
│   └── ModelConfig.swift
├── conversion/                          # Python conversion pipeline
│   ├── convert.py                       # CLI entry point
│   ├── ane_ops.py                       # ANE-optimized ops (RMSNorm, softmax, RoPE)
│   ├── base_model.py
│   ├── exporter.py                      # CoreML export + INT4 palettization
│   └── models/
│       ├── qwen2.py
│       ├── gemma4.py                    # Gemma 4 E2B base model
│       ├── gemma4_swa_chunks.py         # 4-chunk decode (SWA)
│       ├── gemma4_prefill_chunks.py     # 4-chunk prefill (seq=512)
│       └── gemma4_vision.py
├── Examples/CoreMLLLMChat/              # iOS sample app
│   └── CoreMLLLMChat.xcodeproj
└── docs/
    ├── CONVERSION.md
    ├── ADDING_MODELS.md
    └── MULTIMODAL.md
```

## Requirements

- **Conversion**: Python 3.10-3.12, coremltools 8+, PyTorch 2.2+
- **Inference**: iOS 18+ / macOS 15+ (MLState-free stateless path)
- **Sample app**: Xcode 16+

## License

MIT for the CoreML-LLM code. Gemma 4 weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
