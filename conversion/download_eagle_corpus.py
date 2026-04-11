#!/usr/bin/env python3
"""Download diverse text corpus for EAGLE draft model training.

Collects text from multiple domains to ensure the draft model generalizes
across all types of input the target model will see at inference time.

Sources:
  - Wikipedia (wikitext-103)
  - Web text (C4)
  - Instruction following (Alpaca, Dolly)
  - Code (CodeAlpaca)
  - Conversation (UltraChat subset)

Outputs a single JSONL file where each line is a Gemma 4 chat-formatted
sequence ready for forward passes through the target model.

Usage:
    pip install datasets tqdm
    python download_eagle_corpus.py --output ./eagle_corpus.jsonl --num-samples 50000
    python download_eagle_corpus.py --output ./eagle_corpus.jsonl --num-samples 20000 --fast

Then in Colab, upload the .jsonl and run hidden state collection.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from itertools import islice

from tqdm import tqdm


def format_gemma4_chat(user: str, assistant: str = "") -> str:
    """Format as Gemma 4 chat template (matches buildGemmaPrompt in Swift)."""
    s = "<bos>"
    s += f"<|turn>user\n{user}<turn|>\n"
    if assistant:
        s += f"<|turn>model\n{assistant}<turn|>\n"
    return s


def format_gemma4_multiturn(turns: list[dict]) -> str:
    """Format multi-turn conversation."""
    s = "<bos>"
    for turn in turns:
        role = "user" if turn["role"] in ("user", "human") else "model"
        s += f"<|turn>{role}\n{turn['content']}<turn|>\n"
    return s


# ── Dataset loaders ──────────────────────────────────────────


def load_wikitext(n: int) -> list[str]:
    """Wikipedia articles from WikiText-103."""
    from datasets import load_dataset
    print(f"  [wiki] Loading WikiText-103 ({n} samples)...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = []
    for row in ds:
        text = row["text"].strip()
        if len(text) < 300:
            continue
        # Wrap in chat format: "user asks about the topic"
        first_line = text.split("\n")[0].strip(" =")
        if first_line:
            prompt = f"Tell me about {first_line}."
        else:
            prompt = "Continue writing this article."
        texts.append(format_gemma4_chat(prompt, text[:2000]))
        if len(texts) >= n:
            break
    print(f"  [wiki] Got {len(texts)} samples")
    return texts


def load_c4(n: int) -> list[str]:
    """Web text from C4 (Colossal Clean Crawled Corpus)."""
    from datasets import load_dataset
    print(f"  [c4] Loading C4 web text ({n} samples)...")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for row in islice(ds, n * 3):  # oversample, then filter
        text = row["text"].strip()
        if len(text) < 200:
            continue
        texts.append(format_gemma4_chat("Summarize or continue this text.", text[:2000]))
        if len(texts) >= n:
            break
    print(f"  [c4] Got {len(texts)} samples")
    return texts


def load_alpaca(n: int) -> list[str]:
    """Instruction-following from Stanford Alpaca."""
    from datasets import load_dataset
    print(f"  [alpaca] Loading Alpaca ({n} samples)...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    texts = []
    for row in ds:
        inst = row["instruction"].strip()
        inp = row.get("input", "").strip()
        out = row["output"].strip()
        if not inst or not out or len(out) < 50:
            continue
        prompt = f"{inst}\n{inp}" if inp else inst
        texts.append(format_gemma4_chat(prompt, out[:2000]))
        if len(texts) >= n:
            break
    print(f"  [alpaca] Got {len(texts)} samples")
    return texts


def load_dolly(n: int) -> list[str]:
    """Q&A from Databricks Dolly 15k."""
    from datasets import load_dataset
    print(f"  [dolly] Loading Dolly-15k ({n} samples)...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    texts = []
    for row in ds:
        inst = row["instruction"].strip()
        ctx = row.get("context", "").strip()
        resp = row["response"].strip()
        if not inst or not resp:
            continue
        prompt = f"{inst}\n\nContext: {ctx}" if ctx else inst
        texts.append(format_gemma4_chat(prompt, resp[:2000]))
        if len(texts) >= n:
            break
    print(f"  [dolly] Got {len(texts)} samples")
    return texts


def load_code_alpaca(n: int) -> list[str]:
    """Code instruction-following."""
    from datasets import load_dataset
    print(f"  [code] Loading CodeAlpaca ({n} samples)...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    texts = []
    for row in ds:
        inst = row.get("instruction", "").strip()
        inp = row.get("input", "").strip()
        out = row.get("output", "").strip()
        if not inst or not out:
            continue
        prompt = f"{inst}\n{inp}" if inp else inst
        texts.append(format_gemma4_chat(prompt, out[:3000]))
        if len(texts) >= n:
            break
    print(f"  [code] Got {len(texts)} samples")
    return texts


def load_ultrachat(n: int) -> list[str]:
    """Multi-turn conversations from UltraChat."""
    from datasets import load_dataset
    print(f"  [chat] Loading UltraChat ({n} samples)...")
    ds = load_dataset("stingning/ultrachat", split="train", streaming=True)
    texts = []
    for row in islice(ds, n * 2):
        messages = row.get("data", [])
        if not messages or len(messages) < 2:
            continue
        turns = []
        for i, msg in enumerate(messages[:6]):  # max 6 turns
            role = "user" if i % 2 == 0 else "model"
            turns.append({"role": role, "content": msg.strip()[:1500]})
        if len(turns) >= 2:
            texts.append(format_gemma4_multiturn(turns))
        if len(texts) >= n:
            break
    print(f"  [chat] Got {len(texts)} samples")
    return texts


def load_openorca(n: int) -> list[str]:
    """Diverse instruction following from OpenOrca."""
    from datasets import load_dataset
    print(f"  [orca] Loading OpenOrca ({n} samples)...")
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    texts = []
    for row in islice(ds, n * 2):
        system = row.get("system_prompt", "").strip()
        question = row.get("question", "").strip()
        response = row.get("response", "").strip()
        if not question or not response or len(response) < 50:
            continue
        prompt = f"{system}\n\n{question}" if system else question
        texts.append(format_gemma4_chat(prompt, response[:2000]))
        if len(texts) >= n:
            break
    print(f"  [orca] Got {len(texts)} samples")
    return texts


def load_japanese(n: int) -> list[str]:
    """Japanese text for multilingual coverage."""
    from datasets import load_dataset
    print(f"  [ja] Loading Japanese CC-100 ({n} samples)...")
    try:
        ds = load_dataset("cc100", "ja", split="train", streaming=True)
        texts = []
        for row in islice(ds, n * 5):
            text = row["text"].strip()
            if len(text) < 100:
                continue
            texts.append(format_gemma4_chat("この文章について説明してください。", text[:2000]))
            if len(texts) >= n:
                break
        print(f"  [ja] Got {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"  [ja] Skipped: {e}")
        return []


# ── Main ─────────────────────────────────────────────────────


LOADERS = {
    "wiki": (load_wikitext, 0.15),
    "c4": (load_c4, 0.20),
    "alpaca": (load_alpaca, 0.15),
    "dolly": (load_dolly, 0.10),
    "code": (load_code_alpaca, 0.10),
    "ultrachat": (load_ultrachat, 0.15),
    "openorca": (load_openorca, 0.10),
    "japanese": (load_japanese, 0.05),
}

# Fast mode: skip slow streaming datasets
FAST_LOADERS = {"wiki", "alpaca", "dolly", "code"}


def main():
    parser = argparse.ArgumentParser(
        description="Download diverse corpus for EAGLE draft model training")
    parser.add_argument("--output", type=str, default="./eagle_corpus.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--num-samples", type=int, default=50000,
                        help="Total number of text samples to collect")
    parser.add_argument("--fast", action="store_true",
                        help="Skip slow streaming datasets (C4, UltraChat, OpenOrca)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    total = args.num_samples

    loaders = LOADERS
    if args.fast:
        loaders = {k: v for k, v in LOADERS.items() if k in FAST_LOADERS}
        # Redistribute proportions
        total_weight = sum(w for _, w in loaders.values())
        loaders = {k: (fn, w / total_weight) for k, (fn, w) in loaders.items()}

    print(f"Collecting {total} samples from {len(loaders)} sources...")
    print(f"Output: {args.output}")
    if args.fast:
        print("Fast mode: skipping streaming datasets")
    print()

    all_texts = []
    for name, (loader_fn, weight) in loaders.items():
        n = max(100, int(total * weight))
        try:
            texts = loader_fn(n)
            all_texts.extend(texts)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            continue

    random.shuffle(all_texts)
    all_texts = all_texts[:total]

    # Stats
    lengths = [len(t) for t in all_texts]
    total_chars = sum(lengths)
    print(f"\nCollected {len(all_texts)} samples")
    print(f"  Total chars: {total_chars:,} ({total_chars / 1e6:.1f}M)")
    print(f"  Avg length:  {total_chars / len(all_texts):.0f} chars")
    print(f"  Min/Max:     {min(lengths)} / {max(lengths)} chars")

    # Estimate token pairs (rough: 1 token ≈ 4 chars, pairs ≈ tokens - 1)
    est_tokens = total_chars / 4
    print(f"  Est. tokens: ~{est_tokens / 1e6:.1f}M → ~{est_tokens / 1e6:.1f}M training pairs")

    # Write JSONL
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved: {args.output} ({size_mb:.1f} MB)")
    print(f"\nNext: upload to Colab and run hidden state collection.")


if __name__ == "__main__":
    main()
