#!/usr/bin/env python3
"""Collect hidden states from Gemma 4 E2B for EAGLE draft model training.

Run on Colab (needs GPU + target model). Reads the corpus JSONL from
download_eagle_corpus.py, runs forward passes, and saves tensors.

Usage (Colab):
    # 1. Upload eagle_corpus.jsonl to Drive or Colab
    # 2. Run this script:
    !python collect_eagle_hidden_states.py \
        --corpus /content/drive/MyDrive/eagle_corpus.jsonl \
        --output /content/drive/MyDrive/eagle_draft/training_data.pt \
        --num-samples 30000 \
        --seq-len 512

    # 3. Training notebook loads training_data.pt (no target model needed)

The output .pt file contains all tensors needed for training:
    h_in, e_in, h_tgt, tok_tgt (train/test split),
    lm_head_weight, embed_scale
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Collect hidden states for EAGLE draft training")
    parser.add_argument("--corpus", type=str, required=True,
                        help="Path to eagle_corpus.jsonl")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pt file path")
    parser.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--num-samples", type=int, default=30000,
                        help="Number of sequences to process")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Max tokens per sequence")
    parser.add_argument("--test-split", type=float, default=0.05,
                        help="Fraction of data for validation")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for forward passes (1 = safest for memory)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load corpus
    print(f"\nLoading corpus from {args.corpus}...")
    texts = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    print(f"  {len(texts)} sequences in corpus")

    # Load target model
    print(f"\nLoading {args.model_id}...")
    from transformers import Gemma4ForConditionalGeneration, AutoTokenizer

    target = Gemma4ForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map=device)
    target.eval()
    for p in target.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    embed_fn = target.get_input_embeddings()

    hidden_size = target.config.text_config.hidden_size
    embed_scale = hidden_size ** 0.5
    lm_head_weight = target.lm_head.weight.data.clone().cpu().half()
    print(f"  hidden_size={hidden_size}, vocab={lm_head_weight.shape[0]}")
    print(f"  embed_scale={embed_scale:.2f}")

    # Collect hidden states
    all_h_in = []
    all_e_in = []
    all_h_tgt = []
    all_tok_tgt = []
    collected = 0
    skipped = 0
    total_pairs = 0

    num = min(args.num_samples, len(texts))
    print(f"\nCollecting hidden states from {num} sequences (seq_len={args.seq_len})...")
    t0 = time.time()

    with torch.no_grad():
        for text in tqdm(texts[:num], desc="Forward passes"):
            ids = tokenizer.encode(text, return_tensors="pt",
                                   truncation=True, max_length=args.seq_len).to(device)
            N = ids.shape[1]
            if N < 32:
                skipped += 1
                continue

            out = target.model(input_ids=ids, output_hidden_states=False)
            hidden = out.last_hidden_state[0].cpu().half()  # (N, hidden)
            embeds = embed_fn(ids)[0].cpu().half() * embed_scale  # (N, hidden)

            # EAGLE pairs: (h[t], embed(tok[t+1])) → h[t+1]
            all_h_in.append(hidden[:-1])
            all_e_in.append(embeds[1:])
            all_h_tgt.append(hidden[1:])

            logits = F.linear(hidden[1:].float(), lm_head_weight.float())
            all_tok_tgt.append(logits.argmax(dim=-1))

            total_pairs += N - 1
            collected += 1

            if collected % 2000 == 0:
                elapsed = time.time() - t0
                rate = collected / elapsed
                eta = (num - collected) / rate if rate > 0 else 0
                print(f"  {collected}/{num} sequences, "
                      f"{total_pairs:,} pairs, "
                      f"{rate:.1f} seq/s, "
                      f"ETA {eta / 60:.0f}min")

    elapsed = time.time() - t0
    print(f"\nDone: {collected} sequences, {total_pairs:,} pairs in {elapsed:.0f}s")
    print(f"  Skipped {skipped} short sequences")

    # Free target model
    del target, embed_fn
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Target model freed.")

    # Concatenate
    h_in = torch.cat(all_h_in, dim=0)
    e_in = torch.cat(all_e_in, dim=0)
    h_tgt = torch.cat(all_h_tgt, dim=0)
    tok_tgt = torch.cat(all_tok_tgt, dim=0)
    del all_h_in, all_e_in, all_h_tgt, all_tok_tgt
    gc.collect()

    M = h_in.shape[0]
    mem_gb = (h_in.nbytes + e_in.nbytes + h_tgt.nbytes + tok_tgt.nbytes) / 1e9
    print(f"\nTotal pairs: {M:,}")
    print(f"Tensor memory: {mem_gb:.2f} GB")

    # Train/test split
    split = int(M * (1 - args.test_split))
    perm = torch.randperm(M)
    train_idx = perm[:split]
    test_idx = perm[split:]

    save_dict = {
        "train_h_in": h_in[train_idx],
        "train_e_in": e_in[train_idx],
        "train_h_tgt": h_tgt[train_idx],
        "train_tok_tgt": tok_tgt[train_idx],
        "test_h_in": h_in[test_idx],
        "test_e_in": e_in[test_idx],
        "test_h_tgt": h_tgt[test_idx],
        "test_tok_tgt": tok_tgt[test_idx],
        "lm_head_weight": lm_head_weight,
        "embed_scale": embed_scale,
        "hidden_size": hidden_size,
        "meta": {
            "model_id": args.model_id,
            "num_sequences": collected,
            "seq_len": args.seq_len,
            "total_pairs": M,
            "train_pairs": split,
            "test_pairs": M - split,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(save_dict, args.output)
    size_gb = os.path.getsize(args.output) / 1e9
    print(f"\nSaved: {args.output} ({size_gb:.2f} GB)")
    print(f"  Train: {split:,} pairs")
    print(f"  Test:  {M - split:,} pairs")
    print(f"\nNext: load this file in the training notebook (no target model needed).")


if __name__ == "__main__":
    main()
