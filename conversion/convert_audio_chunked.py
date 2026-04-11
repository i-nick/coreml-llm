#!/usr/bin/env python3
"""Convert Gemma 4 E2B audio tower as chunked CoreML models for ANE.

Splits the 12-layer Conformer into 3 chunks (4 layers each):
  chunk 1: SubSampleConv → Layers 0-3 → hidden_states
  chunk 2: Layers 4-7 → hidden_states
  chunk 3: Layers 8-11 → output_proj → embed_audio → audio_features

Each chunk runs independently on ANE. The GPU fp16 precision issue
(feature range compression) is avoided by using ANE execution.

Usage:
    python convert_audio_chunked.py --mel-frames 1000 --output ./output/audio-chunked
"""
import argparse
import json
import math
import os
import shutil

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Gemma4ForConditionalGeneration
from transformers.masking_utils import create_bidirectional_mask
from transformers.models.gemma4.modeling_gemma4 import sliding_window_mask_function

from convert_audio import FixedShapeConformerAttention


class AudioChunk(nn.Module):
    """A chunk of Conformer layers with precomputed mask + position embeddings."""

    def __init__(self, layers, seq_len, cfg, has_subsample=False,
                 subsample=None, has_output=False, output_proj=None,
                 embed_norm=None, embed_proj=None, attention_mask=None,
                 position_embeddings=None):
        super().__init__()
        self.has_subsample = has_subsample
        self.has_output = has_output
        self.num_layers = len(layers)

        if has_subsample:
            self.subsample_conv_projection = subsample

        # Build fixed-shape attention + copy layer submodules
        self.self_attns = nn.ModuleList()
        self.feed_forward1s = nn.ModuleList()
        self.feed_forward2s = nn.ModuleList()
        self.lconv1ds = nn.ModuleList()
        self.norm_pre_attns = nn.ModuleList()
        self.norm_post_attns = nn.ModuleList()
        self.norm_outs = nn.ModuleList()
        self.gradient_clips = []

        for layer in layers:
            self.self_attns.append(
                FixedShapeConformerAttention(layer.self_attn, seq_len))
            self.feed_forward1s.append(layer.feed_forward1)
            self.feed_forward2s.append(layer.feed_forward2)
            self.lconv1ds.append(layer.lconv1d)
            self.norm_pre_attns.append(layer.norm_pre_attn)
            self.norm_post_attns.append(layer.norm_post_attn)
            self.norm_outs.append(layer.norm_out)
            gc = min(layer.gradient_clipping,
                     torch.finfo(layer.norm_pre_attn.weight.dtype).max)
            self.gradient_clips.append(gc)

        if has_output:
            self.output_proj = output_proj
            self.embed_norm = embed_norm
            self.embed_proj = embed_proj

        self.register_buffer("attention_mask", attention_mask)
        self.register_buffer("position_embeddings", position_embeddings)

    def forward(self, x):
        if self.has_subsample:
            x, _ = self.subsample_conv_projection(x, None)

        for i in range(self.num_layers):
            gc = self.gradient_clips[i]
            x = self.feed_forward1s[i](x)
            residual = x
            x = torch.clamp(x, -gc, gc)
            x = self.norm_pre_attns[i](x)
            x, _ = self.self_attns[i](x, self.position_embeddings,
                                       self.attention_mask)
            x = torch.clamp(x, -gc, gc)
            x = self.norm_post_attns[i](x)
            x = x + residual
            x = self.lconv1ds[i](x)
            x = self.feed_forward2s[i](x)
            x = torch.clamp(x, -gc, gc)
            x = self.norm_outs[i](x)

        if self.has_output:
            x = self.output_proj(x)
            x = self.embed_norm(x)
            x = self.embed_proj(x)

        return x


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemma 4 audio tower as chunked CoreML for ANE")
    parser.add_argument("--model-path", type=str,
                        default="./output/gemma4-e2b-final/hf_model")
    parser.add_argument("--output", type=str, default="./output/audio-chunked")
    parser.add_argument("--mel-frames", type=int, default=1000)
    parser.add_argument("--quantize", type=str, default="int8",
                        choices=["int4", "int8", "none"])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    quantize = None if args.quantize == "none" else args.quantize

    print("Loading HF model...")
    hf_model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16)
    hf_model.eval()

    audio = hf_model.model.audio_tower
    cfg = audio.config

    # Compute seq_len and precomputed buffers
    with torch.no_grad():
        dummy = torch.zeros(1, args.mel_frames, 128, dtype=torch.float16)
        h, output_mask = audio.subsample_conv_projection(dummy, None)
        seq_len = int(h.shape[1])
        pos_emb = audio.rel_pos_enc(torch.zeros(1, 1, cfg.hidden_size))
        attn_mask = create_bidirectional_mask(
            config=cfg, inputs_embeds=h, attention_mask=output_mask,
            and_mask_function=sliding_window_mask_function(
                (cfg.attention_context_left - 1, cfg.attention_context_right)))
        attn_mask = audio._convert_4d_mask_to_blocked_5d(attn_mask)

    print(f"Audio: T={args.mel_frames} mel → S={seq_len} tokens, mask={tuple(attn_mask.shape)}")

    # Split 12 layers into 3 chunks of 4
    layer_splits = [(0, 4), (4, 8), (8, 12)]
    chunk_names = ["audio_chunk1", "audio_chunk2", "audio_chunk3"]

    for idx, ((start, end), name) in enumerate(zip(layer_splits, chunk_names)):
        print(f"\n=== {name}: layers {start}-{end-1} ===")
        layers = list(audio.layers[start:end])

        chunk = AudioChunk(
            layers=layers, seq_len=seq_len, cfg=cfg,
            has_subsample=(idx == 0),
            subsample=audio.subsample_conv_projection if idx == 0 else None,
            has_output=(idx == 2),
            output_proj=audio.output_proj if idx == 2 else None,
            embed_norm=hf_model.model.embed_audio.embedding_pre_projection_norm if idx == 2 else None,
            embed_proj=hf_model.model.embed_audio.embedding_projection if idx == 2 else None,
            attention_mask=attn_mask,
            position_embeddings=pos_emb,
        )
        chunk.eval().float()

        # Trace
        if idx == 0:
            sample = torch.zeros(1, args.mel_frames, 128, dtype=torch.float32)
        else:
            sample = torch.zeros(1, seq_len, cfg.hidden_size, dtype=torch.float32)

        with torch.no_grad():
            traced = torch.jit.trace(chunk, (sample,))
        print(f"  Traced OK")

        # Convert
        if idx == 0:
            input_spec = ct.TensorType(name="input_features",
                                        shape=(1, args.mel_frames, 128),
                                        dtype=np.float16)
        else:
            input_spec = ct.TensorType(name="hidden_states",
                                        shape=(1, seq_len, cfg.hidden_size),
                                        dtype=np.float16)

        if idx == 2:
            output_spec = ct.TensorType(name="audio_features", dtype=np.float16)
        else:
            output_spec = ct.TensorType(name="hidden_states_out", dtype=np.float16)

        mlmodel = ct.convert(
            traced,
            inputs=[input_spec],
            outputs=[output_spec],
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,
        )

        # Quantize
        if quantize:
            if quantize == "int4":
                op_config = ct.optimize.coreml.OpPalettizerConfig(
                    nbits=4, granularity="per_grouped_channel", group_size=32)
                config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
                mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config)
            elif quantize == "int8":
                op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                    mode="linear_symmetric", dtype="int8")
                config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
                mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config)

        path = os.path.join(args.output, f"{name}.mlpackage")
        if os.path.exists(path):
            shutil.rmtree(path)
        mlmodel.save(path)
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(path) for f in fns
        ) / 1024 / 1024
        print(f"  Saved {path} ({size_mb:.1f} MB)")

    # Save audio config
    audio_config = {
        "sampling_rate": 16000,
        "feature_size": 128,
        "frame_length": 320,
        "hop_length": 160,
        "fft_length": 512,
        "mel_floor": 1e-5,
        "min_frequency": 0,
        "max_frequency": 8000,
        "log_offset": 0.001,
        "preemphasis": 0.97,
        "mel_frames": args.mel_frames,
        "num_tokens": seq_len,
        "audio_token_id": 258881,
        "boa_token_id": 256000,
        "eoa_token_id": 258883,
        "ms_per_token": 40,
        "quantization": quantize or "fp16",
        "chunked": True,
        "num_chunks": 3,
    }
    config_path = os.path.join(args.output, "audio_config.json")
    with open(config_path, "w") as f:
        json.dump(audio_config, f, indent=2)

    # Copy mel filterbank
    mel_src = os.path.join(os.path.dirname(args.output), "audio", "mel_filterbank.bin")
    mel_dst = os.path.join(args.output, "mel_filterbank.bin")
    if os.path.exists(mel_src) and not os.path.exists(mel_dst):
        shutil.copy2(mel_src, mel_dst)

    print(f"\nDone! 3 chunks in {args.output}/")
    print(f"  Input:  (1, {args.mel_frames}, 128) mel")
    print(f"  Output: (1, {seq_len}, 1536) audio features")


if __name__ == "__main__":
    main()
