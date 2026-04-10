#!/usr/bin/env python3
"""Convert Gemma 4 E2B audio tower (Conformer encoder) to CoreML.

Converts the full audio pipeline:
  mel spectrogram → SubSampleConv → 12 Conformer layers → output_proj → embed_audio

Input:  input_features (1, T, 128) float16 — mel spectrogram
Output: audio_features (1, S, 1536) float16 — projected audio tokens

S ≈ T/4 tokens (each ~40ms of audio).
  T=200 → S=50  (~2 sec)
  T=500 → S=125 (~5 sec)
  T=3000 → S=750 (~30 sec)

Key conversion tricks:
  1. Precompute 5D blocked attention mask (bypass create_bidirectional_mask)
  2. Replace all shape-dependent ops with fixed-shape alternatives
  3. Precompute relative position embeddings as buffer
  4. Trace in float32, convert to fp16 via compute_precision

Usage:
    python convert_audio.py --output ./output/audio
    python convert_audio.py --mel-frames 3000 --output ./output/audio  # 30 sec
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


class FixedShapeConformerAttention(nn.Module):
    """Conformer chunked attention with all shapes hardcoded for CoreML tracing.

    The HF Gemma4AudioAttention uses .shape, //, and dynamic reshape — these
    produce int() cast ops that coremltools cannot convert. This class
    precomputes all dimensions and uses only fixed-shape tensor ops.
    """

    def __init__(self, original_attn, seq_len: int):
        super().__init__()
        # Copy weight modules
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.post = original_attn.post
        self.relative_k_proj = original_attn.relative_k_proj
        self.per_dim_scale = original_attn.per_dim_scale
        self.register_buffer("softcap",
                             original_attn.softcap.clone())

        # Fixed dimensions
        self.seq_len = seq_len
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
        self.hidden_size = self.num_heads * self.head_dim
        self.chunk_size = original_attn.chunk_size
        self.context_size = original_attn.context_size
        self.max_past = original_attn.max_past_horizon
        self.max_future = original_attn.max_future_horizon
        self.q_scale = original_attn.q_scale
        self.k_scale = original_attn.k_scale
        self.invalid_logits = original_attn.config.attention_invalid_logits_value

        # Precomputed shape constants
        self.num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        self.pad_amount = self.num_blocks * self.chunk_size - seq_len
        self.padded_seq = self.num_blocks * self.chunk_size

        # Relative position shift constants
        self.pos_length = 13  # arange(12, -1, -1) = 13 values
        self.rel_shift_pad = self.context_size + 1 - self.pos_length
        self.block_ctx_total = self.chunk_size * (self.context_size + 1)
        self.block_ctx_valid = self.chunk_size * self.context_size

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        # hidden_states: (1, seq_len, hidden_size)
        S = self.seq_len
        H = self.num_heads
        D = self.head_dim
        C = self.chunk_size
        N = self.num_blocks
        CTX = self.context_size

        # Q/K/V projections → fixed reshape (no .shape access)
        q = self.q_proj(hidden_states).float().reshape(1, S, H, D)
        k = self.k_proj(hidden_states).float().reshape(1, S, H, D)
        v = self.v_proj(hidden_states).float().reshape(1, S, H, D)

        # Scaling
        q = q * self.q_scale * F.softplus(self.per_dim_scale)
        k = k * self.k_scale

        # Q → blocks: (1, N, C, H, D)
        q_padded = F.pad(q, (0, 0, 0, 0, 0, self.pad_amount))
        q_blocks = q_padded.reshape(1, N, C, H, D)

        # K/V → overlapping context windows: (1, N, CTX, H, D)
        k_padded = F.pad(k, (0, 0, 0, 0,
                              self.max_past, self.max_future + C - 1))
        v_padded = F.pad(v, (0, 0, 0, 0,
                              self.max_past, self.max_future + C - 1))
        k_blocks = torch.stack(
            [k_padded[:, i * C: i * C + CTX] for i in range(N)], dim=1)
        v_blocks = torch.stack(
            [v_padded[:, i * C: i * C + CTX] for i in range(N)], dim=1)

        # Relative position encoding
        rel_k = self.relative_k_proj(position_embeddings)
        rel_k = rel_k.reshape(self.pos_length, H, D).to(q.dtype)

        # Content attention: Q @ K^T per block
        queries = q_blocks.permute(0, 3, 1, 2, 4)       # (1, H, N, C, D)
        keys_t = k_blocks.permute(0, 3, 1, 4, 2)        # (1, H, N, D, CTX)
        matrix_ac = queries @ keys_t                      # (1, H, N, C, CTX)

        # Position attention: Q @ rel_K^T then shift
        q_flat = queries.reshape(1, H, N * C, D)          # (1, H, N*C, D)
        rel_k_t = rel_k.permute(1, 2, 0)                 # (H, D, 13)
        matrix_bd = q_flat @ rel_k_t                      # (1, H, N*C, 13)
        matrix_bd = matrix_bd.reshape(1, H, N, C, self.pos_length)

        # Relative position shift (Trans-XL appendix B)
        matrix_bd = F.pad(matrix_bd, (0, self.rel_shift_pad))
        matrix_bd = matrix_bd.reshape(1, H, N, self.block_ctx_total)
        matrix_bd = matrix_bd[:, :, :, :self.block_ctx_valid]
        matrix_bd = matrix_bd.reshape(1, H, N, C, CTX)

        # Combine + soft-cap
        attn = matrix_ac + matrix_bd
        attn = torch.tanh(attn / self.softcap) * self.softcap

        # Apply mask
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask.logical_not(),
                                    self.invalid_logits)

        # Softmax + value aggregation
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(v.dtype)
        values_t = v_blocks.permute(0, 3, 1, 2, 4)       # (1, H, N, CTX, D)
        out = attn @ values_t                              # (1, H, N, C, D)
        out = out.permute(0, 2, 3, 1, 4).reshape(1, self.padded_seq,
                                                   self.hidden_size)
        out = out[:, :S].contiguous()
        out = self.post(out.to(dtype=self.post.linear.weight.dtype))

        return out, None


class TraceableAudioTower(nn.Module):
    """Trace-friendly wrapper for the Gemma 4 Conformer audio encoder.

    Replaces dynamic operations with fixed-shape equivalents:
      1. create_bidirectional_mask → precomputed 5D buffer
      2. AudioAttention → FixedShapeConformerAttention
      3. Layer forward → manual call (no decorator kwargs)
    """

    def __init__(self, hf_model, mel_frames: int):
        super().__init__()
        audio = hf_model.model.audio_tower
        cfg = audio.config

        # Encoder input
        self.subsample_conv_projection = audio.subsample_conv_projection

        # Encoder output
        self.output_proj = audio.output_proj
        self.embed_norm = hf_model.model.embed_audio.embedding_pre_projection_norm
        self.embed_proj = hf_model.model.embed_audio.embedding_projection

        self.num_layers = cfg.num_hidden_layers

        # Compute seq_len from a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, mel_frames, 128, dtype=torch.float16)
            h, output_mask = self.subsample_conv_projection(dummy, None)
            seq_len = int(h.shape[1])

        self._seq_len = seq_len
        print(f"  Audio tower: T={mel_frames} mel → S={seq_len} tokens")

        # Precompute position embeddings (constant)
        with torch.no_grad():
            pos_emb = audio.rel_pos_enc(torch.zeros(1, 1, cfg.hidden_size))
            self.register_buffer("position_embeddings", pos_emb)

        # Precompute attention mask for fixed input length
        with torch.no_grad():
            attn_mask = create_bidirectional_mask(
                config=cfg,
                inputs_embeds=h,
                attention_mask=output_mask,
                and_mask_function=sliding_window_mask_function(
                    (cfg.attention_context_left - 1, cfg.attention_context_right)
                ),
            )
            attn_mask = audio._convert_4d_mask_to_blocked_5d(attn_mask)
            self.register_buffer("attention_mask", attn_mask)
        print(f"  Mask shape: {tuple(attn_mask.shape)}")

        # Build fixed-shape attention + copy layer submodules
        self.self_attns = nn.ModuleList()
        self.feed_forward1s = nn.ModuleList()
        self.feed_forward2s = nn.ModuleList()
        self.lconv1ds = nn.ModuleList()
        self.norm_pre_attns = nn.ModuleList()
        self.norm_post_attns = nn.ModuleList()
        self.norm_outs = nn.ModuleList()
        self.gradient_clips = []

        for layer in audio.layers:
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

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: (1, T, 128) mel spectrogram
        Returns:
            audio_features: (1, S, 1536) projected audio tokens
        """
        hidden_states, _ = self.subsample_conv_projection(input_features, None)

        for i in range(self.num_layers):
            gc = self.gradient_clips[i]

            # FFN 1
            hidden_states = self.feed_forward1s[i](hidden_states)

            # Self-attention
            residual = hidden_states
            hidden_states = torch.clamp(hidden_states, -gc, gc)
            hidden_states = self.norm_pre_attns[i](hidden_states)
            hidden_states, _ = self.self_attns[i](
                hidden_states,
                self.position_embeddings,
                self.attention_mask,
            )
            hidden_states = torch.clamp(hidden_states, -gc, gc)
            hidden_states = self.norm_post_attns[i](hidden_states)
            hidden_states = hidden_states + residual

            # Light Conv1D
            hidden_states = self.lconv1ds[i](hidden_states)

            # FFN 2
            hidden_states = self.feed_forward2s[i](hidden_states)

            # Output norm
            hidden_states = torch.clamp(hidden_states, -gc, gc)
            hidden_states = self.norm_outs[i](hidden_states)

        # Output projection + embed_audio
        hidden_states = self.output_proj(hidden_states)
        hidden_states = self.embed_norm(hidden_states)
        hidden_states = self.embed_proj(hidden_states)
        return hidden_states


def verify_output(hf_model, wrapper, mel_frames: int) -> float:
    """Compare HF model output vs wrapper output."""
    with torch.no_grad():
        torch.manual_seed(42)
        dummy = torch.randn(1, mel_frames, 128, dtype=torch.float16)

        # HF reference (fp16)
        audio_out = hf_model.model.audio_tower(
            dummy, attention_mask=None, return_dict=True)
        hf_features = hf_model.model.embed_audio(
            inputs_embeds=audio_out.last_hidden_state)

        # Our wrapper (float32)
        wrapper_features = wrapper(dummy.float())

        diff = (hf_features.float() - wrapper_features.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        cos_sim = F.cosine_similarity(
            hf_features.float().flatten().unsqueeze(0),
            wrapper_features.float().flatten().unsqueeze(0),
        ).item()

        print(f"  Verification: max_err={max_err:.6f}, mean_err={mean_err:.6f}, "
              f"cosine={cos_sim:.6f}")
        return cos_sim


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemma 4 E2B audio tower to CoreML")
    parser.add_argument("--model-path", type=str,
                        default="./output/gemma4-e2b-final/hf_model",
                        help="Path to local HF model directory")
    parser.add_argument("--output", type=str, default="./output/audio")
    parser.add_argument("--mel-frames", type=int, default=200,
                        help="Fixed mel spectrogram frames (200≈2s, 3000≈30s)")
    parser.add_argument("--quantize", type=str, default="int4",
                        choices=["int4", "int8", "none"])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    quantize = None if args.quantize == "none" else args.quantize

    # Load HF model
    print("Loading HF model...")
    hf_model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16)
    hf_model.eval()

    # Build traceable wrapper (float32 for LayerNorm epsilon compatibility)
    print(f"\nBuilding traceable audio tower (T={args.mel_frames})...")
    wrapper = TraceableAudioTower(hf_model, args.mel_frames)
    wrapper.eval().float()

    # Verify wrapper matches HF
    print("\nVerifying wrapper vs HF model...")
    cos_sim = verify_output(hf_model, wrapper, args.mel_frames)
    if cos_sim < 0.99:
        print(f"  WARNING: cosine similarity {cos_sim:.4f} < 0.99")
    else:
        print(f"  PASS: cosine similarity {cos_sim:.4f}")

    # Trace
    print("\nTracing...")
    sample = torch.zeros(1, args.mel_frames, 128, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (sample,))
    print("  Trace OK")

    # Convert to CoreML
    print("\nConverting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_features",
                shape=(1, args.mel_frames, 128),
                dtype=np.float16,
            ),
        ],
        outputs=[
            ct.TensorType(name="audio_features", dtype=np.float16),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )

    # Quantize
    if quantize:
        print(f"  Applying {quantize} quantization...")
        if quantize == "int4":
            op_config = ct.optimize.coreml.OpPalettizerConfig(
                nbits=4, granularity="per_grouped_channel", group_size=32)
            config = ct.optimize.coreml.OptimizationConfig(
                global_config=op_config)
            mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config)
        elif quantize == "int8":
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int8")
            config = ct.optimize.coreml.OptimizationConfig(
                global_config=op_config)
            mlmodel = ct.optimize.coreml.linear_quantize_weights(
                mlmodel, config)

    # Save
    path = os.path.join(args.output, "audio.mlpackage")
    if os.path.exists(path):
        shutil.rmtree(path)
    mlmodel.save(path)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(path) for f in fns
    ) / 1024 / 1024
    print(f"  Saved {path} ({size_mb:.1f} MB)")

    # Save audio config for Swift
    seq_len = wrapper._seq_len
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
    }
    config_path = os.path.join(args.output, "audio_config.json")
    with open(config_path, "w") as f:
        json.dump(audio_config, f, indent=2)
    print(f"  Saved {config_path}")

    print(f"\nAudio conversion complete!")
    print(f"  Model:  {path} ({size_mb:.1f} MB)")
    print(f"  Config: {config_path}")
    print(f"  Input:  (1, {args.mel_frames}, 128) mel → "
          f"Output: (1, {seq_len}, 1536) features")


if __name__ == "__main__":
    main()
