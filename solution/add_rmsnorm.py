"""Residual add + frozen-weight RMSNorm for Qwen2; original Triton implementation.

Only s, weight and one inverse RMS per row survive forward. Row counts live in
the launch grid; widths and epsilon are runtime scalars, including across steps.
"""

from functools import wraps

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n_cols", "eps"])
def add_rmsnorm_fwd_kernel(r_ptr, d_ptr, w_ptr, s_ptr, y_ptr, inv_ptr,
                           n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    valid = col < n_cols
    offset = row * n_cols + col
    r = tl.load(r_ptr + offset, valid, other=0).to(tl.float32)
    d = tl.load(d_ptr + offset, valid, other=0).to(tl.float32)
    s = r + d
    inv = tl.rsqrt(tl.sum(s * s, 0) / n_cols + eps)
    w = tl.load(w_ptr + col, valid, other=0).to(tl.float32)
    tl.store(s_ptr + offset, s, valid)
    tl.store(y_ptr + offset, w * (s * inv), valid)
    tl.store(inv_ptr + row, inv)


@triton.jit(do_not_specialize=["n_cols", "has_ds", "has_dy"])
def add_rmsnorm_bwd_kernel(ds_ptr, dy_ptr, s_ptr, w_ptr, inv_ptr,
                           dr_ptr, dd_ptr, n_cols, has_ds, has_dy,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    valid = col < n_cols
    offset = row * n_cols + col
    ds = tl.load(ds_ptr + offset, valid & has_ds, other=0).to(tl.float32)
    dy = tl.load(dy_ptr + offset, valid & has_dy, other=0).to(tl.float32)
    s = tl.load(s_ptr + offset, valid, other=0)
    w = tl.load(w_ptr + col, valid, other=0)
    inv = tl.load(inv_ptr + row)
    n = s * inv
    g = dy * w
    total = ds + inv * (g - n * (tl.sum(g * n, 0) / n_cols))
    tl.store(dr_ptr + offset, total, valid)
    tl.store(dd_ptr + offset, total, valid)


class AddRMSNorm(torch.autograd.Function):
    """CUDA fp32 residual and frozen weight, fp16/fp32 delta -> fp32 (s, y)."""

    @staticmethod
    def forward(ctx, residual, delta, weight, eps):
        residual, delta = residual.contiguous(), delta.contiguous()
        s = torch.empty_like(residual)
        y = torch.empty_like(residual)
        cols = residual.shape[-1]
        rows = residual.numel() // cols
        inv = torch.empty(rows, device=residual.device, dtype=torch.float32)
        block = triton.next_power_of_2(cols)
        add_rmsnorm_fwd_kernel[(rows,)](
            residual, delta, weight, s, y, inv, cols, eps, BLOCK=block,
            num_warps=8 if block >= 2048 else 4, enable_fp_fusion=False)
        ctx.save_for_backward(s, weight, inv)
        ctx.delta_dtype = delta.dtype
        ctx.block = block
        ctx.set_materialize_grads(False)
        return s, y

    @staticmethod
    def backward(ctx, ds, dy):
        if ds is None and dy is None:
            return None, None, None, None
        s, weight, inv = ctx.saved_tensors
        cols = s.shape[-1]
        dr = torch.empty_like(s)
        dd = torch.empty_like(s, dtype=ctx.delta_dtype)
        add_rmsnorm_bwd_kernel[(s.numel() // cols,)](
            ds.contiguous() if ds is not None else s,
            dy.contiguous() if dy is not None else s,
            s, weight, inv, dr, dd, cols, ds is not None, dy is not None,
            BLOCK=ctx.block, num_warps=8 if ctx.block >= 2048 else 4,
            enable_fp_fusion=False)
        return dr, dd, None, None


def _can_fuse(residual, delta, weight):
    return (residual.is_cuda and delta.device == residual.device
            and weight.device == residual.device and residual.dtype == torch.float32
            and delta.dtype in (torch.float16, torch.float32)
            and weight.dtype == torch.float32 and not weight.requires_grad
            and residual.shape == delta.shape and residual.ndim > 0
            and residual.numel() > 0 and 0 < residual.shape[-1] <= 8192
            and weight.shape == (residual.shape[-1],) and weight.is_contiguous())


def patch_qwen2_add_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2DecoderLayer
    original = cls.forward
    if getattr(original, "_add_rmsnorm_patched", False):
        return

    @wraps(original)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: modeling_qwen2.Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: modeling_qwen2.Unpack[modeling_qwen2.TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        norm = self.post_attention_layernorm
        if (isinstance(norm, modeling_qwen2.Qwen2RMSNorm)
                and _can_fuse(residual, hidden_states, norm.weight)):
            residual, hidden_states = AddRMSNorm.apply(
                residual, hidden_states, norm.weight, norm.variance_epsilon)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    forward._add_rmsnorm_patched = True
    cls.forward = forward
