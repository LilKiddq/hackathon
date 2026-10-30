"""Fused rotary position embedding (RoPE) for Qwen2, forward and backward.

`apply_rotary_pos_emb` in transformers runs q * cos, rotate_half(q) (two slices, a negation and
a cat), * sin and an add, for q and again for k: about ten elementwise kernels per layer and as
many again in the backward pass, all on fp32 tensors in this task. Here one launch per tensor
reads the [batch, heads, seq, dim] projection view in place (no .contiguous() copy), rotates it
and writes the result; only cos/sin are saved, since the gradient of a rotation needs no input.

Math: Su et al., "RoFormer" (2021), https://arxiv.org/abs/2104.09864, in the split-half layout
of transformers 5.16.1's Qwen2. Products are rounded to the output dtype before the add, as the
unfused PyTorch ops do, and FMA contraction is off. The backward pass applies the transposed
rotation. The code is our own (team).
"""

from functools import wraps

import torch
import triton
import triton.language as tl

BLOCK_T = 4  # token rows per program
# The padded length changes every step. Triton would compile a new variant whenever one of these
# integers changes whether it is divisible by 16, and compile time counts in the measured run.
_VARYING = ["sxb", "sxh", "sxt", "scb", "sct", "ssb", "sst", "seq", "tokens"]


@triton.jit(do_not_specialize=_VARYING)
def rope_rotate_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr,
                       sxb, sxh, sxt, sxd, scb, sct, scd, ssb, sst, ssd,
                       seq, tokens, heads,
                       D: tl.constexpr, BACKWARD: tl.constexpr,
                       BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr):
    """y = x * cos + rotate_half(x) * sin (or its transpose for BACKWARD), one head per
    program column; y is a contiguous [batch, heads, seq, D] tensor."""
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    head = tl.program_id(1)
    batch = rows // seq
    pos = rows % seq
    cols = tl.arange(0, BLOCK_D)
    paired = (cols + D // 2) % D
    valid = (rows[:, None] < tokens) & (cols[None, :] < D)

    x_row = (batch * sxb + head * sxh + pos * sxt)[:, None]
    x = tl.load(x_ptr + x_row + cols[None, :] * sxd, valid, other=0).to(tl.float32)
    x_pair = tl.load(x_ptr + x_row + paired[None, :] * sxd, valid, other=0).to(tl.float32)
    c = tl.load(cos_ptr + (batch * scb + pos * sct)[:, None] + cols[None, :] * scd,
                valid, other=0).to(tl.float32)
    s_row = (batch * ssb + pos * sst)[:, None]
    if BACKWARD:
        # transpose of the rotation: dx[j] = dy[j] cos[j] + sign * dy[p(j)] sin[p(j)]
        s = tl.load(sin_ptr + s_row + paired[None, :] * ssd, valid, other=0).to(tl.float32)
        sign = tl.where(cols < D // 2, 1.0, -1.0)
    else:
        # rotate_half(x)[j] = -x[j + D/2] for the first half, x[j - D/2] for the second
        s = tl.load(sin_ptr + s_row + cols[None, :] * ssd, valid, other=0).to(tl.float32)
        sign = tl.where(cols < D // 2, -1.0, 1.0)
    first = (x * c).to(y_ptr.dtype.element_ty).to(tl.float32)
    second = (x_pair * s).to(y_ptr.dtype.element_ty).to(tl.float32)
    y_row = ((batch * heads + head) * seq + pos)[:, None] * D
    tl.store(y_ptr + y_row + cols[None, :], first + sign[None, :] * second, valid)


def _rotate(x, cos, sin, out, backward):
    batch, heads, seq, dim = x.shape
    # a size-one batch of cos/sin is broadcast over the batch
    cb = 0 if cos.shape[0] == 1 else cos.stride(0)
    sb = 0 if sin.shape[0] == 1 else sin.stride(0)
    rope_rotate_kernel[(triton.cdiv(batch * seq, BLOCK_T), heads)](
        x, cos, sin, out, *x.stride(), cb, cos.stride(1), cos.stride(2),
        sb, sin.stride(1), sin.stride(2), seq, batch * seq, heads,
        D=dim, BACKWARD=backward, BLOCK_D=triton.next_power_of_2(dim), BLOCK_T=BLOCK_T,
        num_warps=4, enable_fp_fusion=False)
    return out


class RoPE(torch.autograd.Function):
    """RoPE for [batch, heads, seq, dim] q/k and frozen [batch or 1, seq, dim] cos/sin."""

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        q_out = torch.empty(q.shape, device=q.device, dtype=torch.promote_types(q.dtype, cos.dtype))
        k_out = torch.empty(k.shape, device=k.device, dtype=torch.promote_types(k.dtype, cos.dtype))
        _rotate(q, cos, sin, q_out, False)
        _rotate(k, cos, sin, k_out, False)
        ctx.save_for_backward(cos, sin)
        ctx.dtypes = (q.dtype, k.dtype)
        ctx.set_materialize_grads(False)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        grads = []
        for grad, dtype in zip((dq, dk), ctx.dtypes):
            if grad is None:
                grads.append(None)
            else:
                out = torch.empty(grad.shape, device=grad.device, dtype=dtype)
                grads.append(_rotate(grad, cos, sin, out, True))
        return grads[0], grads[1], None, None


def patch_qwen2_rope():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb
    if getattr(original, "_kc_patched", False):
        return

    @wraps(original)
    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        tensors = (q, k, cos, sin)
        if (unsqueeze_dim != 1 or not all(t.is_cuda for t in tensors)
                or q.ndim != 4 or k.ndim != 4 or cos.ndim != 3 or sin.ndim != 3
                or q.dtype not in (torch.float16, torch.float32) or k.dtype != q.dtype
                or cos.dtype not in (torch.float16, torch.float32) or sin.dtype != cos.dtype
                or cos.requires_grad or sin.requires_grad
                or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]
                or cos.shape != sin.shape or cos.shape[0] not in (1, q.shape[0])
                or cos.shape[1:] != q.shape[2:]
                or any(size == 0 for t in tensors for size in t.shape)
                or q.shape[-1] % 2 or q.shape[-1] > 256):
            return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        return RoPE.apply(q, k, cos, sin)

    apply_rotary_pos_emb._kc_patched = True
    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb
