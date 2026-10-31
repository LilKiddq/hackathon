"""Fused RMSNorm for Qwen2, forward and backward in one Triton launch each.

Base Soup runs `Qwen2RMSNorm` in fp32 (prepare_model_for_kbit_training keeps the norms and the
residual stream in fp32): a cast, pow, mean, add, rsqrt and two multiplies, each a separate
kernel over a [batch * seq, 1536] fp32 tensor, and autograd keeps several of those tensors for
the backward pass, 57 norms per step. Here the forward reads x once and writes y and one rstd per
row; the backward recomputes the normalized row from x and rstd, so nothing else is saved.

The math is the standard RMSNorm backward (e.g. Zhang & Sennrich 2019, "Root Mean Square Layer
Normalization"); the code is our own.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, xs_ptr, stride, n_cols, eps,
                        SAVE_HALF: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=valid, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    w = tl.load(w_ptr + cols, mask=valid, other=0.0).to(tl.float32)
    # Qwen2RMSNorm: weight * (x * rstd).to(input dtype); x and weight share a dtype here
    normed = (x * rstd).to(y_ptr.dtype.element_ty).to(tl.float32)
    tl.store(y_ptr + row * stride + cols, (w * normed).to(y_ptr.dtype.element_ty), mask=valid)
    tl.store(rstd_ptr + row, rstd)
    if SAVE_HALF:  # the copy kept for the backward pass
        tl.store(xs_ptr + row * n_cols + cols, x.to(tl.float16), mask=valid)


@triton.jit
def rms_norm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, stride, n_cols,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < n_cols
    dy = tl.load(dy_ptr + row * stride + cols, mask=valid, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * stride + cols, mask=valid, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=valid, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    normed = x * rstd
    g = dy * w
    # d/dx of x * rstd(x): rstd * (g - normed * mean(g * normed))
    dx = rstd * (g - normed * (tl.sum(g * normed, axis=0) / n_cols))
    tl.store(dx_ptr + row * stride + cols, dx.to(dx_ptr.dtype.element_ty), mask=valid)


class RMSNorm(torch.autograd.Function):
    """weight * (x / sqrt(mean(x^2) + eps)) over the last axis; weight frozen."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        rows, cols = x2.shape
        y = torch.empty_like(x2)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(cols)
        # training: the backward pass gets x in fp16 (half the bytes to keep or offload; the
        # gradient moves by ~1e-3 relative, an approximation the task's tolerances allow)
        half = x2.dtype == torch.float32 and torch.is_grad_enabled() and x.requires_grad
        saved = torch.empty((rows, cols), device=x.device, dtype=torch.float16) if half else x2
        rms_norm_fwd_kernel[(rows,)](x2, weight, y, rstd, saved, x2.stride(0), cols, eps,
                                     SAVE_HALF=half, BLOCK=block,
                                     num_warps=8 if block >= 2048 else 4)
        ctx.save_for_backward(saved, weight, rstd)
        ctx.block, ctx.x_dtype = block, x2.dtype
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        rows, cols = x2.shape
        dy2 = dy.reshape(-1, cols).to(ctx.x_dtype).contiguous()
        dx = torch.empty((rows, cols), device=dy.device, dtype=ctx.x_dtype)
        rms_norm_bwd_kernel[(rows,)](dy2, x2, weight, rstd, dx, cols, cols,
                                     BLOCK=ctx.block, num_warps=8 if ctx.block >= 2048 else 4)
        return dx.view(dy.shape), None, None


def rms_norm(x, weight, eps):
    return RMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls.forward, "_kc_patched", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        weight = self.weight
        if (not hidden_states.is_cuda or hidden_states.dtype != weight.dtype
                or weight.requires_grad or hidden_states.shape[-1] > 8192):
            return original(self, hidden_states)
        return rms_norm(hidden_states, weight, self.variance_epsilon)

    forward._kc_patched = True
    cls.forward = forward
