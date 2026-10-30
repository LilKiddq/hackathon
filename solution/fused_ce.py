"""Fused output head + cross-entropy for Qwen2 training, without the full [batch, seq, vocab] logits.

What base Soup does every step (fp16 autocast on a T4): `lm_head` builds fp16 logits of shape
[4, 512, 151936] (~0.6 GB), `ForCausalLMLoss` upcasts them to fp32 (~1.2 GB) and keeps more
buffers of that size for the backward pass, and TRL / Soup read the same logits three more times
for entropy, token accuracy and the rewind per-row losses.

Here the head runs over at most ROWS_PER_CHUNK token rows at a time, only for rows that carry
attention (padding is skipped), and one Triton kernel reads each chunk of logits once and returns
everything the trainer needs: log-sum-exp, the token loss, the entropy and whether the argmax hits
the label. The logits chunk is then dropped. The backward pass recomputes the logits only for the
labelled rows (the assistant tokens), and a second kernel turns them in place into the gradient
`grad_scale * (softmax - onehot)`, rounded to fp16 exactly where base Soup rounds it.

Ideas: the chunked head + loss of Soup's own `stream_loss_chunk` (#1693), Liger-Kernel's fused
linear cross-entropy and the online softmax of Milakov & Gimelshein (2018), "Online normalizer
calculation for softmax". The code is our own.
"""

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100
ROWS_PER_CHUNK = 512  # 512 x 151936 fp16 logits = 156 MB alive at a time
BLOCK_V = 4096
NEG = -1.0e30  # finite "minus infinity": keeps exp(a - b) free of inf - inf


@triton.jit
def lm_head_ce_stats_kernel(logits_ptr, stride, target_ptr, lse_ptr, loss_ptr, entropy_ptr,
                            hit_ptr, V, BLOCK_V: tl.constexpr, NEG: tl.constexpr):
    """One program per token row: log-sum-exp, entropy, argmax and the label's loss in one read.

    Every lane keeps its own running max m, sum of exp(x - m) and sum of exp(x - m) * x, plus its
    own best value and index; the lanes are merged once at the end."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * stride
    target = tl.load(target_ptr + row)
    lanes = tl.arange(0, BLOCK_V)

    m = tl.full([BLOCK_V], NEG, tl.float32)
    s = tl.zeros([BLOCK_V], tl.float32)
    t = tl.zeros([BLOCK_V], tl.float32)
    best = tl.full([BLOCK_V], NEG, tl.float32)
    best_idx = tl.zeros([BLOCK_V], tl.int32)
    for start in range(0, V, BLOCK_V):
        cols = start + lanes
        valid = cols < V
        x = tl.load(row_ptr + cols, mask=valid, other=NEG).to(tl.float32)
        new_m = tl.maximum(m, x)
        alpha = tl.exp(m - new_m)
        e = tl.where(valid, tl.exp(x - new_m), 0.0)
        s = s * alpha + e
        t = t * alpha + tl.where(valid, e * x, 0.0)
        m = new_m
        # strict >: within a lane the first (lowest) index of a tie wins, as in torch.argmax
        better = valid & (x > best)
        best = tl.where(better, x, best)
        best_idx = tl.where(better, cols, best_idx)

    row_max = tl.max(m, axis=0)
    rescale = tl.exp(m - row_max)
    total = tl.sum(s * rescale, axis=0)
    weighted = tl.sum(t * rescale, axis=0)
    lse = row_max + tl.log(total)
    entropy = lse - weighted / total

    top = tl.max(best, axis=0)
    argmax = tl.min(tl.where(best == top, best_idx, V), axis=0)

    labelled = target != -100
    safe_target = tl.where(labelled, target, 0)
    x_target = tl.load(row_ptr + safe_target).to(tl.float32)
    loss = tl.where(labelled, lse - x_target, 0.0)
    hit = labelled & (argmax == safe_target)

    tl.store(lse_ptr + row, lse)
    tl.store(loss_ptr + row, loss)
    tl.store(entropy_ptr + row, entropy)
    tl.store(hit_ptr + row, hit.to(tl.int32))


@triton.jit
def lm_head_ce_grad_kernel(logits_ptr, stride, target_ptr, lse_ptr, scale_ptr, V,
                           BLOCK_V: tl.constexpr):
    """In place: logits row -> scale * (softmax(row) - onehot(target)), in the logits' dtype."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * stride
    target = tl.load(target_ptr + row)
    lse = tl.load(lse_ptr + row)
    scale = tl.load(scale_ptr)
    lanes = tl.arange(0, BLOCK_V)
    for start in range(0, V, BLOCK_V):
        cols = start + lanes
        valid = cols < V
        x = tl.load(row_ptr + cols, mask=valid, other=0.0).to(tl.float32)
        grad = tl.exp(x - lse) - tl.where(cols == target, 1.0, 0.0)
        tl.store(row_ptr + cols, (grad * scale).to(logits_ptr.dtype.element_ty), mask=valid)



class FusedLMHeadCE(torch.autograd.Function):
    """sum over rows of CE(hidden @ weight.T, target) / denominator, plus per-row metrics.

    hidden [N, H] (any float dtype; it is cast to fp16 as autocast casts the head's input),
    weight [V, H] fp16 and frozen, target [N] int64 with -100 for rows without a label,
    denominator a 0-dim float tensor."""

    @staticmethod
    def forward(ctx, hidden, weight, target, denominator):
        # the dtypes are explicit here; autocast also refuses matmul(out=...)
        with torch.autocast("cuda", enabled=False):
            return FusedLMHeadCE._forward(ctx, hidden, weight, target, denominator)

    @staticmethod
    def _forward(ctx, hidden, weight, target, denominator):
        n, vocab = hidden.shape[0], weight.shape[0]
        x16 = hidden.detach().to(torch.float16).contiguous()
        target = target.contiguous()
        lse = torch.empty(n, device=hidden.device, dtype=torch.float32)
        loss = torch.empty_like(lse)
        entropy = torch.empty_like(lse)
        hit = torch.empty(n, device=hidden.device, dtype=torch.int32)
        buffer = torch.empty((min(n, ROWS_PER_CHUNK), vocab), device=hidden.device,
                             dtype=torch.float16)
        for start in range(0, n, ROWS_PER_CHUNK):
            stop = min(n, start + ROWS_PER_CHUNK)
            logits = buffer[: stop - start]
            torch.matmul(x16[start:stop], weight.t(), out=logits)
            lm_head_ce_stats_kernel[(stop - start,)](
                logits, logits.stride(0), target[start:stop], lse[start:stop], loss[start:stop],
                entropy[start:stop], hit[start:stop], vocab, BLOCK_V=BLOCK_V, NEG=NEG, num_warps=8)
        del buffer
        total = loss.sum() / denominator
        labelled = (target != IGNORE_INDEX).nonzero().squeeze(1)
        ctx.save_for_backward(x16, weight, target, lse, labelled, denominator)
        ctx.hidden_dtype = hidden.dtype
        ctx.mark_non_differentiable(loss, entropy, hit)
        return total, loss, entropy, hit

    @staticmethod
    def backward(ctx, grad_total, _grad_loss, _grad_entropy, _grad_hit):
        x16, weight, target, lse, labelled, denominator = ctx.saved_tensors
        n, vocab = x16.shape[0], weight.shape[0]
        grad_hidden = torch.zeros(x16.shape, device=x16.device, dtype=ctx.hidden_dtype)
        if labelled.numel() == 0:
            return grad_hidden, None, None, None
        # d(total)/d(logit) = grad_total / denominator * (softmax - onehot), as ForCausalLMLoss
        scale = (grad_total.float() / denominator.float()).reshape(1).contiguous()
        rows = min(labelled.numel(), ROWS_PER_CHUNK)
        buffer = torch.empty((rows, vocab), device=x16.device, dtype=torch.float16)
        for start in range(0, labelled.numel(), ROWS_PER_CHUNK):
            index = labelled[start:start + ROWS_PER_CHUNK]
            logits = buffer[: index.numel()]
            torch.matmul(x16.index_select(0, index), weight.t(), out=logits)
            lm_head_ce_grad_kernel[(index.numel(),)](
                logits, logits.stride(0), target.index_select(0, index).contiguous(),
                lse.index_select(0, index).contiguous(), scale, vocab, BLOCK_V=BLOCK_V,
                num_warps=8)
            grad_hidden.index_copy_(0, index, (logits @ weight).to(ctx.hidden_dtype))
        return grad_hidden, None, None, None


def fused_lm_head_ce(hidden, weight, target, denominator):
    return FusedLMHeadCE.apply(hidden, weight, target, denominator)
