"""The tied embedding / lm_head held in int8 (one fp32 scale per vocabulary row) during training.

The tied 151936 x 1536 weight is the largest tensor left on the GPU: 467 MB in fp16. Here it is
stored as int8 with a per-row absmax scale (233 MB + 0.6 MB), the classic LLM.int8()-style
row-wise quantization (Dettmers et al. 2022). This is an approximation, and the only one in the
solution: each weight moves by at most half a quantization step (absmax / 254), which shifts the
logits by a few hundredths and the loss by far less than the task's tolerances (checks 5-7).
The embedding lookup dequantizes the looked-up rows; the head dequantizes the weight a vocabulary
tile at a time, so no full fp16 copy ever exists.

Head + loss in two passes over vocabulary tiles of TILE_V rows:
1. logits tile = x @ dequant(tile).T for every row (fp16, as under autocast); one Triton program
   per row reduces it to partial statistics: running max, sum of exp, sum of exp * logit, best
   value and its index, and the label's logit when it falls in the tile. The partials of all tiles
   are merged into log-sum-exp, entropy, argmax and the loss.
2. for the labelled rows only: the logits tile is recomputed, turned in place into
   coef * (softmax - onehot) in fp16 (as in fused_ce.py), and its product with the tile is added
   to an fp32 gradient. The backward pass only rescales that gradient.
"""

import torch
import triton
import triton.language as tl

from .fused_ce import IGNORE_INDEX, NEG

TILE_V = 8192  # vocabulary rows per tile: 8192 x 1536 fp16 = 25 MB, logits n x 8192
BLOCK_V = 2048
BLOCK_H = 2048
SLICE = 8192  # rows quantized at a time: 8192 x 1536 fp32 = 50 MB


@triton.jit
def int8_rows_dequant_kernel(q_ptr, scale_ptr, out_ptr, H, BLOCK_H: tl.constexpr):
    """out[r] = q[r] * scale[r] in fp16, one program per row."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < H
    q = tl.load(q_ptr + row * H + cols, mask=mask, other=0).to(tl.float32)
    scale = tl.load(scale_ptr + row)
    tl.store(out_ptr + row * H + cols, (q * scale).to(tl.float16), mask=mask)


@triton.jit
def head_partial_kernel(logits_ptr, stride, target_ptr, part_ptr, best_idx_ptr, target_logit_ptr,
                        tile_start, tile_index, n_tiles, TV, BLOCK_V: tl.constexpr,
                        NEG: tl.constexpr):
    """One program per row: this tile's max, sum exp(x - max), sum exp(x - max) * x, best value
    (index of its first occurrence), and the label's logit if the label is in this tile."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * stride
    lanes = tl.arange(0, BLOCK_V)
    m = tl.full([BLOCK_V], NEG, tl.float32)
    s = tl.zeros([BLOCK_V], tl.float32)
    t = tl.zeros([BLOCK_V], tl.float32)
    best = tl.full([BLOCK_V], NEG, tl.float32)
    best_idx = tl.zeros([BLOCK_V], tl.int32)
    for start in range(0, TV, BLOCK_V):
        cols = start + lanes
        valid = cols < TV
        x = tl.load(row_ptr + cols, mask=valid, other=NEG).to(tl.float32)
        new_m = tl.maximum(m, x)
        alpha = tl.exp(m - new_m)
        e = tl.where(valid, tl.exp(x - new_m), 0.0)
        s = s * alpha + e
        t = t * alpha + tl.where(valid, e * x, 0.0)
        m = new_m
        better = valid & (x > best)
        best = tl.where(better, x, best)
        best_idx = tl.where(better, cols, best_idx)
    row_max = tl.max(m, axis=0)
    rescale = tl.exp(m - row_max)
    total = tl.sum(s * rescale, axis=0)
    weighted = tl.sum(t * rescale, axis=0)
    top = tl.max(best, axis=0)
    argmax = tl.min(tl.where(best == top, best_idx, TV), axis=0)
    out = part_ptr + (row * n_tiles + tile_index) * 4
    tl.store(out, row_max)
    tl.store(out + 1, total)
    tl.store(out + 2, weighted)
    tl.store(out + 3, top)
    tl.store(best_idx_ptr + row * n_tiles + tile_index, argmax + tile_start)
    target = tl.load(target_ptr + row)
    local = target - tile_start
    if (target != -100) & (local >= 0) & (local < TV):
        tl.store(target_logit_ptr + row, tl.load(row_ptr + local).to(tl.float32))


@triton.jit
def head_grad_kernel(logits_ptr, stride, target_ptr, lse_ptr, coef_ptr, tile_start, TV,
                     BLOCK_V: tl.constexpr):
    """In place, one program per labelled row: coef * (exp(x - lse) - onehot(target)), fp16."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * stride
    target = tl.load(target_ptr + row) - tile_start
    lse = tl.load(lse_ptr + row)
    coef = tl.load(coef_ptr)
    lanes = tl.arange(0, BLOCK_V)
    for start in range(0, TV, BLOCK_V):
        cols = start + lanes
        valid = cols < TV
        x = tl.load(row_ptr + cols, mask=valid, other=0.0).to(tl.float32)
        grad = tl.exp(x - lse) - tl.where(cols == target, 1.0, 0.0)
        tl.store(row_ptr + cols, (grad * coef).to(tl.float16), mask=valid)


def quantize_rows(weight):
    """int8 rows with one fp32 absmax scale each (round to nearest). In slices of SLICE rows, so
    no full-size fp32 copy of the weight ever exists (that would be the run's memory peak)."""
    rows = weight.shape[0]
    q = torch.empty(weight.shape, dtype=torch.int8, device=weight.device)
    scale = torch.empty(rows, dtype=torch.float32, device=weight.device)
    for start in range(0, rows, SLICE):
        part = weight[start:start + SLICE].float()
        part_scale = part.abs().amax(dim=1).clamp_(min=1e-12).div_(127.0)
        scale[start:start + SLICE] = part_scale
        q[start:start + SLICE] = part.div_(part_scale[:, None]).round_().clamp_(-127, 127) \
            .to(torch.int8)
        del part
    return q, scale


def dequant_rows(q, scale, out=None):
    rows, width = q.shape
    if out is None:
        out = torch.empty((rows, width), dtype=torch.float16, device=q.device)
    if rows:
        int8_rows_dequant_kernel[(rows,)](q, scale, out, width, BLOCK_H=BLOCK_H, num_warps=4)
    return out


class Int8Embedding(torch.autograd.Function):
    """Rows of the int8 table as fp32 (the embedding output was widened to fp32 before)."""

    @staticmethod
    def forward(ctx, ids, q, scale):
        flat = ids.reshape(-1)
        rows = dequant_rows(q.index_select(0, flat), scale.index_select(0, flat))
        return rows.float().view(*ids.shape, q.shape[1])

    @staticmethod
    def backward(ctx, grad):
        return None, None, None


class FusedInt8HeadCE(torch.autograd.Function):
    """As fused_ce.FusedLMHeadCE, with the head weight given as int8 rows + scales."""

    @staticmethod
    def forward(ctx, hidden, q, scale, target, denominator):
        with torch.autocast("cuda", enabled=False):
            return FusedInt8HeadCE._forward(ctx, hidden, q, scale, target, denominator)

    @staticmethod
    def _forward(ctx, hidden, q, scale, target, denominator):
        n, width, vocab = hidden.shape[0], hidden.shape[1], q.shape[0]
        device = hidden.device
        with_grad = ctx.needs_input_grad[0]
        is_labelled = target != IGNORE_INDEX
        labelled = is_labelled.nonzero().squeeze(1)
        n_labelled = labelled.numel()
        order = torch.cat((labelled, (~is_labelled).nonzero().squeeze(1)))
        x16 = hidden.detach().index_select(0, order).to(torch.float16)
        ordered_target = target.index_select(0, order).contiguous()

        denominator = denominator.float().reshape(())
        grad_scale = torch.exp2(torch.floor(torch.log2(denominator.clamp(min=1.0))) + 8.0)
        coef = (grad_scale / denominator).reshape(1)

        n_tiles = triton.cdiv(vocab, TILE_V)
        part = torch.empty((n, n_tiles, 4), device=device, dtype=torch.float32)
        best_idx = torch.empty((n, n_tiles), device=device, dtype=torch.int32)
        target_logit = torch.zeros(n, device=device, dtype=torch.float32)
        tile = torch.empty((min(TILE_V, vocab), width), device=device, dtype=torch.float16)
        logits_buf = torch.empty((n, min(TILE_V, vocab)), device=device, dtype=torch.float16) \
            if n else None
        for index in range(n_tiles):
            start = index * TILE_V
            stop = min(vocab, start + TILE_V)
            w = dequant_rows(q[start:stop], scale[start:stop], tile[: stop - start])
            if n:
                logits = logits_buf[:, : stop - start]
                torch.matmul(x16, w.t(), out=logits)
                head_partial_kernel[(n,)](
                    logits, logits.stride(0), ordered_target, part, best_idx, target_logit,
                    start, index, n_tiles, stop - start, BLOCK_V=BLOCK_V, NEG=NEG, num_warps=8)

        top_m = part[..., 0].amax(dim=1, keepdim=True)
        rescale = torch.exp(part[..., 0] - top_m)
        total = (part[..., 1] * rescale).sum(1)
        weighted = (part[..., 2] * rescale).sum(1)
        lse = top_m[:, 0] + torch.log(total)
        entropy = lse - weighted / total
        tile_of_best = part[..., 3].argmax(dim=1, keepdim=True)  # first tile on ties
        argmax = best_idx.gather(1, tile_of_best)[:, 0].long()
        lab = ordered_target != IGNORE_INDEX
        loss = torch.where(lab, lse - target_logit, torch.zeros_like(lse))
        hit = (lab & (argmax == ordered_target)).to(torch.int32)

        grad = None
        if with_grad and n_labelled:
            grad = torch.zeros((n_labelled, width), device=device, dtype=torch.float32)
            xl, tl_, lse_l = x16[:n_labelled], ordered_target[:n_labelled], lse[:n_labelled]
            for index in range(n_tiles):
                start = index * TILE_V
                stop = min(vocab, start + TILE_V)
                w = dequant_rows(q[start:stop], scale[start:stop], tile[: stop - start])
                logits = logits_buf[:n_labelled, : stop - start]
                torch.matmul(xl, w.t(), out=logits)
                head_grad_kernel[(n_labelled,)](logits, logits.stride(0), tl_, lse_l, coef,
                                                start, stop - start, BLOCK_V=BLOCK_V,
                                                num_warps=8)
                grad += torch.matmul(logits, w).float()
        del tile, logits_buf

        def unorder(values):
            return torch.empty_like(values).index_copy_(0, order, values)

        loss, entropy, hit = unorder(loss), unorder(entropy), unorder(hit)
        total_loss = loss.sum() / denominator
        ctx.save_for_backward(grad, labelled, grad_scale)
        ctx.shape, ctx.hidden_dtype = hidden.shape, hidden.dtype
        ctx.mark_non_differentiable(loss, entropy, hit)
        return total_loss, loss, entropy, hit

    @staticmethod
    def backward(ctx, grad_total, _grad_loss, _grad_entropy, _grad_hit):
        grad, labelled, grad_scale = ctx.saved_tensors
        grad_hidden = torch.zeros(ctx.shape, device=grad_total.device, dtype=ctx.hidden_dtype)
        if grad is not None and labelled.numel():
            factor = grad_total.float() / grad_scale
            grad_hidden.index_copy_(0, labelled, (grad * factor).to(ctx.hidden_dtype))
        return grad_hidden, None, None, None, None


def fused_int8_head_ce(hidden, q, scale, target, denominator):
    return FusedInt8HeadCE.apply(hidden, q, scale, target, denominator)
