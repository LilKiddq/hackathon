"""The tied embedding / lm_head held in int8 (one fp32 scale per vocabulary row) during training.

The tied 151936 x 1536 weight is the largest tensor left on the GPU: 467 MB in fp16. Here it is
stored as int8 with a per-row absmax scale (233 MB + 0.6 MB), the classic LLM.int8()-style
row-wise quantization (Dettmers et al. 2022). This is an approximation, and the only one in the
solution: each weight moves by at most half a quantization step (absmax / 254), which shifts the
logits by a few hundredths and the loss by far less than the task's tolerances (checks 5-7).
The embedding lookup dequantizes the looked-up rows; the head dequantizes the weight a vocabulary
tile at a time, so no full fp16 copy ever exists.

Head + loss in one pass over vocabulary tiles of TILE_V rows:
1. logits tile = x @ dequant(tile).T for every row (fp16, as under autocast); one Triton program
   per row reduces it to partial statistics: running max, sum of exp, sum of exp * logit, best
   value and its index, and the label's logit when it falls in the tile. The partials of all tiles
   are merged into log-sum-exp, entropy, argmax and the loss.
   In the same pass, for the labelled rows, the tile is turned in place into exp(x - running max)
   (fp16) and its product with the tile is added to an fp32 sum, rescaled whenever the row's
   running max grows (the online softmax of FlashAttention, applied to the head's gradient). At
   the end softmax @ W = sum / sum-of-exp, and the gradient is softmax @ W - W[target]: one pass,
   one matmul per tile for the logits and one for the gradient. The backward pass only rescales.
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


@triton.jit(do_not_specialize=["tile_start", "tile_index", "n_tiles", "TV"])
def head_partial_kernel(logits_ptr, stride, target_ptr, part_ptr, best_idx_ptr, target_logit_ptr,
                        run_max_ptr, alpha_ptr, n_grad, tile_start, tile_index, n_tiles, TV,
                        WITH_GRAD: tl.constexpr, BLOCK_V: tl.constexpr, NEG: tl.constexpr):
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
    if WITH_GRAD:
        if row < n_grad:
            # online softmax for the gradient: the row's running max over the tiles so far; the
            # tile becomes exp(x - max) in place, and the sum so far is rescaled by alpha
            old_max = tl.load(run_max_ptr + row)
            new_max = tl.maximum(old_max, row_max)
            tl.store(run_max_ptr + row, new_max)
            tl.store(alpha_ptr + row, tl.exp(old_max - new_max))
            for start in range(0, TV, BLOCK_V):
                cols = start + lanes
                valid = cols < TV
                x = tl.load(row_ptr + cols, mask=valid, other=NEG).to(tl.float32)
                tl.store(row_ptr + cols, tl.exp(x - new_max).to(tl.float16), mask=valid)


def quantize_rows(weight, out=None):
    """int8 rows with one fp32 absmax scale each (round to nearest). In slices of SLICE rows, so
    no full-size fp32 copy of the weight ever exists (that would be the run's memory peak).
    `out` (e.g. a pinned host tensor) receives the int8 rows instead of a new GPU tensor; the
    scales stay with the weight."""
    rows = weight.shape[0]
    q = out if out is not None else \
        torch.empty(weight.shape, dtype=torch.int8, device=weight.device)
    scale = torch.empty(rows, dtype=torch.float32, device=weight.device)
    for start in range(0, rows, SLICE):
        part = weight[start:start + SLICE].float()
        part_scale = part.abs().amax(dim=1).clamp_(min=1e-12).div_(127.0)
        scale[start:start + SLICE] = part_scale
        q[start:start + SLICE].copy_(part.div_(part_scale[:, None]).round_().clamp_(-127, 127)
                                     .to(torch.int8))
        del part
    return q, scale


_COPY_STREAMS = {}
_STAGING = {}


def _copy_stream(device):
    stream = _COPY_STREAMS.get(device)
    if stream is None:
        stream = _COPY_STREAMS[device] = torch.cuda.Stream(device=device)
    return stream


def _host_rows(q, flat):
    """Rows of a pinned host table for GPU ids: gathered on the CPU into a pinned staging
    buffer, then one copy to the GPU. Reading the ids waits for the stream, so the previous
    call's copy out of the staging buffer is finished before it is overwritten."""
    ids = flat.to("cpu")
    count = ids.numel()
    staging = _STAGING.get(q.data_ptr())
    if staging is None or staging.shape[0] < count:
        staging = torch.empty((max(count, 2048), q.shape[1]), dtype=q.dtype, pin_memory=True)
        _STAGING[q.data_ptr()] = staging
    torch.index_select(q, 0, ids, out=staging[:count])
    return staging[:count].to(flat.device, non_blocking=True)


class _Tiles:
    """The head's vocabulary tiles of int8 rows. From a GPU table: slices. From a pinned host
    table: copied to two GPU buffers on a second stream, the next tile while this one is used,
    so the copy (12 MB, ~1 ms) hides behind the tile's matmuls."""

    def __init__(self, q, device):
        self.q, self.vocab = q, q.shape[0]
        self.host = not q.is_cuda and torch.device(device).type == "cuda"
        if self.host:
            self.stream = _copy_stream(device)
            self.bufs = [torch.empty((min(TILE_V, self.vocab), q.shape[1]), dtype=q.dtype,
                                     device=device) for _ in range(2)]
            queued = torch.cuda.Event()
            queued.record(torch.cuda.current_stream(device))  # the buffers' memory is free
            self.free = [queued, queued]
            self.copied = [None, None]
            self._issue(0)

    def _issue(self, index):
        start = index * TILE_V
        if start >= self.vocab:
            return
        stop = min(self.vocab, start + TILE_V)
        slot = index % 2
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.free[slot])
            self.bufs[slot][: stop - start].copy_(self.q[start:stop], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        self.copied[slot] = done

    def get(self, index, start, stop):
        if not self.host:
            return self.q[start:stop]
        self._issue(index + 1)  # its buffer was released after tile index - 1
        slot = index % 2
        torch.cuda.current_stream().wait_event(self.copied[slot])
        return self.bufs[slot][: stop - start]

    def release(self, index):
        if self.host:
            used = torch.cuda.Event()
            used.record(torch.cuda.current_stream())
            self.free[index % 2] = used


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
        picked = q.index_select(0, flat) if q.device == flat.device else _host_rows(q, flat)
        rows = dequant_rows(picked, scale.index_select(0, flat))
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
        n_grad = n_labelled if with_grad else 0
        # sum over tiles of exp(logits - running max) @ tile, for the labelled rows (fp32)
        acc = torch.zeros((max(n_grad, 1), width), device=device, dtype=torch.float32)
        run_max = torch.full((max(n_grad, 1),), NEG, device=device, dtype=torch.float32)
        alpha = torch.empty_like(run_max)
        tile = torch.empty((min(TILE_V, vocab), width), device=device, dtype=torch.float16)
        logits_buf = torch.empty((n, min(TILE_V, vocab)), device=device, dtype=torch.float16) \
            if n else None
        tiles = _Tiles(q, device)
        labels = ordered_target[:n_grad]
        # W[target] of each labelled row, picked from the tile that holds it (no gather from a
        # host table, no wait): the same dequantized values as the head's own tile
        target_rows = torch.zeros((n_grad, width), device=device, dtype=torch.float32) \
            if n_grad else None
        for index in range(n_tiles):
            start = index * TILE_V
            stop = min(vocab, start + TILE_V)
            w = dequant_rows(tiles.get(index, start, stop), scale[start:stop],
                             tile[: stop - start])
            tiles.release(index)
            if n_grad:
                local = labels - start
                inside = ((local >= 0) & (local < stop - start)).unsqueeze(1)
                picked = w.index_select(0, local.clamp(0, stop - start - 1))
                target_rows.add_(picked.float().mul_(inside))
            if n:
                logits = logits_buf[:, : stop - start]
                torch.matmul(x16, w.t(), out=logits)
                head_partial_kernel[(n,)](
                    logits, logits.stride(0), ordered_target, part, best_idx, target_logit,
                    run_max, alpha, n_grad, start, index, n_tiles, stop - start,
                    WITH_GRAD=n_grad > 0, BLOCK_V=BLOCK_V, NEG=NEG, num_warps=8)
                if n_grad:
                    acc.mul_(alpha[:, None])
                    acc += torch.matmul(logits[:n_grad], w).float()
        del tile, logits_buf, tiles

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
        if n_grad:
            # d loss_row / d x = softmax @ W - W[target]; the running max ends at the row's max,
            # so acc / total is softmax @ W
            grad = (acc / total[:n_grad, None] - target_rows) * coef

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
