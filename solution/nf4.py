"""NF4 dequantization in one Triton launch, for bitsandbytes' 4-bit layers in training.

Every 4-bit layer of the model (7 projections x 28 layers) dequantizes its weight to fp16 in the
forward pass and again in the backward pass. With double quantization, which Soup's 4-bit setup
uses, bitsandbytes does that in steps: a dequantize_blockwise launch for the 8-bit absmax, an
add for the offset, then the 4-bit dequantize launch. Here one kernel reads the packed bytes,
rebuilds each block's scale (code2[absmax8] * absmax2 + offset) and writes the fp16 weight,
then cuBLAS does the matmul as before. Only the packed weight is kept for the backward pass, as
in bitsandbytes.

The format (QLoRA, Dettmers et al. 2023): two 4-bit codes per byte, the high nibble first; each
code indexes the 16-value NF4 table; blocks of `blocksize` values share one absmax, and the
absmaxes themselves are 8-bit blockwise-quantized in groups of 256. The code is our own.
"""

import torch
import triton
import triton.language as tl

BLOCK_BYTES = 1024  # packed bytes per program -> 2048 weights

NF4_TABLE = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax8_ptr, code2_ptr, absmax2_ptr, offset_ptr, table_ptr,
                       out_ptr, n_bytes, n_out, HALF_BLOCK: tl.constexpr,
                       GROUP2: tl.constexpr, BLOCK_BYTES: tl.constexpr):
    pid = tl.program_id(0)
    byte_idx = pid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    valid = byte_idx < n_bytes
    packed = tl.load(packed_ptr + byte_idx, mask=valid, other=0).to(tl.int32)
    # both codes of a byte sit in the same quantization block (blocksize is even)
    block = byte_idx // HALF_BLOCK
    absmax8 = tl.load(absmax8_ptr + block, mask=valid, other=0).to(tl.int32)
    scale = tl.load(code2_ptr + absmax8, mask=valid, other=0.0) \
        * tl.load(absmax2_ptr + block // GROUP2, mask=valid, other=0.0)
    scale = scale + tl.load(offset_ptr)
    high = tl.load(table_ptr + (packed >> 4), mask=valid, other=0.0) * scale
    low = tl.load(table_ptr + (packed & 15), mask=valid, other=0.0) * scale
    weights = tl.interleave(high, low).to(out_ptr.dtype.element_ty)
    out_idx = pid * (2 * BLOCK_BYTES) + tl.arange(0, 2 * BLOCK_BYTES)
    tl.store(out_ptr + out_idx, weights, mask=out_idx < n_out)


_tables = {}


def _table(device):
    table = _tables.get(device)
    if table is None:
        table = torch.tensor(NF4_TABLE, dtype=torch.float32, device=device)
        _tables[device] = table
    return table


def supported(quant_state):
    """NF4 with double-quantized absmax in groups of 256: Soup's `quantization: 4bit`."""
    state2 = getattr(quant_state, "state2", None)
    return (getattr(quant_state, "quant_type", None) == "nf4"
            and getattr(quant_state, "nested", False) and state2 is not None
            and state2.blocksize == 256 and quant_state.blocksize % 2 == 0
            and len(quant_state.shape) == 2)


def dequantize_nf4(packed, quant_state, dtype=torch.float16):
    """The [N, K] weight of a packed NF4 tensor, like bitsandbytes' dequantize_4bit."""
    n_out = quant_state.shape[0] * quant_state.shape[1]
    out = torch.empty(tuple(quant_state.shape), device=packed.device, dtype=dtype)
    flat = packed.reshape(-1)
    if flat.dtype != torch.uint8:
        flat = flat.view(torch.uint8)
    n_bytes = (n_out + 1) // 2
    state2 = quant_state.state2
    offset = quant_state.offset
    if not torch.is_tensor(offset):
        offset = torch.tensor(float(offset), device=packed.device)
    nf4_dequant_kernel[(triton.cdiv(n_bytes, BLOCK_BYTES),)](
        flat, quant_state.absmax, state2.code, state2.absmax, offset.float().reshape(1),
        _table(packed.device), out, n_bytes, n_out,
        HALF_BLOCK=quant_state.blocksize // 2, GROUP2=state2.blocksize, BLOCK_BYTES=BLOCK_BYTES,
        num_warps=4)
    return out


class NF4Linear(torch.autograd.Function):
    """x @ dequant(W).T for a frozen NF4 weight; the weight is dequantized again for backward."""

    @staticmethod
    def forward(ctx, x, packed, quant_state):
        weight = dequantize_nf4(packed, quant_state, x.dtype)
        ctx.quant_state = quant_state
        ctx.save_for_backward(packed)
        return torch.nn.functional.linear(x, weight)

    @staticmethod
    def backward(ctx, grad):
        (packed,) = ctx.saved_tensors
        weight = dequantize_nf4(packed, ctx.quant_state, grad.dtype)
        return grad @ weight, None, None


def patch_bnb_matmul_4bit():
    """`bnb.nn.Linear4bit.forward` calls `bnb.matmul_4bit`, looked up at call time. Training
    calls with an fp16 input that needs a gradient go through NF4Linear; everything else (eval,
    generation, bias, other formats) keeps bitsandbytes' own path."""
    import bitsandbytes

    if getattr(bitsandbytes.matmul_4bit, "_kc_patched", False):
        return
    original = bitsandbytes.matmul_4bit

    def matmul_4bit(A, B, quant_state, out=None, bias=None):
        if (out is None and bias is None and A.is_cuda and A.dtype == torch.float16
                and torch.is_grad_enabled() and A.requires_grad and A.numel() > 0
                and supported(quant_state) and A.shape[-1] == quant_state.shape[1]):
            return NF4Linear.apply(A, B, quant_state)
        return original(A, B, quant_state, out=out, bias=bias)

    matmul_4bit._kc_patched = True
    bitsandbytes.matmul_4bit = matmul_4bit
