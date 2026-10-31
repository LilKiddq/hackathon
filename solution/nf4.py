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

from .wstream import STREAMER

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
                       GROUP2: tl.constexpr, BLOCK_BYTES: tl.constexpr, EXACT: tl.constexpr,
                       PACKED_FP16: tl.constexpr):
    pid = tl.program_id(0)
    byte_idx = pid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    valid = byte_idx < n_bytes
    packed = tl.load(packed_ptr + byte_idx, mask=valid, other=0).to(tl.int32)
    # both codes of a byte sit in the same quantization block (blocksize is even)
    block = byte_idx // HALF_BLOCK
    absmax8 = tl.load(absmax8_ptr + block, mask=valid, other=0).to(tl.int32)
    code2 = tl.load(code2_ptr + absmax8, mask=valid, other=0.0)
    absmax2 = tl.load(absmax2_ptr + block // GROUP2, mask=valid, other=0.0)
    if EXACT:
        # bitsandbytes rounds the product before adding the offset (two kernels); a plain
        # `a * b + c` here is contracted into one FMA and differs in the last bit
        scale = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [code2, absmax2],
                                          dtype=tl.float32, is_pure=True, pack=1)
    else:  # the Triton interpreter (CPU) has no PTX
        scale = code2 * absmax2
    scale = scale + tl.load(offset_ptr)
    high = tl.load(table_ptr + (packed >> 4), mask=valid, other=0.0) * scale
    low = tl.load(table_ptr + (packed & 15), mask=valid, other=0.0) * scale
    if PACKED_FP16:
        # out_ptr is the fp16 output viewed as int32: one 4-byte word per packed byte, the high
        # nibble's weight in the low half (little endian), so the stores are plain and coalesced
        high_bits = high.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        low_bits = low.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        tl.store(out_ptr + byte_idx, high_bits | (low_bits << 16), mask=valid)
    else:
        weights = tl.interleave(high, low).to(out_ptr.dtype.element_ty)
        out_idx = pid * (2 * BLOCK_BYTES) + tl.arange(0, 2 * BLOCK_BYTES)
        tl.store(out_ptr + out_idx, weights, mask=out_idx < n_out)


@triton.jit
def nf4_unit_dequant_kernel(packed_ptr, member_ptr, start_ptr, a8_base_ptr, a2_base_ptr,
                            offset_ptr, absmax8_ptr, code2_ptr, absmax2_ptr, table_ptr, out_ptr,
                            n_bytes, HALF_BLOCK: tl.constexpr, GROUP2: tl.constexpr,
                            BLOCK_BYTES: tl.constexpr, EXACT: tl.constexpr):
    """nf4_dequant_kernel for all the weights of a decoder layer in one launch: their packed
    bytes sit back to back in one buffer (wstream.py), each program stays inside one weight
    (its sizes are multiples of BLOCK_BYTES) and reads that weight's scales from the layer's
    concatenated absmax tables. Output: fp16 at twice the byte offset, as int32 words."""
    pid = tl.program_id(0)
    member = tl.load(member_ptr + pid)
    byte_idx = pid.to(tl.int64) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    valid = byte_idx < n_bytes
    local = byte_idx - tl.load(start_ptr + member)
    packed = tl.load(packed_ptr + byte_idx, mask=valid, other=0).to(tl.int32)
    block = local // HALF_BLOCK
    absmax8 = tl.load(absmax8_ptr + tl.load(a8_base_ptr + member) + block, mask=valid,
                      other=0).to(tl.int32)
    code2 = tl.load(code2_ptr + absmax8, mask=valid, other=0.0)
    absmax2 = tl.load(absmax2_ptr + tl.load(a2_base_ptr + member) + block // GROUP2, mask=valid,
                      other=0.0)
    if EXACT:
        scale = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [code2, absmax2],
                                          dtype=tl.float32, is_pure=True, pack=1)
    else:
        scale = code2 * absmax2
    scale = scale + tl.load(offset_ptr + member)
    high = tl.load(table_ptr + (packed >> 4), mask=valid, other=0.0) * scale
    low = tl.load(table_ptr + (packed & 15), mask=valid, other=0.0) * scale
    high_bits = high.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    low_bits = low.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    tl.store(out_ptr + byte_idx, high_bits | (low_bits << 16), mask=valid)


UNIT_BLOCK_BYTES = 512


class UnitMeta:
    """What nf4_unit_dequant_kernel needs for one layer's weights (built once, on the GPU)."""

    def __init__(self, states, starts, device):
        first = states[0]
        self.half_block = first.blocksize // 2
        self.group2 = first.state2.blocksize
        self.code2 = first.state2.code
        nbytes = [state.shape[0] * state.shape[1] // 2 for state in states]
        self.n_bytes = starts[-1] + nbytes[-1]
        members = []
        for number, (start, size) in enumerate(zip(starts, nbytes)):
            members += [number] * (size // UNIT_BLOCK_BYTES)
        self.member = torch.tensor(members, dtype=torch.int32, device=device)
        self.start = torch.tensor(starts, dtype=torch.int64, device=device)
        a8 = [state.absmax.numel() for state in states]
        a2 = [state.state2.absmax.numel() for state in states]
        self.a8_base = torch.tensor([sum(a8[:i]) for i in range(len(a8))], dtype=torch.int64,
                                    device=device)
        self.a2_base = torch.tensor([sum(a2[:i]) for i in range(len(a2))], dtype=torch.int64,
                                    device=device)
        offsets = [state.offset if torch.is_tensor(state.offset)
                   else torch.tensor(float(state.offset)) for state in states]
        self.offsets = torch.stack([o.float().reshape(()).to(device) for o in offsets])
        # the layer's absmax tables, concatenated; each weight's quant state now views its part
        self.absmax8 = torch.cat([state.absmax.reshape(-1) for state in states])
        self.absmax2 = torch.cat([state.state2.absmax.reshape(-1).float() for state in states])
        for i, state in enumerate(states):
            state.absmax = self.absmax8[int(self.a8_base[i]):int(self.a8_base[i]) + a8[i]] \
                .view(state.absmax.shape)
            state.state2.absmax = \
                self.absmax2[int(self.a2_base[i]):int(self.a2_base[i]) + a2[i]] \
                .view(state.state2.absmax.shape)

    @staticmethod
    def fits(states, starts, sizes):
        """One launch works when the weights are NF4 with the same block sizes and code, each a
        multiple of the program size, back to back (no gaps between them)."""
        if len(states) < 2 or not all(supported(state) for state in states):
            return False
        first = states[0]
        for state, start, size in zip(states, starts, sizes):
            if (state.blocksize != first.blocksize
                    or state.state2.blocksize != first.state2.blocksize
                    or state.shape[1] % 2 or size % UNIT_BLOCK_BYTES
                    or size != state.shape[0] * state.shape[1] // 2
                    or state.state2.code.shape != first.state2.code.shape
                    or not torch.equal(state.state2.code, first.state2.code)):
                return False
        return all(starts[i] + sizes[i] == starts[i + 1] for i in range(len(starts) - 1))


def dequantize_unit(packed, meta):
    """fp16 weights of a whole layer from its packed bytes (one GPU buffer) in one launch: the
    weight at byte offset o with n elements is out[2 * o: 2 * o + n]."""
    out = torch.empty(2 * meta.n_bytes, dtype=torch.float16, device=packed.device)
    programs = meta.member.numel()
    nf4_unit_dequant_kernel[(programs,)](
        packed.reshape(-1), meta.member, meta.start, meta.a8_base, meta.a2_base, meta.offsets,
        meta.absmax8, meta.code2, meta.absmax2, _table(packed.device), out.view(torch.int32),
        meta.n_bytes, HALF_BLOCK=meta.half_block, GROUP2=meta.group2,
        BLOCK_BYTES=UNIT_BLOCK_BYTES, EXACT=packed.is_cuda, num_warps=4, enable_fp_fusion=False)
    return out


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
    packed_fp16 = dtype == torch.float16 and quant_state.shape[1] % 2 == 0
    nf4_dequant_kernel[(triton.cdiv(n_bytes, BLOCK_BYTES),)](
        flat, quant_state.absmax, state2.code, state2.absmax, offset.float().reshape(1),
        _table(packed.device), out.view(torch.int32) if packed_fp16 else out, n_bytes, n_out,
        HALF_BLOCK=quant_state.blocksize // 2, GROUP2=state2.blocksize, BLOCK_BYTES=BLOCK_BYTES,
        EXACT=packed.is_cuda, PACKED_FP16=packed_fp16, num_warps=4, enable_fp_fusion=False)
    return out


class NF4Linear(torch.autograd.Function):
    """x @ dequant(W).T for a frozen NF4 weight; the weight is dequantized again for backward."""

    @staticmethod
    def forward(ctx, x, packed, quant_state, bias=None):
        weight = STREAMER.weight(packed, quant_state, x.dtype)
        ctx.quant_state = quant_state
        ctx.save_for_backward(packed)
        return torch.nn.functional.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad):
        (packed,) = ctx.saved_tensors
        weight = STREAMER.weight(packed, ctx.quant_state, grad.dtype)
        return grad @ weight, None, None, None


def patch_bnb_matmul_4bit():
    """`bnb.nn.Linear4bit.forward` calls `bnb.matmul_4bit`, looked up at call time. Training
    calls with an fp16 input that needs a gradient go through NF4Linear (a frozen bias, as
    q/k/v_proj have, is added by the same F.linear bitsandbytes uses); everything else (eval,
    generation, a trainable bias, other formats) keeps bitsandbytes' own path."""
    import bitsandbytes

    if getattr(bitsandbytes.matmul_4bit, "_kc_patched", False):
        return
    original = bitsandbytes.matmul_4bit

    def matmul_4bit(A, B, quant_state, out=None, bias=None):
        if (out is None and A.is_cuda and A.dtype == torch.float16
                and (bias is None or (not bias.requires_grad and bias.dtype == A.dtype))
                and torch.is_grad_enabled() and A.requires_grad and A.numel() > 0
                and supported(quant_state) and A.shape[-1] == quant_state.shape[1]):
            return NF4Linear.apply(A, B, quant_state, bias)
        if A.is_cuda and not B.is_cuda:  # a weight streamed from the host (wstream.py)
            B = STREAMER.get(B)
        return original(A, B, quant_state, out=out, bias=bias)

    matmul_4bit._kc_patched = True
    bitsandbytes.matmul_4bit = matmul_4bit
