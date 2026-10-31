"""Qwen2 MLP on the fp16 path: the same numbers with half the saved activations.

In base Soup the residual stream is fp32 (prepare_model_for_kbit_training), so each 4-bit
projection gets an fp32 input, casts it to fp16 for the matmul and casts its fp16 result back
to fp32 (`Linear4bit.forward` ends with `.to(inp_dtype)`). The MLP's gate and up outputs,
[batch * seq, 8960] each, are therefore kept for the backward pass in fp32: about 147 MB per
layer, 4 GB over 28 layers.

Here gate_proj and up_proj each get their own fp16 cast of the input (as the 4-bit layers make
it themselves, so their input gradients still meet in fp32), their fp16 results go straight to
SwiGLU, and down_proj returns fp16, which the residual add widens to fp32. Every value that was
fp16 before is the same fp16 value now, and every fp32 value is the same fp32 value: an fp16
value cast to fp32 and back is unchanged, so only the copies go away, not precision.
"""

import torch
import triton

from .nf4 import dequantize_nf4, supported
from .offload import swiglu_maybe_offloaded
from .swiglu import BLOCK, example_swiglu_bwd_kernel, example_swiglu_fwd_kernel
from .swiglu import swiglu  # the example's SwiGLU kernel; it does not count as ours
from .wstream import STREAMER


def _fp16_4bit(module):
    return getattr(module, "compute_dtype", None) == torch.float16 and \
        type(module).__name__ == "Linear4bit"


def _weight(module):
    """The fp16 weight of a 4-bit projection (streamed in from the host by wstream.py)."""
    weight = module.weight
    return STREAMER.weight(weight, weight.quant_state, torch.float16)


class FusedMLP(torch.autograd.Function):
    """The whole Qwen2 MLP in training as one autograd node: down(silu(gate(x)) * up(x)).

    The same kernels and matmuls as the module path (NF4 dequantization, cuBLAS, the example's
    SwiGLU kernels), so the same numbers: x is cast to fp16 once (each 4-bit layer cast it on its
    own), and only that fp16 input is kept for the backward pass (offloaded with the layer,
    offload.py), which recomputes gate and up from it.
    One Python node instead of three Linear4bit -> matmul_4bit -> NF4Linear chains, a SwiGLU node
    and two casts, both ways: the step is limited by the CPU launching work, not by the GPU."""

    @staticmethod
    def forward(ctx, x, gate_proj, up_proj, down_proj):
        x16 = x if x.dtype == torch.float16 else x.to(torch.float16)
        flat = x16.reshape(-1, x16.shape[-1])
        gate = torch.nn.functional.linear(flat, _weight(gate_proj))
        up = torch.nn.functional.linear(flat, _weight(up_proj))
        act = torch.empty_like(gate)
        n = gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](gate, up, act, n, BLOCK=BLOCK)
        del gate, up
        out = torch.nn.functional.linear(act, _weight(down_proj))
        # only the fp16 input is kept (3 KB per token instead of gate and up, 35.8 KB): the
        # backward pass recomputes gate and up with the same matmuls (the same numbers), which
        # costs two matmuls there but saves two thirds of the bytes offloaded through PCIe
        ctx.save_for_backward(flat)
        ctx.modules, ctx.x_dtype, ctx.x_shape = (gate_proj, up_proj, down_proj), x.dtype, x.shape
        return out.view(*x.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad):
        (x16,) = ctx.saved_tensors
        gate_proj, up_proj, down_proj = ctx.modules
        flat = grad.reshape(-1, grad.shape[-1]).to(torch.float16)
        dact = flat @ _weight(down_proj)  # backward order: down, up, gate (wstream's window)
        w_up, w_gate = _weight(up_proj), _weight(gate_proj)
        up = torch.nn.functional.linear(x16, w_up)  # recomputed: the forward's exact values
        gate = torch.nn.functional.linear(x16, w_gate)
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](dact, gate, up, dgate, dup, n,
                                                           BLOCK=BLOCK)
        del dact, gate, up
        # as the module path: each input gradient in fp16, the two summed in x's dtype
        dx = (dup @ w_up).to(ctx.x_dtype)
        dx.add_(dgate @ w_gate)
        return dx.view(ctx.x_shape), None, None, None


def _fusable(mlp, x):
    if not (x.is_cuda and x.dtype in (torch.float32, torch.float16) and x.requires_grad
            and torch.is_grad_enabled()):
        return False
    for module in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        weight = getattr(module, "weight", None)
        state = getattr(weight, "quant_state", None)
        if (not _fp16_4bit(module) or getattr(module, "bias", None) is not None
                or state is None or weight.dtype != torch.uint8 or not supported(state)):
            return False
    return True


def patch_qwen2_mlp_fp16():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls.forward, "_kc_patched", False):
        return
    original = cls.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act == "silu" and _fusable(self, x):
            return FusedMLP.apply(x, self.gate_proj, self.up_proj, self.down_proj)
        if (act != "silu" or not x.is_cuda or x.dtype != torch.float32
                or not all(_fp16_4bit(m) for m in (self.gate_proj, self.up_proj, self.down_proj))):
            return original(self, x)
        gate = self.gate_proj(x.to(torch.float16))
        up = self.up_proj(x.to(torch.float16))
        # in training, gate and up wait for the backward pass in host memory (offload.py)
        return self.down_proj(swiglu_maybe_offloaded(gate, up, swiglu))

    forward._kc_patched = True
    cls.forward = forward
