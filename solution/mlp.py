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

from .offload import swiglu_maybe_offloaded
from .swiglu import swiglu  # the example's SwiGLU kernel; it does not count as ours


def _fp16_4bit(module):
    return getattr(module, "compute_dtype", None) == torch.float16 and \
        type(module).__name__ == "Linear4bit"


def patch_qwen2_mlp_fp16():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls.forward, "_kc_patched", False):
        return
    original = cls.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if (act != "silu" or not x.is_cuda or x.dtype != torch.float32
                or not all(_fp16_4bit(m) for m in (self.gate_proj, self.up_proj, self.down_proj))):
            return original(self, x)
        gate = self.gate_proj(x.to(torch.float16))
        up = self.up_proj(x.to(torch.float16))
        # in training, gate and up wait for the backward pass in host memory (offload.py)
        return self.down_proj(swiglu_maybe_offloaded(gate, up, swiglu))

    forward._kc_patched = True
    cls.forward = forward
