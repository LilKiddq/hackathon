"""Where does GPU memory go outside PyTorch's allocator? (diagnostic, not part of the solution)

Run from the repo root on a free GPU:  python tools/diag_mem.py
Prints nvidia-smi memory.used (what the harness scores) next to the allocator's reserved memory
after each step: CUDA context, cuBLAS, SDPA, our Triton kernels, bitsandbytes, ...
"""
import os
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.getcwd())


def smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return sum(int(line.split(",")[1]) for line in out.splitlines() if "," in line)


print(f"{'before torch':52s} smi {smi():5d} MiB", flush=True)
import torch  # noqa: E402

last = [0]


def mark(name):
    torch.cuda.synchronize()
    time.sleep(0.2)
    used = smi()
    res = torch.cuda.memory_reserved() >> 20
    print(f"{name:52s} smi {used:5d} MiB (+{used - last[0]:4d})  reserved {res:5d}  outside allocator {used - res:5d}",
          flush=True)
    last[0] = used


torch.empty(1, device="cuda")
mark("CUDA context (torch.empty)")
import transformers  # noqa: E402,F401

mark("import transformers")
h = torch.float16
x = torch.randn(826, 1536, device="cuda", dtype=h)
w = torch.randn(8960, 1536, device="cuda", dtype=h)
y = x @ w.t()
mark("first cuBLAS GEMM fp16")
dy = torch.randn(826, 8960, device="cuda", dtype=h)
dx = dy @ w
a = torch.randn(16, 1536, device="cuda", dtype=h)
b = torch.randn(1536, 16, device="cuda", dtype=h)
z = (x @ a.t()) @ b.t()
xf = x.float().requires_grad_()
af = a.float().requires_grad_()
(xf @ af.t()).sum().backward()
torch.addmm(torch.zeros(1536, device="cuda", dtype=h), x, x.new_empty(1536, 1536).normal_())
mark("more GEMM shapes (dx, LoRA, fp32, addmm)")
torch.nn.functional.dropout(x, 0.05, training=True)
mark("dropout")
q = torch.randn(1, 12, 300, 128, device="cuda", dtype=h, requires_grad=True)
k = torch.randn(1, 12, 300, 128, device="cuda", dtype=h, requires_grad=True)
v = torch.randn(1, 12, 300, 128, device="cuda", dtype=h, requires_grad=True)
o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
mark("SDPA forward (fp16, causal)")
o.sum().backward()
mark("SDPA backward")
torch.cuda.empty_cache()
mark("empty_cache")

from solution import warmup  # noqa: E402

t = time.perf_counter()
warmup._compile_all()
print(f"  (our kernels compiled in {time.perf_counter() - t:.1f} s)")
mark("our Triton kernels (warm-up set)")

import triton  # noqa: E402

print("Triton", triton.__version__)
import solution  # noqa: E402

from triton.runtime.jit import JITFunction  # noqa: E402


def compiled(fn):
    found = []
    for attr in ("cache", "device_caches"):
        table = getattr(fn, attr, None)
        if not isinstance(table, dict):
            continue
        for entry in table.values():
            entry = entry[0] if isinstance(entry, tuple) else entry
            if isinstance(entry, dict):
                found.extend(entry.values())
    return found


seen = set()
for mod in list(sys.modules.values()):
    if not getattr(mod, "__name__", "").startswith("solution"):
        continue
    for name, fn in list(vars(mod).items()):
        while not isinstance(fn, JITFunction) and hasattr(fn, "fn"):
            fn = fn.fn
        if not isinstance(fn, JITFunction) or id(fn) in seen:
            continue
        seen.add(id(fn))
        for kern in compiled(fn):
            meta = getattr(kern, "metadata", None)
            print(f"  {name:30s} regs {getattr(kern, 'n_regs', '?')!s:>4}  spills {getattr(kern, 'n_spills', '?')!s:>5}"
                  f"  warps {getattr(meta, 'num_warps', '?')}  shared {getattr(meta, 'shared', '?')}", flush=True)

import bitsandbytes.functional as bnbf  # noqa: E402

wq, state = bnbf.quantize_4bit(torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16),
                               quant_type="nf4", compress_statistics=True)
mark("bitsandbytes quantize_4bit")
bnbf.dequantize_4bit(wq, state)
mark("bitsandbytes dequantize_4bit")
s1 = torch.cuda.Stream()
with torch.cuda.stream(s1):
    yy = x @ w.t()
mark("GEMM on a second stream")
pin = torch.empty(256 << 20, dtype=torch.uint8, pin_memory=True)
mark("256 MiB pinned host buffer")
gpu = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
gpu.copy_(pin, non_blocking=True)
mark("H2D copy 256 MiB")
del gpu
torch.cuda.empty_cache()
mark("final empty_cache")
