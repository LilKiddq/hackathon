"""Compile our Triton kernels in the background while Soup loads the model.

Triton compiles a kernel the first time it is launched, so in a plain run every kernel compiles
during the first training steps, while the GPU waits. Soup spends its first ~15 s loading and
quantizing the model, mostly reading files. `start()` (called from apply()) launches each
kernel once on tiny tensors in a background thread, so the compilation overlaps that loading.
It is the same compilation, inside the same measured run, starting from the same empty caches;
only the order changes. The tiny tensors match the real launches in dtype and in the integer
properties Triton specializes on (divisibility by 16), so the real launches find the compiled
kernels. `wait()` runs before the first training step.
"""

import threading
import types

import torch

_thread = None


def _compile_all():
    from .add_rmsnorm import AddRMSNorm
    from .fused_ce import fused_lm_head_ce
    from .nf4 import dequantize_nf4
    from .rmsnorm import rms_norm
    from .rope import RoPE
    from .swiglu import swiglu

    device = torch.device("cuda")
    hidden = 1536  # Qwen2.5-1.5B: the norms' and the head's width

    x = torch.randn(2, hidden, device=device, requires_grad=True)
    weight = torch.ones(hidden, device=device)
    rms_norm(x, weight, 1e-6).sum().backward()

    residual = torch.randn(2, hidden, device=device, requires_grad=True)
    delta = torch.randn(2, hidden, device=device, dtype=torch.float16, requires_grad=True)
    s, y = AddRMSNorm.apply(residual, delta, weight, 1e-6)
    (s.sum() + y.sum()).backward()

    q = torch.randn(1, 5, 12, 128, device=device).transpose(1, 2).requires_grad_()
    k = torch.randn(1, 5, 2, 128, device=device).transpose(1, 2).requires_grad_()
    cos, sin = torch.rand(1, 5, 128, device=device), torch.rand(1, 5, 128, device=device)
    q_out, k_out = RoPE.apply(q, k, cos, sin)
    # dense gradients: .sum() would hand back stride-0 ones, another specialization
    torch.autograd.backward((q_out, k_out), (torch.ones_like(q_out), torch.ones_like(k_out)))

    rows, width, vocab = 8, 64, 4096  # the vocabulary only has to share 151936's divisibility
    head_in = torch.randn(rows, width, device=device, requires_grad=True)
    head = torch.randn(vocab, width, device=device, dtype=torch.float16)
    target = torch.randint(0, vocab, (rows,), device=device)
    target[::2] = -100
    loss = fused_lm_head_ce(head_in, head, target, (target != -100).sum().float())[0]
    loss.backward()

    state2 = types.SimpleNamespace(code=torch.linspace(-1, 1, 256, device=device),
                                   absmax=torch.ones(1, device=device), blocksize=256)
    state = types.SimpleNamespace(shape=(32, 64), blocksize=64, state2=state2,
                                  absmax=torch.zeros(32, device=device, dtype=torch.uint8),
                                  offset=torch.zeros((), device=device))
    dequantize_nf4(torch.zeros(1024, device=device, dtype=torch.uint8), state)

    gate = torch.randn(2, 8960, device=device, dtype=torch.float16, requires_grad=True)
    up = torch.randn(2, 8960, device=device, dtype=torch.float16, requires_grad=True)
    swiglu(gate, up).sum().backward()
    torch.cuda.synchronize()


def _run():
    # on the default stream: a stream of its own would get its own cuBLAS workspace (+44 MiB peak)
    try:
        _compile_all()
    except Exception:  # noqa: BLE001 - a failed warm-up only means the kernels compile later
        pass


def start():
    """Start the background compilation once, when there is a GPU."""
    global _thread
    if _thread is None and torch.cuda.is_available():
        # not a daemon: a process that ends early (the regression tests) waits for it
        _thread = threading.Thread(target=_run, name="kernel-warmup")
        _thread.start()


def wait():
    if _thread is not None:
        _thread.join()
