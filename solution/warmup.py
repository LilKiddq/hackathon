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

import atexit
import threading
import types

import torch

_thread = None
TIMEOUT_S = 120  # the warm-up takes ~10 s; past this the kernels just compile on first use


def _compile_all():
    from .add_rmsnorm import AddRMSNorm
    from .fused_ce import fused_lm_head_ce
    from .nf4 import dequantize_nf4
    from .rmsnorm import rms_norm
    from .rope import RoPE
    from .swiglu import swiglu

    device = torch.device("cuda")
    hidden = 1536  # Qwen2.5-1.5B: the norms' and the head's width
    # A generator of its own: Soup seeds the global one while this thread may still be running,
    # and drawing from it here would shift the random numbers training uses (LoRA dropout).
    gen = torch.Generator(device=device)
    gen.manual_seed(0)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, device=device, dtype=dtype, generator=gen)

    x = randn(2, hidden).requires_grad_()
    weight = torch.ones(hidden, device=device)
    rms_norm(x, weight, 1e-6).sum().backward()

    residual = randn(2, hidden).requires_grad_()
    delta = randn(2, hidden, dtype=torch.float16).requires_grad_()
    s, y = AddRMSNorm.apply(residual, delta, weight, 1e-6)
    (s.sum() + y.sum()).backward()

    q = randn(1, 5, 12, 128).transpose(1, 2).requires_grad_()
    k = randn(1, 5, 2, 128).transpose(1, 2).requires_grad_()
    cos, sin = randn(1, 5, 128), randn(1, 5, 128)
    q_out, k_out = RoPE.apply(q, k, cos, sin, torch.float16)  # as under fp16 autocast
    # dense gradients: .sum() would hand back stride-0 ones, another specialization
    torch.autograd.backward((q_out, k_out), (torch.ones_like(q_out), torch.ones_like(k_out)))

    rows, width, vocab = 8, 64, 4096  # the vocabulary only has to share 151936's divisibility
    head_in = randn(rows, width).requires_grad_()
    head = randn(vocab, width, dtype=torch.float16)
    target = torch.randint(0, vocab, (rows,), device=device, generator=gen)
    target[::2] = -100
    loss = fused_lm_head_ce(head_in, head, target, (target != -100).sum().float())[0]
    loss.backward()

    from .head8 import TILE_V, Int8Embedding, fused_int8_head_ce, quantize_rows

    q8, s8 = quantize_rows(randn(TILE_V + 4480, width, dtype=torch.float16))  # 151936's tiles
    Int8Embedding.apply(torch.zeros(2, 3, dtype=torch.long, device=device), q8, s8)
    head_in = randn(rows, width).requires_grad_()
    target = torch.randint(0, TILE_V + 4480, (rows,), device=device, generator=gen)
    target[::2] = -100
    fused_int8_head_ce(head_in, q8, s8, target, (target != -100).sum().float())[0].backward()

    state2 = types.SimpleNamespace(code=torch.linspace(-1, 1, 256, device=device),
                                   absmax=torch.ones(1, device=device), blocksize=256)
    state = types.SimpleNamespace(shape=(32, 64), blocksize=64, state2=state2,
                                  absmax=torch.zeros(32, device=device, dtype=torch.uint8),
                                  offset=torch.zeros((), device=device))
    dequantize_nf4(torch.zeros(1024, device=device, dtype=torch.uint8), state)

    gate = randn(2, 8960, dtype=torch.float16).requires_grad_()
    up = randn(2, 8960, dtype=torch.float16).requires_grad_()
    swiglu(gate, up).sum().backward()
    torch.cuda.synchronize()


def _run():
    # on the default stream: a stream of its own would get its own cuBLAS workspace (+44 MiB peak)
    from .diagtime import mark  # DIAG
    mark("warmup start")
    try:
        _compile_all()
        mark("warmup end")
    except Exception:  # noqa: BLE001 - a failed warm-up only means the kernels compile later
        pass


def start():
    """Start the background compilation once, when there is a GPU."""
    global _thread
    if _thread is None and torch.cuda.is_available():
        # A daemon, so it can never keep a finished process alive; a process that ends early
        # (the regression tests) still lets it finish, for at most TIMEOUT_S, at exit.
        _thread = threading.Thread(target=_run, name="kernel-warmup", daemon=True)
        _thread.start()
        atexit.register(_thread.join, TIMEOUT_S)


def wait():
    global _thread
    if _thread is not None:
        from .diagtime import mark  # DIAG
        mark("warmup wait begin")
        _thread.join(TIMEOUT_S)
        mark("warmup wait end")
        _thread = None
        # the warm-up's small blocks would otherwise stay cached next to the training's (~18 MiB)
        torch.cuda.empty_cache()
