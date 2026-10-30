"""The MLP's saved activations go to pinned host memory, overlapped with compute.

After padding removal the largest tensors kept for the backward pass are the MLP's gate and up
projections: two [tokens, 8960] fp16 tensors per layer, ~1 MB per token over 28 layers, about
half of the activation memory at the peak. SwiGLU's backward needs both.

Here, right after SwiGLU's forward, they are copied to pinned host memory on a second CUDA
stream while the GPU goes on with the next layers, and the GPU copies are released once the
transfer is done (at most MAX_IN_FLIGHT layers wait for their transfer). In the backward pass each
layer's pair is copied back while the layer after it is still in its backward pass, so the
transfer again overlaps compute. The copies are exact, so the numbers do not change. The task
allows recomputation or offloading in our own code; its time is part of the measured run.

SwiGLU itself is still the example's kernel (it does not count as ours); only what is kept for
its backward pass, and where, is ours.
"""

import threading

import torch
import triton

from .swiglu import BLOCK, example_swiglu_bwd_kernel, example_swiglu_fwd_kernel

MAX_IN_FLIGHT = 3  # forward transfers the CPU lets run ahead before it waits for the oldest
ALIGN = 512


class _Entry:
    __slots__ = ("index", "device", "shape", "dtype", "gpu", "host", "copied", "back", "fetched")


class Offloader:
    def __init__(self):
        self.stream = None
        self.arena = None  # pinned uint8 host buffer, reused every step
        self.cursor = 0
        self.entries = []  # this step's entries, in forward order
        self.in_flight = []  # entries whose GPU tensors are kept until their transfer is done
        self.armed = False
        self._reserving = None

    def reserve(self, nbytes):
        """Pinned host memory for one step."""
        if self.arena is None or self.arena.numel() < nbytes:
            self.arena = None
            self.arena = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)

    def reserve_in_background(self, nbytes):
        """Pin the host memory while Soup builds the trainer (pinning ~2 GB takes a while)."""
        if self._reserving is None and torch.cuda.is_available():
            self._reserving = threading.Thread(target=self._reserve_quietly, args=(nbytes,),
                                               name="offload-pinning", daemon=True)
            self._reserving.start()

    def _reserve_quietly(self, nbytes):
        try:
            self.reserve(nbytes)
        except Exception:  # noqa: BLE001 - without the arena the step reserves it, or keeps all on GPU
            self.arena = None

    def new_step(self, nbytes=None):
        """Called before each training forward; the previous step's backward has finished."""
        if self._reserving is not None:
            self._reserving.join()
            self._reserving = None
        if self.stream is not None:
            self.stream.synchronize()
        self.entries, self.in_flight, self.cursor = [], [], 0
        if nbytes:
            self._reserve_quietly(nbytes)  # if pinning fails, the step runs without offloading
        self.armed = self.arena is not None

    def active(self, gate):
        return (self.armed and gate.is_cuda and torch.is_grad_enabled()
                and gate.requires_grad and gate.dtype in (torch.float16, torch.bfloat16))

    def _host(self, nbytes):
        start = -(-self.cursor // ALIGN) * ALIGN
        if self.arena is None or start + nbytes > self.arena.numel():
            return None
        self.cursor = start + nbytes
        return self.arena[start:start + nbytes]

    def save(self, gate, up):
        entry = _Entry()
        entry.index, entry.device = len(self.entries), gate.device
        entry.shape, entry.dtype = gate.shape, gate.dtype
        entry.gpu, entry.host, entry.copied, entry.back, entry.fetched = (gate, up), None, None, \
            None, None
        self.entries.append(entry)
        nbytes = gate.numel() * gate.element_size()
        host = self._host(2 * nbytes)
        if host is None:  # no room left: this pair simply stays on the GPU
            return entry
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=gate.device)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(gate.device))
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)  # gate and up are computed
            entry.host = (host[:nbytes].view(gate.dtype).view(gate.shape),
                          host[nbytes:].view(up.dtype).view(up.shape))
            entry.host[0].copy_(gate, non_blocking=True)
            entry.host[1].copy_(up, non_blocking=True)
            entry.copied = torch.cuda.Event()
            entry.copied.record(self.stream)
        self.in_flight.append(entry)
        self._retire()
        return entry

    def _retire(self):
        """Drop the GPU tensors whose transfer is done. The caching allocator reuses their memory
        in the compute stream's order, which is safe once the copy stream has read them."""
        waiting = []
        for entry in self.in_flight:
            if entry.copied.query():
                entry.gpu = None
            else:
                waiting.append(entry)
        while len(waiting) > MAX_IN_FLIGHT:
            entry = waiting.pop(0)
            entry.copied.synchronize()
            entry.gpu = None
        self.in_flight = waiting

    def _fetch(self, entry):
        """Copy a pair back to the GPU on the copy stream. The buffers are allocated in the
        compute stream's order, and the copy starts after the compute work queued so far."""
        compute = torch.cuda.current_stream(entry.device)
        gate = torch.empty(entry.shape, dtype=entry.dtype, device=entry.device)
        up = torch.empty(entry.shape, dtype=entry.dtype, device=entry.device)
        ready = torch.cuda.Event()
        ready.record(compute)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            gate.copy_(entry.host[0], non_blocking=True)
            up.copy_(entry.host[1], non_blocking=True)
            entry.fetched = torch.cuda.Event()
            entry.fetched.record(self.stream)
        entry.back = (gate, up)

    def load(self, entry):
        """gate and up for the backward pass; starts fetching the previous layer's pair."""
        if entry.gpu is not None:
            pair = entry.gpu
        else:
            if entry.back is None:
                self._fetch(entry)
            torch.cuda.current_stream(entry.device).wait_event(entry.fetched)
            pair = entry.back
        entry.gpu = entry.back = None
        if entry.index > 0:
            previous = self.entries[entry.index - 1]
            if previous.gpu is None and previous.back is None and previous.host is not None:
                self._fetch(previous)
        return pair


OFFLOADER = Offloader()


def arena_bytes(config, max_tokens):
    """Host memory for one step: gate and up (fp16) of every layer for max_tokens tokens."""
    layers = getattr(config, "num_hidden_layers", 0)
    inter = getattr(config, "intermediate_size", 0)
    if not layers or not inter or not max_tokens:
        return 0
    return layers * (2 * max_tokens * inter * 2 + 2 * ALIGN)


class OffloadedSwiGLU(torch.autograd.Function):
    """silu(gate) * up with the example's kernels; gate and up wait for the backward pass in
    pinned host memory instead of GPU memory."""

    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.to(gate.dtype).contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n, BLOCK=BLOCK)
        ctx.entry = OFFLOADER.save(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = OFFLOADER.load(ctx.entry)
        ctx.entry = None
        grad = grad.to(gate.dtype).contiguous()
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](grad, gate, up, dgate, dup, n,
                                                           BLOCK=BLOCK)
        return dgate, dup


def swiglu_maybe_offloaded(gate, up, fallback):
    if OFFLOADER.active(gate):
        return OffloadedSwiGLU.apply(gate, up)
    return fallback(gate, up)
