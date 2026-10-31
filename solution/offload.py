"""The MLP's saved activations go to pinned host memory, overlapped with compute.

After padding removal the largest tensors kept for the backward pass are the MLP's gate and up
projections: two [tokens, 8960] fp16 tensors per layer, ~1 MB per token over 28 layers, about
half of the activation memory at the peak. SwiGLU's backward needs both.

Here, right after SwiGLU's forward, they are copied to pinned host memory on a second CUDA
stream while the GPU goes on with the next layers, and the GPU copies are released once the
transfer is done (at most MAX_IN_FLIGHT_BYTES wait for their transfer). In the backward pass each
layer's pair is copied back while the layer after it is still in its backward pass, so the
transfer again overlaps compute.

`patch_qwen2_layer_offload` does the same for everything else a decoder layer saves for its
backward pass (norm inputs, LoRA inputs and dropout masks, q/k/v and the attention output),
through autograd's saved-tensor hooks (the mechanism of torch.autograd.graph.save_on_cpu, with
our own arena, streams and per-layer prefetch). The copies are exact, so the numbers do not change. The task
allows recomputation or offloading in our own code; its time is part of the measured run.

SwiGLU itself is still the example's kernel (it does not count as ours); only what is kept for
its backward pass, and where, is ours.
"""

import atexit
import threading
import weakref

import torch
import triton

from .swiglu import BLOCK, example_swiglu_bwd_kernel, example_swiglu_fwd_kernel

MAX_IN_FLIGHT_BYTES = 128 << 20  # transfers the CPU lets run ahead before it waits for the oldest
RELEASE_WAIT_BYTES = 48 << 20  # before the head, the CPU waits only above this much in flight
SAVED_MIN_BYTES = 256 << 10  # smaller saved tensors stay on the GPU (not worth a transfer)
SAVED_PER_HIDDEN = 16  # arena estimate: bytes per token and hidden unit a layer saves
# (measured ~12.5: norms 2 x 2, LoRA inputs 2 x 2, masks 2 x 1, attention q + grouped k/v ~2.7)
ALIGN = 512


class _Entry:
    __slots__ = ("index", "group", "device", "specs", "gpu", "host", "copied", "back", "fetched",
                 "nbytes", "waited", "remaining")


class _Saved:
    """What autograd keeps for one offloaded saved tensor: its layer's entry and its slot."""
    __slots__ = ("entry", "slot")

    def __init__(self, entry, slot):
        self.entry, self.slot = entry, slot


def _dense(t):
    """Non-overlapping and dense: the elements fill exactly numel slots of the storage, so a
    host tensor with the same strides is a plain byte copy (a transposed view qualifies)."""
    if t.is_contiguous():
        return True
    expected = 1
    for stride, size in sorted((s, n) for s, n in zip(t.stride(), t.shape) if n != 1):
        if stride != expected:
            return False
        expected *= size
    return True


class Offloader:
    def __init__(self):
        self.stream = None
        self.arena = None  # pinned uint8 host buffer, reused every step
        self.cursor = 0
        self.entries = []  # this step's entries, in forward order
        self.groups = {}  # group (decoder layer, or entry index outside layers) -> its entries
        self.in_flight = []  # entries whose GPU tensors are kept until their transfer is done
        self.in_flight_bytes = 0
        self.armed = False
        self.layer = None  # the decoder layer whose forward is running (layer_offload)
        self.compute = None  # the compute stream, looked up once per step
        self.seen = {}  # this step's packed tensors, so a tensor saved twice is copied once
        self.pending = []  # the running layer's saved tensors, copied together when it ends
        self.pending_entry = None
        self._reserving = None
        self.peak_cursor = 0
        self.kept_on_gpu = 0

    def reserve(self, nbytes):
        """Pinned host memory for one step."""
        if self.arena is None or self.arena.numel() < nbytes:
            self.arena = None
            self.arena = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)

    def reserve_in_background(self, nbytes, first=None):
        """Pin the host memory in the background (pinning a few GB takes seconds); `first`
        runs before, in the same thread."""
        if self._reserving is None and torch.cuda.is_available():
            def work():
                if first is not None:
                    first()
                self._reserve_quietly(nbytes)

            self._reserving = threading.Thread(target=work, name="offload-pinning", daemon=True)
            self._reserving.start()
            # a process that ends before training (the regression tests) lets it finish first
            atexit.register(self._reserving.join, 120)

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
        self.peak_cursor = max(self.peak_cursor, self.cursor)
        self.entries, self.groups, self.in_flight, self.cursor = [], {}, [], 0
        self.in_flight_bytes, self.layer, self.seen = 0, None, {}
        self.pending, self.pending_entry = [], None
        if nbytes:
            self._reserve_quietly(nbytes)  # if pinning fails, the step runs without offloading
        self.armed = self.arena is not None
        self.compute = torch.cuda.current_stream() if torch.cuda.is_available() else None

    def _compute(self, device):
        compute = self.compute
        if compute is None or compute.device != device:
            compute = torch.cuda.current_stream(device)
        return compute

    def active(self, gate):
        return (self.armed and gate.is_cuda and torch.is_grad_enabled()
                and gate.requires_grad and gate.dtype in (torch.float16, torch.bfloat16))

    def _host(self, nbytes):
        start = -(-self.cursor // ALIGN) * ALIGN
        if self.arena is None or start + nbytes > self.arena.numel():
            return None
        self.cursor = start + nbytes
        return self.arena[start:start + nbytes]

    def _new_entry(self, tensors, group):
        entry = _Entry()
        entry.index, entry.device, entry.group = len(self.entries), tensors[0].device, group
        entry.gpu, entry.host, entry.copied, entry.back, entry.fetched = tensors, None, None, \
            None, None
        entry.waited, entry.remaining = False, 0
        self.entries.append(entry)
        if group is not None:
            self.groups.setdefault(group, []).append(entry)
        return entry

    def _start_copy(self, entry):
        """One event and one pass on the copy stream for all of the entry's tensors."""
        tensors = entry.gpu
        entry.specs = tuple((t.shape, t.stride(), t.dtype) for t in tensors)
        sizes = [t.numel() * t.element_size() for t in tensors]
        entry.nbytes = sum(sizes)
        host = self._host(sum(-(-n // ALIGN) * ALIGN for n in sizes))
        if host is None:  # no room left: these simply stay on the GPU
            self.kept_on_gpu += 1
            return
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=entry.device)
        ready = torch.cuda.Event()
        ready.record(self._compute(entry.device))
        views, offset = [], 0
        for t, n in zip(tensors, sizes):
            views.append(host[offset:offset + n].view(t.dtype).as_strided(t.shape, t.stride()))
            offset += -(-n // ALIGN) * ALIGN
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)  # the tensors are computed
            for view, t in zip(views, tensors):
                view.copy_(t, non_blocking=True)
            entry.copied = torch.cuda.Event()
            entry.copied.record(self.stream)
        entry.host = tuple(views)
        self.in_flight.append(entry)
        self.in_flight_bytes += entry.nbytes
        self._retire()

    def save(self, *tensors):
        """Start copying dense GPU tensors to the arena; returns their entry for `load`."""
        entry = self._new_entry(tensors, self.layer)
        if entry.group is None:
            entry.group = entry.index
            self.groups.setdefault(entry.group, []).append(entry)
        self._start_copy(entry)
        return entry

    def flush(self):
        """The running layer has ended: copy its saved tensors to the host in one go."""
        entry, self.pending_entry = self.pending_entry, None
        if entry is not None:
            entry.gpu, self.pending = tuple(self.pending), []
            entry.remaining = len(entry.gpu)
            self._start_copy(entry)

    def _retire(self):
        """Drop the GPU tensors whose transfer is done. The caching allocator reuses their memory
        in the compute stream's order, which is safe once the copy stream has read them."""
        waiting = []
        for entry in self.in_flight:
            if entry.copied.query():
                entry.gpu = None
                self.in_flight_bytes -= entry.nbytes
            else:
                waiting.append(entry)
        while waiting and self.in_flight_bytes > MAX_IN_FLIGHT_BYTES:
            entry = waiting.pop(0)
            entry.copied.synchronize()
            entry.gpu = None
            self.in_flight_bytes -= entry.nbytes
        self.in_flight = waiting

    def release_all(self):
        """Before the head: the last layer's tensors start their copy, and the GPU tensors of
        finished copies are dropped. The CPU waits only while more than RELEASE_WAIT_BYTES are
        still in flight (the head's buffers would sit next to them); otherwise the head is queued
        at once and the GPU never idles here. What remains is dropped when the backward starts."""
        self.flush()
        self._retire()
        while self.in_flight and self.in_flight_bytes > RELEASE_WAIT_BYTES:
            entry = self.in_flight.pop(0)
            entry.copied.synchronize()
            entry.gpu = None
            self.in_flight_bytes -= entry.nbytes

    def _retire_all_done(self):
        """Drop the GPU tensors of every finished copy (no waiting); the rest wait on events."""
        if self.in_flight:
            waiting = []
            for entry in self.in_flight:
                if entry.copied.query():
                    entry.gpu = None
                    self.in_flight_bytes -= entry.nbytes
                else:
                    waiting.append(entry)
            self.in_flight = waiting

    def _fetch_group(self, group):
        for entry in self.groups.get(group, ()):
            if entry.gpu is None and entry.back is None and entry.host is not None:
                self._fetch(entry)

    def prefetch_last(self):
        """Start copying the last group back: it is the first one the backward pass needs.
        (The forward's last copies are done by now: their GPU tensors go.)"""
        self._retire_all_done()
        if self.entries:
            self._fetch_group(self.entries[-1].group)

    def _fetch(self, entry):
        """Copy an entry back to the GPU on the copy stream. The buffers are allocated in the
        compute stream's order, and the copy starts after the compute work queued so far."""
        compute = self._compute(entry.device)
        back = tuple(torch.empty_strided(shape, stride, dtype=dtype, device=entry.device)
                     for shape, stride, dtype in entry.specs)
        ready = torch.cuda.Event()
        ready.record(compute)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            for gpu, host in zip(back, entry.host):
                gpu.copy_(host, non_blocking=True)
            entry.fetched = torch.cuda.Event()
            entry.fetched.record(self.stream)
        entry.back, entry.waited = list(back), False

    def load(self, entry):
        """The entry's tensors for the backward pass; the rest of its group is fetched with it,
        and the previous group starts coming back while this one is used."""
        if entry.gpu is not None:
            tensors = entry.gpu
        else:
            if entry.back is None:
                if entry.group is not None:
                    self._fetch_group(entry.group)
                if entry.back is None:  # not in a group
                    self._fetch(entry)
            self._compute(entry.device).wait_event(entry.fetched)
            tensors = tuple(entry.back)
        entry.gpu = entry.back = None
        if entry.host is None:  # it stayed on the GPU: a second load needs it again
            entry.gpu = tensors
        if entry.group is not None:
            self._fetch_group(entry.group - 1)
        return tensors

    # saved_tensors_hooks for everything else a decoder layer keeps for its backward pass

    def pack(self, tensor):
        if not self.armed or tensor.numel() * tensor.element_size() < SAVED_MIN_BYTES:
            return tensor  # most saved tensors (small ones): nothing else to look at
        seen = self.seen.get(id(tensor))
        if seen is not None and seen[0]() is tensor:  # the same tensor saved again
            if seen[1].entry.group == self.layer:
                return seen[1]
            return tensor  # by another layer (RoPE's cos/sin, < 1 MB): it stays on the GPU
        if (not self.armed or not tensor.is_cuda or isinstance(tensor, torch.nn.Parameter)
                or tensor.dtype == torch.uint8
                or tensor.numel() * tensor.element_size() < SAVED_MIN_BYTES
                or not _dense(tensor)):
            return tensor
        if self.pending_entry is None:
            self.pending_entry = self._new_entry((tensor,), self.layer)
        saved = _Saved(self.pending_entry, len(self.pending))
        self.pending.append(tensor)
        # a weak reference: the check above matches only this very tensor object, never a new
        # tensor that happens to get the same id or address after this one is freed
        self.seen[id(tensor)] = (weakref.ref(tensor), saved)
        return saved

    def unpack(self, packed):
        if isinstance(packed, torch.Tensor):
            return packed
        entry, slot = packed.entry, packed.slot
        if entry.gpu is not None:  # still on the GPU (copy in flight, or no room on the host)
            return entry.gpu[slot]
        if entry.back is None or entry.back[slot] is None:
            if entry.back is not None:  # unpacked twice: bring this one tensor back again
                shape, stride, dtype = entry.specs[slot]
                again = torch.empty_strided(shape, stride, dtype=dtype, device=entry.device)
                return again.copy_(entry.host[slot])
            if entry.group is not None:
                self._fetch_group(entry.group)
            if entry.back is None:
                self._fetch(entry)
        if not entry.waited:
            self._compute(entry.device).wait_event(entry.fetched)
            entry.waited = True
            if entry.group is not None:
                self._fetch_group(entry.group - 1)  # the previous layer starts coming back
        tensor, entry.back[slot] = entry.back[slot], None
        return tensor


OFFLOADER = Offloader()


def arena_bytes(config, max_tokens, mlp_pairs=False):
    """Host memory for one step, for max_tokens tokens: what a decoder layer saves for its
    backward pass (fp16 norm inputs, fp16 LoRA inputs and their dropout masks, q and the grouped
    k/v, the MLP's fp16 input: about 14.5 x hidden bytes per token, 18 with headroom), plus gate
    and up (fp16) of every layer when the MLP keeps them (mlp_pairs: the module path)."""
    layers = getattr(config, "num_hidden_layers", 0)
    inter = getattr(config, "intermediate_size", 0)
    hidden = getattr(config, "hidden_size", 0) or 0
    if not layers or not inter or not max_tokens:
        return 0
    per_token = (SAVED_PER_HIDDEN + 2) * hidden + (2 * inter * 2 if mlp_pairs else 0)
    return layers * (max_tokens * per_token + 32 * ALIGN)


def _report():
    if OFFLOADER.peak_cursor or OFFLOADER.cursor:
        import sys

        arena = OFFLOADER.arena.numel() if OFFLOADER.arena is not None else 0
        print(f"[solution] offload: peak host use {max(OFFLOADER.peak_cursor, OFFLOADER.cursor) >> 20}"
              f" MiB of {arena >> 20} MiB, entries kept on GPU {OFFLOADER.kept_on_gpu}",
              file=sys.stderr)


atexit.register(_report)


def patch_qwen2_layer_offload():
    """Everything a decoder layer saves for its backward pass (above SAVED_MIN_BYTES) goes to
    the host arena too, through autograd's saved-tensor hooks, grouped by layer so the backward
    pass fetches layer i - 1 while layer i runs. Exact copies: the numbers do not change."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2DecoderLayer
    if getattr(cls.forward, "_kc_offload", False):
        return
    original = cls.forward

    def forward(self, *args, **kwargs):
        if not (OFFLOADER.armed and self.training and torch.is_grad_enabled()
                and torch._C._current_graph_task_id() == -1):
            return original(self, *args, **kwargs)
        previous = OFFLOADER.layer
        OFFLOADER.layer = getattr(self.self_attn, "layer_idx", None)
        try:
            with torch.autograd.graph.saved_tensors_hooks(OFFLOADER.pack, OFFLOADER.unpack):
                return original(self, *args, **kwargs)
        finally:
            OFFLOADER.flush()
            OFFLOADER.layer = previous

    forward._kc_offload = True
    forward._kc_patched = True
    cls.forward = forward


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
