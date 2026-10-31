"""The decoder's 4-bit weights live in pinned host memory and stream to the GPU layer by layer.

The packed NF4 weights of the 196 projections take ~650 MB of GPU memory for the whole run,
yet each one is needed only for its own matmul, once in the forward pass and once in the
backward pass. At the first training step they move to pinned host memory (the bytes are
unchanged; the absmax/scale tables, ~20 MB, stay on the GPU). Each NF4Linear (nf4.py) asks for
its packed weight; the next AHEAD projections in the direction the pass runs (forward: later
layers, backward: earlier layers) are already being copied on a second CUDA stream, so the
transfer overlaps compute and only a window of ~AHEAD weights sits on the GPU at a time.
The dequantization reads exactly the same bytes, so the numbers do not change.
"""

import contextlib
import threading

import torch

AHEAD = 1  # units (decoder layers) fetched ahead of the one in use


class _Unit:
    """What one host-to-GPU copy brings over: a decoder layer's packed weights when they sit
    next to each other in host memory (one region, one copy, one event), else one weight."""
    __slots__ = ("region", "members")

    def __init__(self, region, members):
        self.region = region  # pinned uint8 host tensor
        self.members = members  # [(position, byte offset, nbytes, shape)]


class WeightStreamer:
    def __init__(self):
        self.host = []  # pinned packed weights, in forward order
        self.index = {}  # host data_ptr -> position
        self.units = []  # copy units, in forward order
        self.unit_of = {}  # position -> (unit number, offset, nbytes, shape)
        self.ready = {}  # unit number -> [GPU buffer, event, waited, uses left]
        self.stream = None
        self.compute = None  # the compute stream, looked up once per step
        self.installed = False

    def install(self, model):
        """Move the packed weights of every 4-bit projection in the decoder layers to the host."""
        if self.installed:
            return
        self.installed = True
        from .nf4 import supported

        try:
            layers = model.model.layers
        except AttributeError:
            return
        for layer in layers:
            first = len(self.host)
            for module in layer.modules():
                weight = getattr(module, "weight", None)
                state = getattr(weight, "quant_state", None)
                if (type(module).__name__ != "Linear4bit" or weight is None or state is None
                        or hasattr(module, "base_layer")  # peft's LoRA wrapper: its base counts
                        or weight.dtype != torch.uint8 or not supported(state)
                        or not (weight.is_cuda or weight.is_pinned())
                        or (not weight.is_cuda and weight.data_ptr() in self.index)):
                    continue
                if weight.is_cuda:
                    host = torch.empty(weight.shape, dtype=weight.dtype, pin_memory=True)
                    host.copy_(weight.data)
                    weight.data = host
                host = weight.data  # already on the host when loading put it there
                HOSTED.add(host.data_ptr())
                self.index[host.data_ptr()] = len(self.host)
                self.host.append(host)
            self._add_units(range(first, len(self.host)))
        if self.host:
            self.stream = torch.cuda.Stream()
            torch.cuda.empty_cache()  # the freed weights leave the GPU, not just the allocator

    def _add_units(self, positions):
        """One unit for the positions when their weights form one region of the host block
        (loading takes them from it one after the other), else one unit per weight."""
        positions = list(positions)
        if not positions:
            return
        block = HOST_BLOCK.block if HOST_BLOCK is not None else None
        spans = [(self.host[p].data_ptr(), self.host[p].numel()) for p in positions]
        low = min(ptr for ptr, _ in spans)
        high = max(ptr + n for ptr, n in spans)
        total = sum(n for _, n in spans)
        if block is not None and len(positions) > 1:
            base = block.data_ptr()
            inside = all(base <= ptr and ptr + n <= base + block.numel() for ptr, n in spans)
            if inside and high - low <= total + 1024 * len(positions):
                region = block[low - base:high - base]
                members = [(p, ptr - low, n, tuple(self.host[p].shape))
                           for p, (ptr, n) in zip(positions, spans)]
                self._register(_Unit(region, members))
                return
        for p in positions:
            host = self.host[p]
            self._register(_Unit(host.view(-1).view(torch.uint8),
                                 [(p, 0, host.numel(), tuple(host.shape))]))

    def _register(self, unit):
        number = len(self.units)
        self.units.append(unit)
        for position, offset, nbytes, shape in unit.members:
            self.unit_of[position] = (number, offset, nbytes, shape)

    def _ensure_units(self):
        if len(self.unit_of) < len(self.host):  # weights added without install (tests)
            self._add_units(p for p in range(len(self.host)) if p not in self.unit_of)

    def new_step(self):
        """Before each training forward: the first layer's weights start coming over."""
        self.ready = {}
        self._ensure_units()
        if self.host:
            self.compute = torch.cuda.current_stream()
            for number in range(AHEAD + 1):
                self._fetch(number)

    def _fetch(self, number):
        if not 0 <= number < len(self.units) or number in self.ready:
            return
        compute = self.compute if self.compute is not None else torch.cuda.current_stream()
        unit = self.units[number]
        gpu = torch.empty(unit.region.shape, dtype=torch.uint8, device=compute.device)
        queued = torch.cuda.Event()
        queued.record(compute)  # the buffer may reuse memory the queued compute still reads
        done = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(queued)
            gpu.copy_(unit.region, non_blocking=True)
            done.record(self.stream)
        self.ready[number] = [gpu, done, False, len(unit.members)]

    def prefetch_backward(self):
        """Before the backward pass: its first weights are the last layer's."""
        for number in range(len(self.units) - 1, len(self.units) - 2 - AHEAD, -1):
            self._fetch(number)

    def get(self, packed):
        """The GPU copy of a packed weight (the tensor itself when it is not streamed)."""
        if packed.is_cuda:
            return packed
        position = self.index.get(packed.data_ptr())
        if position is None:
            return packed.cuda()
        self._ensure_units()
        number, offset, nbytes, shape = self.unit_of[position]
        step = 1 if torch._C._current_graph_task_id() == -1 else -1  # forward or backward
        self._fetch(number)
        for ahead in range(1, AHEAD + 1):
            self._fetch(number + step * ahead)
        entry = self.ready[number]
        if not entry[2]:
            compute = self.compute if self.compute is not None else torch.cuda.current_stream()
            compute.wait_event(entry[1])
            entry[2] = True
        entry[3] -= 1
        if entry[3] <= 0:  # every weight of the unit taken: its buffer goes with the last view
            del self.ready[number]
        return entry[0][offset:offset + nbytes].view(shape)


STREAMER = WeightStreamer()


class _HostBlock:
    """One pinned block for all packed weights, pinned in the background before the model loads
    (one pinning call instead of one per weight while loading); weights take slices of it."""

    def __init__(self):
        self.block, self.cursor, self.ready = None, 0, threading.Event()

    def reserve(self, nbytes):
        try:
            self.block = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        except Exception:  # noqa: BLE001 - without it each weight is pinned on its own
            self.block = None
        finally:
            self.ready.set()

    def take(self, nbytes):
        if not self.ready.is_set() and not self.ready.wait(30):
            return None
        start = -(-self.cursor // 512) * 512
        if self.block is None or start + nbytes > self.block.numel():
            return None
        self.cursor = start + nbytes
        return self.block[start:start + nbytes]


HOST_BLOCK = None


def host_bytes(nbytes):
    """Pinned host bytes from the block (or pinned on their own when it has no room)."""
    host = HOST_BLOCK.take(nbytes) if HOST_BLOCK is not None else None
    if host is None:
        try:
            host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        except Exception:  # noqa: BLE001 - no pinned memory: the caller keeps its GPU copy
            return None
    return host.view(torch.int8)


def reserve_weight_block(config):
    """Start-of-setup: the packed weights' total size from the checkpoint's config.json (all
    projections are NF4, two weights per byte), plus the int8 embedding / head table (one byte
    per weight, head8.py), reserved by the offload pinning thread."""
    global HOST_BLOCK
    layers = getattr(config, "num_hidden_layers", 0) or 0
    hidden = getattr(config, "hidden_size", 0) or 0
    inter = getattr(config, "intermediate_size", 0) or 0
    heads = getattr(config, "num_attention_heads", 0) or 0
    kv = getattr(config, "num_key_value_heads", heads) or heads
    if not (layers and hidden and inter and heads):
        return None
    head_dim = getattr(config, "head_dim", None) or hidden // heads
    per_layer = (2 * heads * head_dim * hidden + 2 * kv * head_dim * hidden + 3 * inter * hidden) // 2
    vocab = getattr(config, "vocab_size", 0) or 0
    HOST_BLOCK = _HostBlock()
    return lambda: HOST_BLOCK.reserve(layers * (per_layer + 7 * 512) + vocab * hidden + 512)


EMBEDDING_ON_HOST = [False]  # set when loading put the tied embedding in host memory
LOADED_TO_HOST = [0]  # 4-bit weights loading put in host memory
HOSTED = set()  # data pointers of the 4-bit weights that live in host memory


def keep_hosted_weights_on_host():
    """`Params4bit.to(cuda)` on a weight that lives in our host memory leaves it there.

    At the start of training accelerate's `prepare_model` calls `model.to(device)`; on the base
    run that is a no-op, but here it would copy all 196 packed weights (624 MiB) back to the GPU
    (the run's memory peak) only for the first training step to move them out again. The weights
    reach the GPU through wstream's window instead, so the move is skipped for them; any other
    tensor moves as usual."""
    try:
        import bitsandbytes

        cls = bitsandbytes.nn.Params4bit
    except Exception:  # noqa: BLE001
        return
    original = cls.to
    if getattr(original, "_kc_patched", False):
        return

    def to(self, *args, **kwargs):
        data = self.data
        if HOSTED and not data.is_cuda and data.data_ptr() in HOSTED:
            try:
                device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
            except Exception:  # noqa: BLE001 - not a device move we understand
                device, dtype = None, None
            if device is not None and device.type == "cuda" and dtype in (None, self.dtype):
                return self
        return original(self, *args, **kwargs)

    to._kc_patched = True
    cls.to = to


def _is_gpu(device):
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, int):
        return device >= 0
    return isinstance(device, str) and (device.startswith("cuda") or device.isdigit())


@contextlib.contextmanager
def _patched(module, name, replacement):
    original = getattr(module, name, None)
    if original is None:
        yield None
        return
    setattr(module, name, replacement(original))
    try:
        yield original
    finally:
        setattr(module, name, original)


@contextlib.contextmanager
def weights_to_host_while_loading():
    """While Soup loads the model, nothing large stays on the GPU:

    - each 4-bit weight goes to pinned host memory as soon as bitsandbytes has quantized it on
      the GPU, so the packed weights never sit on the GPU together;
    - the tied embedding (fp16, 467 MB) is loaded into host memory instead of the GPU; it becomes
      the int8 host table at the end of setup (head8.py), passing through the GPU in slices;
    - checkpoint tensors headed for the GPU are converted (bf16 -> fp16) on the GPU;
    - transformers' caching-allocator warm-up is skipped: it allocates (and the allocator then
      keeps) one block the size of the whole model up front, only to make the later per-weight
      allocations faster. Here those weights do not stay on the GPU, so the block would only be
      the run's memory peak (~1.07 GB). Loading itself is unchanged."""
    hosted = []
    keep_hosted_weights_on_host()
    with contextlib.ExitStack() as stack:
        try:
            import bitsandbytes

            def quantize(original):
                def _quantize(self, device):
                    result = original(self, device)
                    data = result.data
                    if data.is_cuda and data.dtype == torch.uint8:
                        host = HOST_BLOCK.take(data.numel()) if HOST_BLOCK is not None else None
                        host = host.view(data.shape) if host is not None else \
                            torch.empty(data.shape, dtype=data.dtype, pin_memory=True)
                        host.copy_(data)
                        result.data = host
                        hosted.append(host.data_ptr())
                        HOSTED.add(host.data_ptr())
                    return result
                return _quantize

            stack.enter_context(_patched(bitsandbytes.nn.Params4bit, "_quantize", quantize))
        except Exception:  # noqa: BLE001 - another bitsandbytes: weights move at the first step
            pass
        try:
            from transformers import core_model_loading, modeling_utils

            def get_device(original):
                def _get_device(device_map, param_name, *args, **kwargs):
                    device = original(device_map, param_name, *args, **kwargs)
                    if str(param_name).endswith("embed_tokens.weight") and _is_gpu(device):
                        EMBEDDING_ON_HOST[0] = True
                        return torch.device("cpu") if isinstance(device, torch.device) else "cpu"
                    return device
                return _get_device

            stack.enter_context(_patched(core_model_loading, "get_device", get_device))

            def materialize(original):
                def _materialize_copy(tensor, device=None, dtype=None):
                    # a bf16 checkpoint tensor headed for the GPU in fp16 goes over as it is and
                    # is converted there: the same rounding (round to nearest even) as on the
                    # CPU, where torch would otherwise convert 3 GB on a few cores first
                    tensor = tensor[...]
                    if (device is not None and dtype is not None and _is_gpu(device)
                            and tensor.device.type == "cpu" and tensor.dtype != dtype
                            and tensor.is_floating_point()):
                        return tensor.to(device=device).to(dtype=dtype)
                    if device is not None or dtype is not None:
                        tensor = tensor.to(device=device, dtype=dtype)
                    return tensor
                return _materialize_copy

            stack.enter_context(_patched(core_model_loading, "_materialize_copy", materialize))
            stack.enter_context(_patched(modeling_utils, "caching_allocator_warmup",
                                         lambda original: lambda *args, **kwargs: None))
        except Exception:  # noqa: BLE001 - another transformers: the usual loading
            pass
        try:
            yield hosted
        finally:
            LOADED_TO_HOST[0] += len(hosted)


def trainable_to_gpu(model):
    """peft puts each LoRA adapter on its base layer's device; with the base weight on the host
    that is the CPU, so the (small) trainable parameters are moved back to the GPU."""
    device = torch.device("cuda", torch.cuda.current_device())
    for param in model.parameters():
        if param.requires_grad and not param.is_cuda:
            param.data = param.data.to(device)
