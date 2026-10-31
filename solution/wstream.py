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

AHEAD = 8  # projections fetched ahead of the one in use (~one decoder layer, ~25 MB)


class WeightStreamer:
    def __init__(self):
        self.host = []  # pinned packed weights, in forward order
        self.index = {}  # host data_ptr -> position
        self.ready = {}  # position -> (GPU copy, event)
        self.stream = None
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
            for module in layer.modules():
                weight = getattr(module, "weight", None)
                state = getattr(weight, "quant_state", None)
                if (type(module).__name__ != "Linear4bit" or weight is None or state is None
                        or weight.dtype != torch.uint8 or not supported(state)
                        or not (weight.is_cuda or weight.is_pinned())):
                    continue
                if weight.is_cuda:
                    host = torch.empty(weight.shape, dtype=weight.dtype, pin_memory=True)
                    host.copy_(weight.data)
                    weight.data = host
                host = weight.data  # already on the host when loading put it there
                self.index[host.data_ptr()] = len(self.host)
                self.host.append(host)
        if self.host:
            self.stream = torch.cuda.Stream()
            torch.cuda.empty_cache()  # the freed weights leave the GPU, not just the allocator

    def new_step(self):
        """Before each training forward: the first layer's projections start coming over."""
        self.ready = {}
        for position in range(AHEAD):
            self._fetch(position)

    def _fetch(self, position):
        if not 0 <= position < len(self.host) or position in self.ready:
            return
        compute = torch.cuda.current_stream()
        host = self.host[position]
        gpu = torch.empty(host.shape, dtype=host.dtype, device=compute.device)
        queued = torch.cuda.Event()
        queued.record(compute)  # the buffer may reuse memory the queued compute still reads
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(queued)
            gpu.copy_(host, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        self.ready[position] = (gpu, done)

    def prefetch_backward(self):
        """Before the backward pass: its first projections are the last layer's."""
        for position in range(len(self.host) - 1, len(self.host) - 1 - AHEAD, -1):
            self._fetch(position)

    def get(self, packed):
        """The GPU copy of a packed weight (the tensor itself when it is not streamed)."""
        if packed.is_cuda:
            return packed
        position = self.index.get(packed.data_ptr())
        if position is None:
            return packed.cuda()
        step = 1 if torch._C._current_graph_task_id() == -1 else -1  # forward or backward
        self._fetch(position)
        for ahead in range(1, AHEAD + 1):
            self._fetch(position + step * ahead)
        gpu, done = self.ready.pop(position)
        torch.cuda.current_stream().wait_event(done)
        return gpu


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
    - transformers' caching-allocator warm-up is skipped: it allocates (and the allocator then
      keeps) one block the size of the whole model up front, only to make the later per-weight
      allocations faster. Here those weights do not stay on the GPU, so the block would only be
      the run's memory peak (~1.07 GB). Loading itself is unchanged."""
    hosted = []
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
