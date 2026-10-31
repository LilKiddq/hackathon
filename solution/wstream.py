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
import os

import torch

AHEAD = 8  # projections fetched ahead of the one in use (~one decoder layer, ~25 MB)


class WeightStreamer:
    def __init__(self):
        self.host = []  # pinned packed weights, in forward order
        self.index = {}  # host data_ptr -> position
        self.ready = {}  # position -> (GPU copy, event)
        self.stream = None
        self.installed = False

    def install(self, model, resident_layers=0):
        """Move the packed weights of every 4-bit projection in the decoder layers to the host."""
        if self.installed:
            return
        self.installed = True
        from .nf4 import supported

        try:
            layers = model.model.layers
        except AttributeError:
            return
        if os.environ.get("KC_NO_WSTREAM"):
            resident_layers = len(layers)
        for number, layer in enumerate(layers):
            resident = number >= len(layers) - resident_layers
            for module in layer.modules():
                weight = getattr(module, "weight", None)
                state = getattr(weight, "quant_state", None)
                if (type(module).__name__ != "Linear4bit" or weight is None or state is None
                        or weight.dtype != torch.uint8 or not supported(state)
                        or not (weight.is_cuda or weight.is_pinned())):
                    continue
                if resident:  # the last layers keep their weights on the GPU
                    if not weight.is_cuda:
                        weight.data = weight.data.to(torch.cuda.current_device())
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


@contextlib.contextmanager
def weights_to_host_while_loading():
    """While Soup loads the model: each 4-bit weight goes to pinned host memory as soon as
    bitsandbytes has quantized it on the GPU, so the packed weights never all sit on the GPU
    together (that, next to the embedding, was the run's memory peak during setup)."""
    try:
        import bitsandbytes
        cls = bitsandbytes.nn.Params4bit
        original = cls._quantize
    except Exception:  # noqa: BLE001 - another bitsandbytes: weights move at the first step
        yield []
        return
    hosted = []

    def _quantize(self, device):
        result = original(self, device)
        data = result.data
        if data.is_cuda and data.dtype == torch.uint8:
            host = torch.empty(data.shape, dtype=data.dtype, pin_memory=True)
            host.copy_(data)
            result.data = host
            hosted.append(host.data_ptr())
        return result

    cls._quantize = _quantize
    try:
        yield hosted
    finally:
        cls._quantize = original


def trainable_to_gpu(model):
    """peft puts each LoRA adapter on its base layer's device; with the base weight on the host
    that is the CPU, so the (small) trainable parameters are moved back to the GPU."""
    device = torch.device("cuda", torch.cuda.current_device())
    for param in model.parameters():
        if param.requires_grad and not param.is_cuda:
            param.data = param.data.to(device)
