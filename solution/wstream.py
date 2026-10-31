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

import torch

import os
AHEAD = int(os.environ.get('KC_AHEAD', '8'))  # projections fetched ahead of the one in use (~one decoder layer, ~25 MB)


class WeightStreamer:
    def __init__(self):
        self.host = []  # pinned packed weights, in forward order
        self.index = {}  # host data_ptr -> position
        self.ready = {}  # position -> (GPU copy, event)
        self.stream = None
        self.installed = False

    def install(self, model):
        """Move the packed weights of every 4-bit projection in the decoder layers to the host."""
        if self.installed or os.environ.get("KC_NO_WSTREAM"):
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
                        or not weight.is_cuda or weight.dtype != torch.uint8
                        or not supported(state)):
                    continue
                host = torch.empty(weight.shape, dtype=weight.dtype, pin_memory=True)
                host.copy_(weight.data)
                weight.data = host
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
