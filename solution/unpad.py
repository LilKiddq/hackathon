"""Padding removed inside each batch: the decoder layers run on the real tokens only.

TRL pads every batch to its longest row. With this task's rows (median ~140 tokens, a few cut
at 512) about 43 % of the [batch, seq] positions are padding, and every projection, MLP, norm
and LoRA matmul spends the same work on them as on real tokens. The rules allow removing
padding inside a batch (rows of different batches are never mixed).

How (training only, when the batch has padding):
- `Qwen2Model.forward` notes which [batch * seq] positions are real (`attention_mask == 1`);
  the masks and the rotary cos/sin are still built for the padded batch, as before.
- The first decoder layer packs its input to [1, n_real, hidden]; every layer after it works on
  that. Norms, MLP and residual adds are per token, so they are unchanged for real tokens.
- Attention projects q/k/v on the packed tokens, puts them back at their padded positions
  (zeros elsewhere), runs RoPE and attention exactly as before with the same mask, and packs
  the output again for o_proj. Padding keys are masked and padding queries are dropped, as
  before, so real tokens get the same attention.
- LoRA's dropout (p = 0.05, unchanged) draws its mask for the packed real tokens only. The
  masks are other draws of the same distribution than in the padded run (the random stream
  advances differently); the first step is the same either way, as LoRA's B starts at zero.
- The model's output is put back in the padded shape, so the head, loss and metrics see the
  same tensor as before (padding rows are zeros there and are skipped by the fused loss).
"""

import torch

# while a padded batch runs unpadded: (real positions [n] int64, batch, seq, row lengths or None);
# the lengths are set when every row is a prefix of real tokens (TRL pads on the right)
_ctx = None
# the latest model forward's context, or None when it ran without unpadding: gradient
# checkpointing re-runs layers inside that forward's backward pass and needs it again
_last_ctx = None
_rope_cache = {}  # per forward: the padded cos/sin -> cos/sin at each packed token's position


def _pack(x, index):
    return x.reshape(-1, x.shape[-1]).index_select(0, index).unsqueeze(0)


def _unpack(x, index, batch, seq):
    flat = x.reshape(-1, x.shape[-1])
    full = flat.new_zeros((batch * seq, flat.shape[-1]))
    return full.index_copy(0, index, flat).view(batch, seq, flat.shape[-1])


def _is_packed(x):
    return _ctx is not None and x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == _ctx[0].numel()


def _row_attention(module, hidden_states, position_embeddings, index, seq, rows):
    """Attention on packed tokens, one causal SDPA call per row at its real length.

    The rows are contiguous runs of the packed tokens, so each row's q/k/v is a slice: no padded
    copies and no mask. Each real token attends to the same keys as with the padded mask (the
    padding keys were masked out, the padding queries dropped). Keys and values are repeated for
    the grouped heads and cast to fp16 once, as transformers' SDPA path does per call."""
    from transformers.models.qwen2 import modeling_qwen2

    n = hidden_states.shape[1]
    shape = (1, n, -1, module.head_dim)
    parts = _projection_parts(module, hidden_states)
    if parts is not None:  # q/k/v as one autograd node, LoRA added as peft adds it
        bases, wrappers = parts
        query, key, value = FusedQKV.apply(hidden_states, *bases)
        query = query.float().add_(_lora_delta(wrappers[0], hidden_states)) \
            if wrappers[0] is not None else query
        if wrappers[2] is not None:
            value = value + _lora_delta(wrappers[2], hidden_states)
        if wrappers[1] is not None:
            key = key + _lora_delta(wrappers[1], hidden_states)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
    else:
        query = module.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = module.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = module.v_proj(hidden_states).view(shape).transpose(1, 2)

    cos, sin = position_embeddings
    packed = _rope_cache.get(id(cos))
    if packed is None:
        if cos.shape[0] == 1:  # one table for the batch: flat position i is position i % seq
            packed = tuple(t[0].index_select(0, index % seq)[None] for t in (cos, sin))
        else:
            packed = tuple(t.reshape(-1, t.shape[-1]).index_select(0, index)[None]
                           for t in (cos, sin))
        _rope_cache[id(cos)] = packed
    query, key = modeling_qwen2.apply_rotary_pos_emb(query, key, *packed)

    if torch.is_autocast_enabled("cuda"):  # the casts SDPA would make under autocast
        dtype = torch.get_autocast_dtype("cuda")
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
    groups = getattr(module, "num_key_value_groups", 1)
    attn_output = RowsAttention.apply(query, key, value, tuple(rows), groups, module.scaling)
    o_proj = module.o_proj
    if parts is not None and _plain_4bit(o_proj) and o_proj.bias is None \
            and attn_output.dtype == torch.float16:
        from .nf4 import NF4Linear  # bitsandbytes' wrapper and our matmul_4bit patch skipped

        return NF4Linear.apply(attn_output, o_proj.weight, o_proj.weight.quant_state, None), None
    return o_proj(attn_output), None


def _plain_4bit(module):
    from .nf4 import supported

    weight = getattr(module, "weight", None)
    state = getattr(weight, "quant_state", None)
    return (type(module).__name__ == "Linear4bit" and not hasattr(module, "base_layer")
            and getattr(module, "compute_dtype", None) == torch.float16
            and state is not None and weight.dtype == torch.uint8 and supported(state))


def _lora_wrapper(module):
    """peft's LoRA layer around a 4-bit projection, in the plain case (vanilla LoRA, not merged,
    adapters on): the one this training uses. None for anything else."""
    if not hasattr(module, "base_layer"):
        return None
    if (getattr(module, "disable_adapters", True) or getattr(module, "merged", True)
            or getattr(module, "lora_variant", None)):
        return None
    active = [name for name in module.active_adapters if name in module.lora_A.keys()]
    return module if active else None


def _projection_parts(module, hidden_states):
    """The 4-bit base layers of q/k/v (and their LoRA wrappers) when the fused path applies:
    training under fp16 autocast, frozen 4-bit bases, LoRA as plain peft LoRA. Else None."""
    if not (torch.is_grad_enabled() and hidden_states.requires_grad and hidden_states.is_cuda
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16):
        return None
    bases, wrappers = [], []
    for proj in (module.q_proj, module.k_proj, module.v_proj):
        wrapper = _lora_wrapper(proj)
        if wrapper is None and hasattr(proj, "base_layer"):
            return None  # some other adapter type: the module path
        base = proj.base_layer if wrapper is not None else proj
        if not _plain_4bit(base) or (base.bias is not None and base.bias.requires_grad):
            return None
        bases.append(base)
        wrappers.append(wrapper)
    return bases, wrappers


def _lora_delta(wrapper, x):
    """peft's LoRA term for the active adapters, as peft computes it under autocast:
    lora_B(lora_A(dropout(x))) * scaling (the same modules, the same random draws)."""
    total = None
    for name in wrapper.active_adapters:
        if name not in wrapper.lora_A.keys():
            continue
        out = wrapper.lora_B[name](wrapper.lora_A[name](wrapper.lora_dropout[name](x))) \
            * wrapper.scaling[name]
        total = out if total is None else total + out
    return total


def _fp16_bias(layer):
    bias = layer.bias
    if bias is not None and bias.dtype != torch.float16:
        bias.data = bias.data.to(torch.float16)  # what bitsandbytes' Linear4bit.forward does
    return bias


class FusedQKV(torch.autograd.Function):
    """The 4-bit q/k/v base projections as one autograd node (fp16 outputs, frozen fp16 biases):
    x cast to fp16 once, each weight dequantized (streamed, wstream.py) and multiplied as
    before. The backward pass returns x's gradient as the sum of the three input gradients in
    x's dtype, as autograd summed them before."""

    @staticmethod
    def forward(ctx, x, q_layer, k_layer, v_layer):
        from .mlp import _weight

        flat = x.reshape(-1, x.shape[-1]).to(torch.float16)
        outs = tuple(torch.nn.functional.linear(flat, _weight(layer), _fp16_bias(layer))
                     for layer in (q_layer, k_layer, v_layer))
        ctx.layers, ctx.x_dtype, ctx.x_shape = (q_layer, k_layer, v_layer), x.dtype, x.shape
        return tuple(out.view(*x.shape[:-1], out.shape[-1]) for out in outs)

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        from .mlp import _weight

        dx = None
        # backward order of the streamed weights: v, k, q
        for grad, layer in ((grad_v, ctx.layers[2]), (grad_k, ctx.layers[1]),
                            (grad_q, ctx.layers[0])):
            if grad is None:
                continue
            part = (grad.reshape(-1, grad.shape[-1]).to(torch.float16) @ _weight(layer)) \
                .to(ctx.x_dtype)
            dx = part if dx is None else dx.add_(part)
        return (dx.view(ctx.x_shape) if dx is not None else None), None, None, None


def _attend(query, key, value, rows, groups, scale):
    """Causal SDPA per row on packed q [1, H, n, D] and k/v [1, KV, n, D] -> [1, n, H * D]."""
    from transformers.models.qwen2 import modeling_qwen2

    key = modeling_qwen2.repeat_kv(key, groups)
    value = modeling_qwen2.repeat_kv(value, groups)
    outputs, start = [], 0
    for length in rows:
        if length == 0:
            continue
        stop = start + length
        outputs.append(torch.nn.functional.scaled_dot_product_attention(
            query[:, :, start:stop], key[:, :, start:stop], value[:, :, start:stop],
            dropout_p=0.0, scale=scale, is_causal=length > 1).transpose(1, 2))
        start = stop
    return torch.cat(outputs, dim=1).reshape(1, query.shape[2], -1)


class RowsAttention(torch.autograd.Function):
    """The per-row attention, keeping for the backward pass only q and the grouped k/v (before
    they are repeated for the 12 query heads): 4 KB per token instead of q, the repeated k/v and
    the output (12 KB), a third of the bytes to offload. The backward pass recomputes the
    attention forward from them (deterministic kernels, the same numbers; a few percent of a
    layer's work) and differentiates it."""

    @staticmethod
    def forward(ctx, query, key, value, rows, groups, scale):
        ctx.save_for_backward(query, key, value)
        ctx.rows, ctx.groups, ctx.scale = rows, groups, scale
        return _attend(query, key, value, rows, groups, scale)

    @staticmethod
    def backward(ctx, grad):
        query, key, value = ctx.saved_tensors
        with torch.enable_grad():
            inputs = tuple(t.detach().requires_grad_(need) for t, need
                           in zip((query, key, value), ctx.needs_input_grad[:3]))
            out = _attend(*inputs, ctx.rows, ctx.groups, ctx.scale)
            wanted = [t for t in inputs if t.requires_grad]
            grads = iter(torch.autograd.grad(out, wanted, grad)) if wanted else iter(())
        return tuple(next(grads) if t.requires_grad else None for t in inputs) + \
            (None, None, None)


def patch_qwen2_unpad():
    from transformers.models.qwen2 import modeling_qwen2

    model_cls = modeling_qwen2.Qwen2Model
    if getattr(model_cls.forward, "_kc_patched", False):
        return
    model_forward = model_cls.forward
    layer_forward = modeling_qwen2.Qwen2DecoderLayer.forward
    attention_forward = modeling_qwen2.Qwen2Attention.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        global _ctx, _last_ctx
        if _ctx is None:
            _last_ctx = None  # a new forward: never reuse an earlier batch's positions
        mask = attention_mask
        caching = use_cache if use_cache is not None else getattr(self.config, "use_cache", False)
        if (_ctx is not None or not self.training or caching or past_key_values is not None
                or not torch.is_tensor(mask) or mask.dim() != 2 or not mask.is_cuda
                or getattr(self.config, "_attn_implementation", None) not in ("sdpa", "eager")
                or bool(mask.all())):
            return model_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                 position_ids=position_ids, past_key_values=past_key_values,
                                 inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        batch, seq = mask.shape
        lengths = mask.sum(1)
        prefix = torch.arange(seq, device=mask.device)[None, :] < lengths[:, None]
        rows = lengths.tolist() if bool((mask.bool() == prefix).all()) else None
        _ctx = (mask.reshape(-1).nonzero().squeeze(1), batch, seq, rows)
        _last_ctx = _ctx
        try:
            outputs = model_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                    position_ids=position_ids, past_key_values=past_key_values,
                                    inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
            if _is_packed(outputs.last_hidden_state):
                outputs.last_hidden_state = _unpack(outputs.last_hidden_state, *_ctx[:3])
        finally:
            _ctx = None
            _rope_cache.clear()
        return outputs

    def decoder_layer(self, hidden_states, *args, **kwargs):
        global _ctx
        ctx = _ctx
        if (ctx is None and _last_ctx is not None and torch._C._current_graph_task_id() != -1
                and hidden_states.dim() == 3
                and tuple(hidden_states.shape[:2]) in ((1, _last_ctx[0].numel()),
                                                       (_last_ctx[1], _last_ctx[2]))):
            # gradient checkpointing re-runs this layer inside the backward pass of the latest
            # forward (only there: the autograd engine is running) with that forward's positions
            ctx = _last_ctx
        if ctx is None:
            return layer_forward(self, hidden_states, *args, **kwargs)
        if hidden_states.dim() == 3 and tuple(hidden_states.shape[:2]) == (ctx[1], ctx[2]):
            hidden_states = _pack(hidden_states, ctx[0])  # the first layer: pack once
        previous, _ctx = _ctx, ctx
        try:
            return layer_forward(self, hidden_states, *args, **kwargs)
        finally:
            _ctx = previous

    def attention(self, hidden_states, position_embeddings, attention_mask,
                  past_key_values=None, **kwargs):
        if not _is_packed(hidden_states) or past_key_values is not None:
            return attention_forward(self, hidden_states, position_embeddings, attention_mask,
                                     past_key_values=past_key_values, **kwargs)
        index, batch, seq, rows = _ctx
        if rows is not None and self.config._attn_implementation == "sdpa" \
                and not kwargs.get("output_attentions", False):
            return _row_attention(self, hidden_states, position_embeddings, index, seq, rows)
        hidden_shape = (batch, seq, -1, self.head_dim)

        def padded(proj):
            return _unpack(proj(hidden_states), index, batch, seq).view(hidden_shape).transpose(1, 2)

        query_states = padded(self.q_proj)
        key_states = padded(self.k_proj)
        value_states = padded(self.v_proj)
        cos, sin = position_embeddings
        query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(
            query_states, key_states, cos, sin)
        attention_interface = modeling_qwen2.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, modeling_qwen2.eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
        attn_output = _pack(attn_output.reshape(batch, seq, -1), index)
        return self.o_proj(attn_output), attn_weights

    for fn in (forward, decoder_layer, attention):
        fn._kc_patched = True
    model_cls.forward = forward
    modeling_qwen2.Qwen2DecoderLayer.forward = decoder_layer
    modeling_qwen2.Qwen2Attention.forward = attention
