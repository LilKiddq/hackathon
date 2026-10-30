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
- LoRA's dropout draws its mask for the padded shape and then packs it, so every real token
  gets the same random mask as in the padded run and the generator advances the same way.
- The model's output is put back in the padded shape, so the head, loss and metrics see the
  same tensor as before (padding rows are zeros there and are skipped by the fused loss).
"""

import torch

_ctx = None  # (real positions [n] int64, batch, seq) while a padded batch runs unpadded


def _pack(x, index):
    return x.reshape(-1, x.shape[-1]).index_select(0, index).unsqueeze(0)


def _unpack(x, index, batch, seq):
    flat = x.reshape(-1, x.shape[-1])
    full = flat.new_zeros((batch * seq, flat.shape[-1]))
    return full.index_copy(0, index, flat).view(batch, seq, flat.shape[-1])


def _is_packed(x):
    return _ctx is not None and x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == _ctx[0].numel()


def patch_qwen2_unpad():
    from transformers.models.qwen2 import modeling_qwen2

    model_cls = modeling_qwen2.Qwen2Model
    if getattr(model_cls.forward, "_kc_patched", False):
        return
    model_forward = model_cls.forward
    layer_forward = modeling_qwen2.Qwen2DecoderLayer.forward
    attention_forward = modeling_qwen2.Qwen2Attention.forward
    dropout_forward = torch.nn.Dropout.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        global _ctx
        mask = attention_mask
        if (_ctx is not None or not self.training or use_cache or past_key_values is not None
                or not torch.is_tensor(mask) or mask.dim() != 2 or not mask.is_cuda
                or getattr(self.config, "_attn_implementation", None) not in ("sdpa", "eager")
                or bool(mask.all())):
            return model_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                 position_ids=position_ids, past_key_values=past_key_values,
                                 inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        batch, seq = mask.shape
        _ctx = (mask.reshape(-1).nonzero().squeeze(1), batch, seq)
        try:
            outputs = model_forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                    position_ids=position_ids, past_key_values=past_key_values,
                                    inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
            if _is_packed(outputs.last_hidden_state):
                outputs.last_hidden_state = _unpack(outputs.last_hidden_state, *_ctx)
        finally:
            _ctx = None
        return outputs

    def decoder_layer(self, hidden_states, *args, **kwargs):
        global _ctx
        ctx = _ctx
        if ctx is not None:
            self._kc_ctx = ctx
        else:
            # gradient checkpointing re-runs a layer in the backward pass, after the model's
            # forward has returned: use the positions this layer ran with
            saved = getattr(self, "_kc_ctx", None)
            if (saved is not None and torch.is_grad_enabled() and hidden_states.dim() == 3
                    and tuple(hidden_states.shape[:2]) in ((1, saved[0].numel()),
                                                           (saved[1], saved[2]))):
                ctx = saved
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
        index, batch, seq = _ctx
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

    def dropout(self, x):
        # LoRA's dropout on packed tokens: draw the mask for the padded shape, then pack it
        if self.training and self.p > 0 and _is_packed(x):
            index, batch, seq = _ctx
            return _pack(dropout_forward(self, _unpack(x, index, batch, seq)), index)
        return dropout_forward(self, x)

    for fn in (forward, decoder_layer, attention, dropout):
        fn._kc_patched = True
    model_cls.forward = forward
    modeling_qwen2.Qwen2DecoderLayer.forward = decoder_layer
    modeling_qwen2.Qwen2Attention.forward = attention
    torch.nn.Dropout.forward = dropout
