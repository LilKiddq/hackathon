"""Wiring of the fused head + loss (fused_ce.py) into Soup's trainer.

Two class patches, both looked up at call time:

- `Qwen2ForCausalLM.forward`: while our trainer step asks for it (`_kc_active`), the decoder runs
  as usual, the head is held back, and the loss comes from `fused_lm_head_ce` over the token rows
  that carry attention. The output has the loss and no logits; what the trainer's metrics need is
  left on the model as `_kc_step`. Any other call (generation, eval without labels, tests) takes
  the original forward.
- `trl.SFTTrainer.compute_loss`: TRL reads `outputs.logits` for `entropy` and
  `mean_token_accuracy`. For a fused step we call transformers' own `compute_loss` and log the same
  three columns by TRL's formulas. Soup's rewind recorder (a subclass of SFTTrainer) then finds
  no logits and reads each token's loss from `trainer._soup_chunked_step.token_loss`, the hook
  Soup itself uses for its chunked loss (#1693), so rewind.jsonl keeps the same rows and values.
"""

import atexit
import types

import torch
from torch.nn import functional

from . import offload, warmup
from .wstream import STREAMER, trainable_to_gpu, weights_to_host_while_loading
from .fused_ce import IGNORE_INDEX, fused_lm_head_ce
from .head8 import Int8Embedding, dequant_rows, fused_int8_head_ce, quantize_rows

STEP_ATTR = "_soup_chunked_step"  # soup_cli.utils.stream_loss_chunk.STEP_ATTR, read by rewind


_PEAKS = []  # (where, MiB): each point where the allocator's peak had grown since the last one
_STEP = [0]


def _note_peak(where):
    """Reads the allocator's peak (a host-side counter: no sync, nothing reset)."""
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        peak = torch.cuda.max_memory_allocated() >> 20
        if not _PEAKS or peak > _PEAKS[-1][1]:
            _PEAKS.append((where, peak))


def _report_peaks():
    if _PEAKS:
        import sys

        from .wstream import EMBEDDING_ON_HOST, LOADED_TO_HOST

        print("[solution] allocator peak grew at: "
              + " | ".join(f"{where} {mib} MiB" for where, mib in _PEAKS[-12:])
              + f" | loading: {LOADED_TO_HOST[0]} 4-bit weights and "
              + ("the" if EMBEDDING_ON_HOST[0] else "not the") + " embedding to host",
              file=sys.stderr)


atexit.register(_report_peaks)


def _fp16_weight(module):
    """A cached fp16 copy of the frozen head weight (the tied fp32 embedding). Base autocast makes
    this same cast on every forward and keeps it for the backward; we make it once."""
    weight = module.weight
    cached = getattr(module, "_kc_weight16", None)
    key = (weight.data_ptr(), weight._version, weight.device)
    if cached is None or cached[0] != key:
        cached = (key, weight.detach().to(torch.float16).contiguous())
        module._kc_weight16 = cached
    return cached[1]


def _fused_forward(self, original, input_ids=None, attention_mask=None, position_ids=None,
                   past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                   logits_to_keep=0, **kwargs):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    num_items_in_batch = kwargs.get("num_items_in_batch")
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         position_ids=position_ids, past_key_values=past_key_values,
                         inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
    hidden = outputs.last_hidden_state
    batch, seq, width = hidden.shape

    # as ForCausalLMLoss: position t is scored against labels[t + 1], the last one against nothing
    target = functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].to(hidden.device)
    keep = target != IGNORE_INDEX
    if attention_mask is not None:
        keep = keep | attention_mask.to(hidden.device).bool()  # entropy needs every real token
    else:
        keep = torch.ones_like(keep)
    rows = keep.reshape(-1).nonzero().squeeze(1)
    flat_target = target.reshape(-1)

    if num_items_in_batch is None:
        denominator = (flat_target != IGNORE_INDEX).sum()
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(hidden.device)
    else:
        denominator = torch.tensor(num_items_in_batch, device=hidden.device)

    _note_peak(f"step {_STEP[0]} forward")
    offload.OFFLOADER.release_all()  # the last layers' MLP pairs leave the GPU before the head
    int8 = getattr(self, "_kc_int8", None)
    if int8 is not None:  # the tied weight held as int8 rows (head8.py)
        loss, row_loss, row_entropy, row_hit = fused_int8_head_ce(
            hidden.reshape(-1, width).index_select(0, rows), int8[0], int8[1],
            flat_target.index_select(0, rows), denominator.float())
    else:
        loss, row_loss, row_entropy, row_hit = fused_lm_head_ce(
            hidden.reshape(-1, width).index_select(0, rows), _fp16_weight(self.lm_head),
            flat_target.index_select(0, rows), denominator.float())
    offload.OFFLOADER.prefetch_last()  # and the last one comes back while the backward starts
    STREAMER.prefetch_backward()  # as do the last layer's weights
    _note_peak(f"step {_STEP[0]} head")

    def scatter(values, dtype):
        full = torch.zeros(batch * seq, device=hidden.device, dtype=dtype)
        return full.index_copy_(0, rows, values.to(dtype)).view(batch, seq)

    self._kc_step = types.SimpleNamespace(
        token_loss=scatter(row_loss, torch.float32),        # [B, T], 0 where no label
        token_entropy=scatter(row_entropy, torch.float32),  # [B, T], 0 on padding
        correct=row_hit.sum(),
        labelled=(flat_target != IGNORE_INDEX).sum(),
    )
    return CausalLMOutputWithPast(loss=loss, logits=None, past_key_values=outputs.past_key_values,
                                  hidden_states=outputs.hidden_states,
                                  attentions=outputs.attentions)


def patch_causal_lm():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    if getattr(cls.forward, "_kc_patched", False):
        return
    original = cls.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        if (getattr(self, "_kc_active", False) and labels is not None
                and "shift_labels" not in kwargs and self.training):
            return _fused_forward(self, original, input_ids, attention_mask, position_ids,
                                  past_key_values, inputs_embeds, labels, use_cache,
                                  logits_to_keep, **kwargs)
        return original(self, input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                        logits_to_keep=logits_to_keep, **kwargs)

    forward._kc_patched = True
    forward.__doc__ = original.__doc__
    cls.forward = forward


def _causal_lm(model):
    from transformers.models.qwen2 import modeling_qwen2

    get_base_model = getattr(model, "get_base_model", None)
    inner = get_base_model() if callable(get_base_model) else model
    return inner if isinstance(inner, modeling_qwen2.Qwen2ForCausalLM) else None


def patch_sft_trainer():
    import trl
    from transformers import Trainer

    cls = trl.SFTTrainer
    if getattr(cls.compute_loss, "_kc_patched", False):
        return
    original = cls.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        lm = _causal_lm(model) if isinstance(model, torch.nn.Module) else None
        if (lm is None or not lm.training or not isinstance(inputs, dict)
                or "labels" not in inputs or "shift_labels" in inputs
                or "attention_mask" not in inputs
                or getattr(self.args, "use_liger_kernel", False)
                or getattr(self, "num_virtual_tokens", 0) > 0
                or getattr(self, "aux_loss_enabled", False)
                or not lm.lm_head.weight.is_cuda):
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)

        _note_peak(f"step {_STEP[0]} backward/optimizer")
        _STEP[0] += 1
        warmup.wait()  # the background compilation (warmup.py) is done before the first step
        max_tokens = (getattr(self.args, "per_device_train_batch_size", 0) or 0) * \
            (getattr(self.args, "max_length", 0) or 0) or inputs["input_ids"].numel()
        offload.OFFLOADER.new_step(offload.arena_bytes(lm.config, max_tokens))
        STREAMER.install(lm)  # once: the 4-bit weights move to host memory (wstream.py)
        STREAMER.new_step()
        mode = "train"
        inputs.pop("_prediction_loss_only", None)  # as TRL does before it delegates
        inputs["use_cache"] = False
        lm._kc_active = True
        try:
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
        finally:
            lm._kc_active = False
        step = lm.__dict__.pop("_kc_step")

        # TRL's three columns, by TRL's formulas (trl 0.29.1 SFTTrainer.compute_loss)
        gather = self.accelerator.gather_for_metrics
        attention_mask = inputs["attention_mask"]
        with torch.no_grad():
            entropy = torch.sum(step.token_entropy * attention_mask) / attention_mask.sum()
        self._metrics[mode]["entropy"].append(gather(entropy).mean().item())
        self._total_train_tokens += gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
        correct, total = gather(step.correct), gather(step.labelled)
        total_sum = total.sum()
        accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        setattr(self, STEP_ATTR, types.SimpleNamespace(token_loss=step.token_loss))
        return (loss, outputs) if return_outputs else loss

    compute_loss._kc_patched = True
    cls.compute_loss = compute_loss


def _tied_embedding(lm):
    """The tied embedding / lm_head weight: on the GPU, or in host memory when our loading put it
    there (wstream.weights_to_host_while_loading)."""
    from .wstream import EMBEDDING_ON_HOST

    embed, head = lm.get_input_embeddings(), lm.lm_head
    weight = getattr(embed, "weight", None)
    if weight is None or weight is not getattr(head, "weight", None):
        return None
    if not (weight.is_cuda or (EMBEDDING_ON_HOST[0] and weight.device.type == "cpu"
                               and torch.cuda.is_available())):
        return None
    return weight


def _fp16_tied_embedding(model):
    """Keep the tied embedding / lm_head weight in fp16 instead of fp32.

    prepare_model_for_kbit_training upcasts it to fp32: 151936 x 1536 x 4 bytes = 933 MB, and
    the head still needs an fp16 copy for its matmul. Qwen2.5's checkpoint stores it in bf16,
    and every bf16 value in fp16's normal range is exact in fp16, so the fp16 tensor holds the
    same numbers (the head used fp16 under autocast anyway). The embedding output is widened back
    to fp32 by a forward hook, so the residual stream and everything after it see fp32 as before.
    `patch_kbit_preparation` keeps the upcast from happening at all (its fp32 copy next to the
    fp16 one was the run's memory peak during setup); this handles the weight either way.
    """
    lm = _causal_lm(model)
    if lm is None:
        return
    weight = _tied_embedding(lm)
    if weight is None or weight.requires_grad or getattr(weight, "_kc_fp16", False) is None:
        return
    if weight.dtype == torch.float32:
        half = torch.nn.Parameter(weight.detach().to(torch.float16), requires_grad=False)
        lm.get_input_embeddings().weight = half
        lm.lm_head.weight = half
        del weight
    elif not getattr(weight, "_kc_fp16", False):
        return
    else:
        weight._kc_fp16 = None  # hooked once
    lm.get_input_embeddings().register_forward_hook(lambda module, args, output: output.float())


def _int8_tied_embedding(model):
    """Replace the fp16 tied embedding / lm_head weight by int8 rows + scales (head8.py).

    The embedding looks up and dequantizes its rows; the fused head reads the int8 table. The
    modules keep a `weight` of the right shape and dtype (one fp16 zero, expanded: 2 bytes), and
    any other call of the head (not training's fused step) dequantizes the table in full."""
    lm = _causal_lm(model)
    if lm is None:
        return
    weight = _tied_embedding(lm)
    if weight is None or weight.dtype != torch.float16 or weight.requires_grad \
            or weight.dim() != 2:
        return
    embed, head = lm.get_input_embeddings(), lm.lm_head
    # the int8 rows live in pinned host memory (233 MB off the GPU for the whole run): the head
    # streams them a tile ahead, the embedding gathers its rows on the CPU (head8.py)
    from .wstream import host_bytes

    device = torch.device("cuda", torch.cuda.current_device())
    host = host_bytes(weight.numel())
    q, scale = quantize_rows(weight.detach(), host.view(weight.shape) if host is not None
                             else None, device=device)
    stand_in = torch.nn.Parameter(
        torch.zeros((1, 1), dtype=weight.dtype, device=device).expand(weight.shape),
        requires_grad=False)
    embed.weight = stand_in
    head.weight = stand_in
    del weight
    lm._kc_int8 = (q, scale)

    def embed_forward(ids):
        return Int8Embedding.apply(ids, q, scale)

    def head_forward(hidden):  # any call besides training's fused step: the full table
        return functional.linear(hidden.to(torch.float16),
                                 dequant_rows(q.to(hidden.device), scale))

    embed.forward = embed_forward
    head.forward = head_forward
    torch.cuda.empty_cache()


def patch_kbit_preparation():
    """peft's prepare_model_for_kbit_training without the fp32 copy of the tied embedding.

    It casts every fp16/bf16 parameter to fp32; for the 151936 x 1536 tied embedding that is a
    933 MB tensor allocated next to the 467 MB one it replaces, which `_fp16_tied_embedding`
    then turns back into fp16. Here the embedding sits out the cast (an empty stand-in takes its
    place for the call) and comes back as fp16 holding the same values (bf16 -> fp16 is exact in
    fp16's range; fp16 stays fp16), frozen like every base parameter. Soup imports the function
    from peft at call time."""
    import peft

    original = peft.prepare_model_for_kbit_training
    if getattr(original, "_kc_patched", False):
        return

    def prepare_model_for_kbit_training(model, *args, **kwargs):
        lm = model if hasattr(model, "lm_head") and hasattr(model, "get_input_embeddings") \
            else None
        weight = None
        if lm is not None:
            try:
                weight = _tied_embedding(lm)
            except Exception:  # noqa: BLE001 - anything unusual: peft's own path
                weight = None
        if weight is None or weight.dtype not in (torch.float16, torch.bfloat16):
            return original(model, *args, **kwargs)
        data = weight.data
        weight.data = data.new_empty(0)
        try:
            result = original(model, *args, **kwargs)
        finally:
            weight.data = data if data.dtype == torch.float16 else data.to(torch.float16)
        weight.requires_grad_(False)
        weight._kc_fp16 = True
        return result

    prepare_model_for_kbit_training._kc_patched = True
    peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training


def _start_pinning(cfg=None, tcfg=None, *_):
    """Pin the offload arena (offload.py) in the background while Soup loads the model.

    Pinning ~2 GB of host memory takes seconds; started here it overlaps the model loading that
    follows. The arena's size comes from the checkpoint's config.json in the local Hugging Face
    cache (a file read, nothing is loaded), the batch size and the max length. Without a cached
    config nothing starts, and the first training step reserves the arena itself."""
    try:
        import json

        from huggingface_hub import try_to_load_from_cache

        if not torch.cuda.is_available() or not isinstance(cfg.base, str):
            return
        path = try_to_load_from_cache(cfg.base, "config.json")
        if not isinstance(path, str):
            return
        with open(path, encoding="utf-8") as handle:
            config = types.SimpleNamespace(**json.load(handle))
        max_tokens = int(tcfg.batch_size) * int(cfg.data.max_length)
        nbytes = offload.arena_bytes(config, max_tokens)
        if isinstance(nbytes, int) and nbytes > 0:
            from .wstream import reserve_weight_block

            offload.OFFLOADER.reserve_in_background(nbytes, first=reserve_weight_block(config))
    except Exception:  # noqa: BLE001 - only a head start; training never depends on it
        return


def patch_soup_setup():
    """Starts pinning the offload arena before Soup loads the model, and runs
    `_fp16_tied_embedding` when Soup has built the final 4-bit PEFT model."""
    from soup_cli.trainer import sft

    cls = sft.SFTTrainerWrapper
    if getattr(cls._setup_transformers, "_kc_patched", False):
        return
    original = cls._setup_transformers

    def _setup_transformers(self, *args, **kwargs):
        _start_pinning(*args)
        try:
            patch_kbit_preparation()  # peft is imported here anyway, right before it is used
        except Exception:  # noqa: BLE001 - without it the upcast happens and is undone below
            pass
        with weights_to_host_while_loading() as hosted:
            result = original(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if isinstance(model, torch.nn.Module):
            if hosted and _causal_lm(model) is not None:
                trainable_to_gpu(model)
            _fp16_tied_embedding(model)
            _int8_tied_embedding(model)
            _note_peak("setup")
        return result

    _setup_transformers._kc_patched = True
    cls._setup_transformers = _setup_transformers
