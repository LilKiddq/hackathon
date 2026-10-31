"""Correctness of your kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1000)])
def test_rms_norm_matches_pytorch(shape):
    from solution.rmsnorm import rms_norm

    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.rand(shape[-1], device="cuda", dtype=torch.float32) + 0.5
    grad = torch.randn(shape, device="cuda", dtype=torch.float32)

    ref = weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
    ref.backward(grad)
    ref_dx = x.grad
    x.grad = None

    out = rms_norm(x, weight, 1e-6)
    out.backward(grad)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad, ref_dx, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("rows,vocab", [(700, 151936), (37, 5000)])
def test_fused_lm_head_ce_matches_pytorch(rows, vocab):
    from solution.fused_ce import fused_lm_head_ce

    torch.manual_seed(0)
    hidden = torch.randn(rows, 256, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = (torch.randn(vocab, 256, device="cuda") * 0.05).half()
    target = torch.randint(0, vocab, (rows,), device="cuda")
    target[::3] = -100
    denominator = (target != -100).sum().float()
    scale = 1024.0  # like the fp16 grad scaler

    # base Soup: fp16 head under autocast, fp32 cross-entropy, metrics from the same logits
    logits = hidden.half() @ weight.t()
    ref = torch.nn.functional.cross_entropy(logits.float(), target, ignore_index=-100,
                                            reduction="sum") / denominator
    (ref * scale).backward()
    ref_grad, hidden.grad = hidden.grad, None
    logp = torch.log_softmax(logits.float(), -1)
    ref_entropy = -(logp.exp() * logp).sum(-1)
    labelled = target != -100
    ref_hits = ((logits.argmax(-1) == target) & labelled).sum()
    ref_rows = torch.where(labelled, -logp.gather(-1, target.clamp(min=0)[:, None])[:, 0], 0.0)

    loss, row_loss, entropy, hit = fused_lm_head_ce(hidden, weight, target, denominator)
    (loss * scale).backward()
    torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(row_loss, ref_rows, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-4, atol=1e-4)
    assert int(hit.sum()) == int(ref_hits)
    torch.testing.assert_close(hidden.grad, ref_grad, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("shape", [(1536, 1536), (8960, 1536), (256, 1536)])
def test_nf4_dequant_matches_bitsandbytes(shape):
    import bitsandbytes.functional as F
    from solution.nf4 import NF4Linear, dequantize_nf4

    torch.manual_seed(0)
    w = (torch.randn(shape, device="cuda") * 0.02).half()
    packed, state = F.quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=True)
    ref = F.dequantize_4bit(packed, state).half()
    torch.testing.assert_close(dequantize_nf4(packed, state), ref, rtol=0, atol=0)

    x = torch.randn(3, 77, shape[1], device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(3, 77, shape[0], device="cuda", dtype=torch.float16)
    out = NF4Linear.apply(x, packed, state)
    out.backward(grad)
    xr = x.detach().requires_grad_(True)
    ref_out = xr @ ref.t()
    ref_out.backward(grad)
    torch.testing.assert_close(out, ref_out, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad, xr.grad, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("table_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("batch,seq,dim,table_batch", [(4, 512, 128, 4), (3, 77, 96, 1)])
@pytest.mark.parametrize("used", ["both", "q", "k"])
def test_rope_matches_qwen2(dtype, table_dtype, batch, seq, dim, table_batch, used):
    from transformers.models.qwen2 import modeling_qwen2
    from solution.rope import patch_qwen2_rope

    patch_qwen2_rope()
    patched = modeling_qwen2.apply_rotary_pos_emb
    original = patched.__wrapped__
    patch_qwen2_rope()
    assert modeling_qwen2.apply_rotary_pos_emb is patched

    torch.manual_seed(17)
    # Projection views in Qwen2 are noncontiguous [B, H, T, D].
    q = torch.randn(batch, seq, 12, dim, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
    k = torch.randn(batch, seq, 2, dim, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
    # Strided tables and independent halves also catch incorrect inverse-rotation backward.
    angles = torch.randn(table_batch, seq, dim * 2, device="cuda")
    cos, sin = angles.cos().to(table_dtype)[..., ::2], angles.sin().to(table_dtype)[..., ::2]
    out_dtype = torch.promote_types(dtype, table_dtype)
    gq = torch.randn(batch, seq, 12, dim, device="cuda", dtype=out_dtype).transpose(1, 2)
    gk = torch.randn(batch, seq, 2, dim, device="cuda", dtype=out_dtype).transpose(1, 2)
    outputs = (0, 1) if used == "both" else ((0,) if used == "q" else (1,))

    with torch.autocast("cuda", dtype=torch.float16, enabled=dtype == torch.float16):
        ref = original(q, k, cos, sin)
        ref_grads = torch.autograd.grad(tuple(ref[i] for i in outputs), (q, k),
                                       tuple((gq, gk)[i] for i in outputs), allow_unused=True)
        out = patched(q, k, cos, sin)
        grads = torch.autograd.grad(tuple(out[i] for i in outputs), (q, k),
                                   tuple((gq, gk)[i] for i in outputs), allow_unused=True)
    tol = TOLERANCE[dtype]
    for actual, expected in zip(out, ref):
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    for actual, expected in zip(grads, ref_grads):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("unsqueeze_dim", [1, 2])
def test_rope_cpu_fallback(dtype, unsqueeze_dim):
    from transformers.models.qwen2 import modeling_qwen2
    from solution.rope import patch_qwen2_rope

    patch_qwen2_rope()
    patched = modeling_qwen2.apply_rotary_pos_emb
    original = patched.__wrapped__
    shape = (2, 3, 5, 8) if unsqueeze_dim == 1 else (2, 5, 3, 8)
    q = torch.randn(shape, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, dtype=dtype, requires_grad=True)
    cos = torch.randn(2, 5, 8, dtype=dtype, requires_grad=True)
    sin = torch.randn_like(cos, requires_grad=True)
    ref = original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    out = patched(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    for actual, expected in zip(out, ref):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = tuple(torch.randn_like(t) for t in ref)
    ref_grads = torch.autograd.grad(ref, (q, k, cos, sin), grad)
    grads = torch.autograd.grad(out, (q, k, cos, sin), grad)
    for actual, expected in zip(grads, ref_grads):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("delta_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1000)])
def test_add_rmsnorm_matches_pytorch(delta_dtype, shape):
    from solution.add_rmsnorm import AddRMSNorm

    torch.manual_seed(0)
    residual = torch.randn(shape, device="cuda", requires_grad=True)
    delta = torch.randn(shape, device="cuda").to(delta_dtype).requires_grad_()
    weight = torch.rand(shape[-1], device="cuda") + 0.5
    grad_s, grad_y = torch.randn(shape, device="cuda"), torch.randn(shape, device="cuda")

    s = residual + delta
    y = weight * (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6))
    torch.autograd.backward((s, y), (grad_s, grad_y))
    ref_dr, ref_dd = residual.grad, delta.grad
    residual.grad = delta.grad = None

    s2, y2 = AddRMSNorm.apply(residual, delta, weight, 1e-6)
    torch.autograd.backward((s2, y2), (grad_s, grad_y))
    tol = TOLERANCE[delta_dtype]
    torch.testing.assert_close(s2, s, rtol=0, atol=0)
    torch.testing.assert_close(y2, y, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(residual.grad, ref_dr, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(delta.grad, ref_dd, rtol=tol, atol=tol)


def test_rope_fp16_store_under_autocast_matches_cast():
    """Under fp16 autocast the patched RoPE stores fp32 math as fp16: the same values attention
    gets from casting transformers' fp32 output, and the same fp32 gradients."""
    from transformers.models.qwen2 import modeling_qwen2
    from solution.rope import patch_qwen2_rope

    patch_qwen2_rope()
    patched = modeling_qwen2.apply_rotary_pos_emb
    original = patched.__wrapped__
    torch.manual_seed(3)
    q = torch.randn(4, 77, 12, 128, device="cuda").transpose(1, 2).requires_grad_()
    k = torch.randn(4, 77, 2, 128, device="cuda").transpose(1, 2).requires_grad_()
    angles = torch.randn(4, 77, 64, device="cuda").repeat(1, 1, 2)
    cos, sin = angles.cos(), angles.sin()
    gq = torch.randn(4, 12, 77, 128, device="cuda", dtype=torch.float16)
    gk = torch.randn(4, 2, 77, 128, device="cuda", dtype=torch.float16)

    with torch.autocast("cuda", dtype=torch.float16):
        ref_q, ref_k = (t.half() for t in original(q, k, cos, sin))
        ref_grads = torch.autograd.grad((ref_q, ref_k), (q, k), (gq, gk))
        out_q, out_k = patched(q, k, cos, sin)
        grads = torch.autograd.grad((out_q, out_k), (q, k), (gq, gk))
    assert out_q.dtype == out_k.dtype == torch.float16
    torch.testing.assert_close(out_q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(out_k, ref_k, rtol=0, atol=0)
    for actual, expected in zip(grads, ref_grads):
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_offloaded_swiglu_matches_and_frees_gpu_memory():
    """A chain of MLP-like layers: the offloaded SwiGLU gives the same outputs and gradients,
    and once its transfers are done the gate/up pairs no longer occupy GPU memory."""
    from solution.offload import OFFLOADER, OffloadedSwiGLU, arena_bytes
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    layers, tokens, width, inter = 6, 333, 256, 1024
    weights = [((torch.randn(width, inter, device="cuda") * 0.05).half(),
                (torch.randn(width, inter, device="cuda") * 0.05).half(),
                (torch.randn(inter, width, device="cuda") * 0.05).half()) for _ in range(layers)]
    x0 = torch.randn(tokens, width, device="cuda", dtype=torch.float16)
    grad_out = torch.randn(tokens, width, device="cuda", dtype=torch.float16)

    def run(act):
        x = x0.clone().requires_grad_()
        h = x
        for wg, wu, wd in weights:
            h = h + act(h @ wg, h @ wu) @ wd
        return x, h

    before = torch.cuda.memory_allocated()
    x, ref = run(swiglu)
    kept_plain = torch.cuda.memory_allocated() - before
    ref.backward(grad_out)
    ref_grad = x.grad
    del x

    config = type("Config", (), {"num_hidden_layers": layers, "intermediate_size": inter})()
    OFFLOADER.new_step(arena_bytes(config, tokens))
    try:
        before = torch.cuda.memory_allocated()
        x, out = run(OffloadedSwiGLU.apply)
        torch.cuda.synchronize()
        OFFLOADER._retire()
        assert all(entry.host is not None and entry.gpu is None for entry in OFFLOADER.entries)
        kept = torch.cuda.memory_allocated() - before
        pairs = layers * 2 * tokens * inter * 2  # every layer's gate and up, fp16
        assert kept <= kept_plain - 0.9 * pairs  # they are no longer on the GPU
        out.backward(grad_out)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, ref_grad, rtol=0, atol=0)
    finally:
        OFFLOADER.armed = False


def test_offloaded_swiglu_without_synchronization():
    """Forward and backward straight after each other, transfers still in flight."""
    from solution.offload import OFFLOADER, OffloadedSwiGLU, arena_bytes
    from solution.swiglu import swiglu

    torch.manual_seed(1)
    layers, tokens, width, inter = 8, 517, 256, 2048
    # small weights: SwiGLU grows like h^2, and 8 layers of 0.05 overflow fp16 to inf/NaN
    weights = [((torch.randn(width, inter, device="cuda") * 0.01).half(),
                (torch.randn(width, inter, device="cuda") * 0.01).half(),
                (torch.randn(inter, width, device="cuda") * 0.01).half()) for _ in range(layers)]
    x0 = torch.randn(tokens, width, device="cuda", dtype=torch.float16)
    grad_out = torch.randn(tokens, width, device="cuda", dtype=torch.float16)

    def run(act):
        x = x0.clone().requires_grad_()
        h = x
        for wg, wu, wd in weights:
            h = h + act(h @ wg, h @ wu) @ wd
        h.backward(grad_out)
        return h.detach(), x.grad

    ref, ref_grad = run(swiglu)
    assert torch.isfinite(ref).all() and torch.isfinite(ref_grad).all()
    config = type("Config", (), {"num_hidden_layers": layers, "intermediate_size": inter})()
    try:
        for _ in range(3):  # several steps reuse the same host arena
            OFFLOADER.new_step(arena_bytes(config, tokens))
            out, grad = run(OffloadedSwiGLU.apply)
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
            torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)
            OFFLOADER.new_step(arena_bytes(config, tokens))
            x = x0.clone().requires_grad_()
            h = x
            for wg, wu, wd in weights:
                h = h + OffloadedSwiGLU.apply(h @ wg, h @ wu) @ wd
            OFFLOADER.release_all()  # as before the head
            OFFLOADER.prefetch_last()
            h.backward(grad_out)
            torch.testing.assert_close(x.grad, ref_grad, rtol=0, atol=0)
    finally:
        OFFLOADER.armed = False


def test_saved_tensor_offload_matches_and_frees_gpu_memory():
    """Layers whose saved tensors (attention on transposed views, fp32 norms, matmul inputs) go
    through the offloader's saved-tensor hooks: same outputs and gradients bit for bit, and the
    saved tensors leave the GPU."""
    from solution.offload import OFFLOADER

    torch.manual_seed(2)
    layers, tokens, width, heads = 6, 300, 512, 8
    weights = [(torch.randn(width, 3 * width, device="cuda") * 0.03).half() for _ in range(layers)]
    x0 = torch.randn(tokens, width, device="cuda")
    grad_out = torch.randn(tokens, width, device="cuda")

    def layer(h, w):
        n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)  # fp32, saved
        q, k, v = (n.half() @ w).view(1, tokens, 3, heads, -1).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # dense non-contiguous views
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return h + a.transpose(1, 2).reshape(tokens, width).float()

    def run(hooked):
        x = x0.clone().requires_grad_()
        h = x
        for index, w in enumerate(weights):
            if hooked:
                OFFLOADER.layer = index
                with torch.autograd.graph.saved_tensors_hooks(OFFLOADER.pack, OFFLOADER.unpack):
                    h = layer(h, w)
                OFFLOADER.layer = None
            else:
                h = layer(h, w)
        return x, h

    x, ref = run(False)
    ref.backward(grad_out)
    ref_grad = x.grad
    del x
    try:
        for _ in range(2):
            OFFLOADER.new_step(64 << 20)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            x, out = run(True)
            OFFLOADER.release_all()
            assert OFFLOADER.entries and all(e.host is not None for e in OFFLOADER.entries)
            assert torch.cuda.memory_allocated() - before < 8 * tokens * width * 4
            OFFLOADER.prefetch_last()
            out.backward(grad_out)
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, ref_grad, rtol=0, atol=0)
            del x, out
    finally:
        OFFLOADER.armed = False


def test_warmup_leaves_the_global_rng_alone():
    from solution import warmup

    state = torch.cuda.get_rng_state()
    warmup._compile_all()
    assert torch.equal(torch.cuda.get_rng_state(), state)


def test_unpad_never_reuses_an_earlier_batch():
    """A padded batch, then a batch without padding of the same shape: the second one must run
    on all of its tokens (it once reused the first batch's positions)."""
    from transformers import Qwen2Config, Qwen2Model
    from solution.unpad import patch_qwen2_unpad

    patch_qwen2_unpad()
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, attn_implementation="sdpa")
    model = Qwen2Model(config).cuda()
    ids = torch.randint(0, 128, (4, 9), device="cuda")
    padded = torch.ones(4, 9, dtype=torch.long, device="cuda")
    padded[1, 6:] = 0
    padded[3, 4:] = 0
    full = torch.ones(4, 9, dtype=torch.long, device="cuda")

    model.eval()
    with torch.no_grad():
        expected = model(input_ids=ids, attention_mask=full, use_cache=False).last_hidden_state
    model.train()
    out = model(input_ids=ids, attention_mask=padded, use_cache=False).last_hidden_state
    out.sum().backward()
    got = model(input_ids=ids, attention_mask=full, use_cache=False).last_hidden_state
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
