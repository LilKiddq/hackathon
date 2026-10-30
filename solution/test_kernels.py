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
