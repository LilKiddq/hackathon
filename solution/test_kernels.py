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
