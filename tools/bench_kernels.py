"""Task-shape micro-benchmarks; run in the Kaggle training environment.

Each implementation gets 5 warm-ups and 50 individually synchronized CUDA-event
samples. Times include forward and backward (NF4 is dequantization only), with
fp16 autocast enabled. RMSNorm and RoPE inputs remain fp32, as in the task.
Peak memory is total allocated MiB, including shared inputs, after resetting
peak stats following warm-up. Reference outputs are not retained on the GPU
while timing our implementation. Max diff covers outputs, metrics and input
gradients; CE uses a fixed 1024 loss scale to exercise fp16 backward rounding.
Compilation and NF4 quantization happen outside the measured samples.
No solution patches are applied, so reference operations remain unmodified.
"""

import gc
from pathlib import Path
import statistics
import sys

# `python tools/bench_kernels.py` puts tools/, rather than the repo, on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WARMUPS = 5
RUNS = 50
LOSS_SCALE = 1024.0


def measure(torch, run, cleanup):
    """Return median GPU milliseconds and peak allocated MiB without keeping outputs."""
    for _ in range(WARMUPS):
        cleanup()
        result = run()
        del result
    cleanup()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(RUNS)]
    # Initialize events before the measured work.
    for start, end in events:
        start.record()
        end.record()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for start, end in events:
        cleanup()  # release leaf gradients; do not accumulate across samples
        start.record()
        result = run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        del result
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    cleanup()
    return statistics.median(times), peak_mb


def compare(torch, name, reference, ours, cleanup):
    print(f"Benchmarking {name}...", file=sys.stderr, flush=True)
    ref_ms, ref_mb = measure(torch, reference, cleanup)
    our_ms, our_mb = measure(torch, ours, cleanup)

    # Copies and the difference calculation are outside timing/memory samples.
    cleanup()
    values = reference()
    ref_values = tuple(value.detach().cpu() for value in values)
    del values
    cleanup()
    values = ours()
    our_values = tuple(value.detach().cpu() for value in values)
    del values
    cleanup()
    if len(ref_values) != len(our_values):
        raise RuntimeError(f"{name}: output count differs")
    differences = []
    for ref, actual in zip(ref_values, our_values):
        if ref.shape != actual.shape or ref.dtype != actual.dtype:
            raise RuntimeError(f"{name}: output shape or dtype differs")
        differences.append((ref.double() - actual.double()).abs().max().item())
    max_diff = max(differences)
    if not all(value == value and value != float("inf") for value in differences):
        raise RuntimeError(f"{name}: non-finite output difference")
    return (name, ref_ms, our_ms, ref_ms / our_ms, ref_mb, our_mb, max_diff)


def bench_ce(torch, functional, fused_lm_head_ce):
    rows, width, vocab = 2048, 1536, 151936
    hidden = torch.randn(rows, width, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(vocab, width, device="cuda", dtype=torch.float16).mul_(0.02)
    target = torch.randint(vocab, (rows,), device="cuda")
    denominator = torch.tensor(float(rows), device="cuda")

    def cleanup():
        hidden.grad = None

    def reference():
        with torch.autocast("cuda", dtype=torch.float16):
            logits = hidden.half() @ weight.t()
            loss = functional.cross_entropy(logits.float(), target, reduction="sum") / denominator
            with torch.no_grad():
                logp = functional.log_softmax(logits.float(), dim=-1)
                entropy = -(logp.exp() * logp).sum(-1)
                row_loss = -logp.gather(-1, target[:, None]).squeeze(-1)
                hit = (logits.argmax(-1) == target).to(torch.int32)
                del logp
        (loss * LOSS_SCALE).backward()
        return loss.detach(), row_loss, entropy, hit, hidden.grad.detach()

    def ours():
        with torch.autocast("cuda", dtype=torch.float16):
            loss, row_loss, entropy, hit = fused_lm_head_ce(hidden, weight, target, denominator)
        (loss * LOSS_SCALE).backward()
        return loss.detach(), row_loss, entropy, hit, hidden.grad.detach()

    return compare(torch, "fused_ce (2048 x 1536 x 151936, fwd+bwd)", reference, ours, cleanup)


def bench_rmsnorm(torch, rms_norm):
    x = torch.randn(4, 512, 1536, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.rand(1536, device="cuda", dtype=torch.float32) + 0.5
    grad = torch.randn_like(x)
    eps = 1e-6

    def cleanup():
        x.grad = None

    def step(forward):
        with torch.autocast("cuda", dtype=torch.float16):
            out = forward()
        out.backward(grad)
        return out.detach(), x.grad.detach()

    def reference_forward():
        values = x.float()
        variance = values.pow(2).mean(-1, keepdim=True)
        return weight * (values * torch.rsqrt(variance + eps)).to(x.dtype)

    return compare(torch, "rmsnorm (4 x 512 x 1536, fp32, fwd+bwd)",
                   lambda: step(reference_forward), lambda: step(lambda: rms_norm(x, weight, eps)),
                   cleanup)


def bench_rope(torch, RoPE, apply_rotary_pos_emb):
    # Match the transposed projection views used by Qwen2 attention.
    q = torch.randn(4, 512, 12, 128, device="cuda", dtype=torch.float32).transpose(1, 2).requires_grad_()
    k = torch.randn(4, 512, 2, 128, device="cuda", dtype=torch.float32).transpose(1, 2).requires_grad_()
    angles = torch.randn(1, 512, 64, device="cuda", dtype=torch.float32).repeat(1, 1, 2)
    cos, sin = angles.cos(), angles.sin()
    grad_q, grad_k = torch.randn_like(q), torch.randn_like(k)

    def cleanup():
        q.grad = k.grad = None

    def step(forward):
        with torch.autocast("cuda", dtype=torch.float16):
            out_q, out_k = forward(q, k, cos, sin)
        torch.autograd.backward((out_q, out_k), (grad_q, grad_k))
        return out_q.detach(), out_k.detach(), q.grad.detach(), k.grad.detach()

    return compare(torch, "rope (Q=4x12x512x128, K=4x2x512x128, fp32, fwd+bwd)",
                   lambda: step(apply_rotary_pos_emb), lambda: step(RoPE.apply), cleanup)


def bench_nf4(torch, bnb_functional, dequantize_nf4, shape):
    weight = torch.randn(shape, device="cuda", dtype=torch.float16).mul_(0.02)
    packed, state = bnb_functional.quantize_4bit(
        weight, blocksize=64, quant_type="nf4", compress_statistics=True)
    del weight

    def reference():
        with torch.autocast("cuda", dtype=torch.float16):
            return (bnb_functional.dequantize_4bit(packed, quant_state=state),)

    def ours():
        with torch.autocast("cuda", dtype=torch.float16):
            return (dequantize_nf4(packed, state, dtype=torch.float16),)

    return compare(torch, f"nf4 ({shape[0]} x {shape[1]}, block=64, double quant)",
                   reference, ours, lambda: None)


def main():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required; run this in the Kaggle T4 training environment.")

    import bitsandbytes.functional as bnb_functional
    from torch.nn import functional
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    from solution.fused_ce import fused_lm_head_ce
    from solution.nf4 import dequantize_nf4
    from solution.rmsnorm import rms_norm
    from solution.rope import RoPE

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"GPU: {torch.cuda.get_device_name()}; warm-ups: {WARMUPS}; runs: {RUNS}")
    print("Times: forward+backward except NF4. Memory: total peak allocated MiB including inputs.")
    print("Max diff: maximum over outputs/metrics/input gradients; CE loss scale=1024, all rows labelled.")
    results = []
    cases = [lambda: bench_ce(torch, functional, fused_lm_head_ce),
             lambda: bench_rmsnorm(torch, rms_norm),
             lambda: bench_rope(torch, RoPE, apply_rotary_pos_emb),
             lambda: bench_nf4(torch, bnb_functional, dequantize_nf4, (8960, 1536)),
             lambda: bench_nf4(torch, bnb_functional, dequantize_nf4, (1536, 1536))]
    for case in cases:
        results.append(case())
        gc.collect()
        torch.cuda.empty_cache()

    print("\n| kernel | PyTorch ms | ours ms | speedup | PyTorch MB | ours MB | max diff |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, ref_ms, our_ms, speedup, ref_mb, our_mb, diff in results:
        print(f"| {name} | {ref_ms:.4f} | {our_ms:.4f} | {speedup:.2f}x | "
              f"{ref_mb:.2f} | {our_mb:.2f} | {diff:.6g} |")


if __name__ == "__main__":
    main()
