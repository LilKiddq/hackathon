# Kernel micro-benchmarks (Kaggle Tesla T4)

`python tools/bench_kernels.py` at commit ba9b261. 5 warm-ups, median of 50 CUDA-event samples,
forward + backward (NF4: dequantization only), fp16 autocast. Memory: peak allocated MiB
including inputs. Max diff: over outputs, metrics and input gradients (CE with loss scale 1024).

| kernel | PyTorch ms | ours ms | speedup | PyTorch MB | ours MB | max diff |
|---|---:|---:|---:|---:|---:|---:|
| fused_ce (2048 x 1536 x 151936, fwd+bwd) | 197.3435 | 107.5612 | 1.83x | 5819.77 | 634.75 | 4.57764e-05 |
| rmsnorm (4 x 512 x 1536, fp32, fwd+bwd) | 1.4721 | 0.4726 | 3.11x | 112.26 | 64.26 | 1.43051e-06 |
| rope (Q=4x12x512x128, K=4x2x512x128, fp32, fwd+bwd) | 1.7674 | 0.6867 | 2.57x | 97.00 | 85.00 | 0 |
| nf4 (8960 x 1536, block=64, double quant) | 0.3413 | 0.2227 | 1.53x | 50.09 | 49.27 | 0 |
| nf4 (1536 x 1536, block=64, double quant) | 0.1526 | 0.0961 | 1.59x | 22.31 | 22.16 | 0 |

`python -m pytest solution -q` on the same T4: 43 passed.
