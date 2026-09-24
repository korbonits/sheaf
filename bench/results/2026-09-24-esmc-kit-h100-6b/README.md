# ESM C kit benchmark — esmc_6b

- GPU: NVIDIA H100 80GB HBM3 (cc 9.0, 81079 MiB); nvidia-smi: ['NVIDIA H100 80GB HBM3, 580.95.05, 81559 MiB, 1980 MHz, 700.00 W']
- torch 2.11.0+cu130 / CUDA 13.0 / cuDNN 91900 / triton 3.6.0
- flash-attn 2.7.4.post1 / Transformer Engine 2.15.0 / Python 3.12.1
- esm 3.4.0 (https://github.com/Biohub/esm.git@43ccece2ad485f27db46afdb67da2a9601e8f106)
- transformers 4.57.6 (https://github.com/huggingface/transformers.git@ef32577f55da19a4989cd7b22e004dc43a4998cb)
- esmc_opt 0.4.0; kits https://github.com/anthropics/uplifting-biomolecular-modeling.git@f4f62fa6592ae4938d49b1757bea0cfeff9f468e
- sheaf None @ 4bd26af95802b9011484b9027287e88c4d2cb16f
- measured 2026-09-24T05:37:44+00:00; 20 timed iterations per cell (3 warm-up)

Load time: off 22.7 s, exact 23.2 s

`forward` = model call only (where the kit acts). `predict` = `backend.predict` end to end, incl. JSON serialisation of logits (same cost in every mode). Speedup = off p50 / mode p50.

| L | B | mode | forward p50 ms | p95 ms | speedup | residues/s | predict p50 ms | predict speedup | peak alloc MiB |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | off | 78.49 | 80.62 | 1.00x | 815 | 79.5 | 1.00x | 12,262 |
| 64 | 1 | exact | 30.09 | 30.51 | 2.61x | 2,127 | 30.1 | 2.64x | 12,287 |
| 64 | 8 | off | 82.60 | 83.76 | 1.00x | 6,199 | 83.0 | 1.00x | 12,292 |
| 64 | 8 | exact | 32.59 | 34.25 | 2.53x | 15,710 | 33.8 | 2.45x | 12,322 |
| 64 | 32 | off | 96.45 | 98.51 | 1.00x | 21,234 | 110.2 | 1.00x | 12,383 |
| 64 | 32 | exact | 56.26 | 58.10 | 1.71x | 36,400 | 64.5 | 1.71x | 12,437 |
| 128 | 1 | off | 76.48 | 79.92 | 1.00x | 1,674 | 75.7 | 1.00x | 12,267 |
| 128 | 1 | exact | 27.71 | 27.95 | 2.76x | 4,619 | 29.5 | 2.56x | 12,292 |
| 128 | 8 | off | 83.51 | 86.32 | 1.00x | 12,262 | 86.5 | 1.00x | 12,320 |
| 128 | 8 | exact | 31.14 | 34.54 | 2.68x | 32,884 | 36.9 | 2.34x | 12,361 |
| 128 | 32 | off | 139.02 | 141.75 | 1.00x | 29,463 | 165.1 | 1.00x | 12,505 |
| 128 | 32 | exact | 110.46 | 113.28 | 1.26x | 37,080 | 130.5 | 1.26x | 12,589 |
| 256 | 1 | off | 79.60 | 81.10 | 1.00x | 3,216 | 83.2 | 1.00x | 12,279 |
| 256 | 1 | exact | 28.45 | 28.78 | 2.80x | 8,997 | 31.6 | 2.63x | 12,305 |
| 256 | 8 | off | 96.77 | 98.99 | 1.00x | 21,164 | 104.2 | 1.00x | 12,382 |
| 256 | 8 | exact | 54.89 | 56.42 | 1.76x | 37,310 | 63.0 | 1.65x | 12,436 |
| 256 | 32 | off | 260.32 | 261.19 | 1.00x | 31,469 | 309.7 | 1.00x | 12,748 |
| 256 | 32 | exact | 214.97 | 218.60 | 1.21x | 38,108 | 261.1 | 1.19x | 12,893 |
| 512 | 1 | off | 84.38 | 85.00 | 1.00x | 6,067 | 86.0 | 1.00x | 12,297 |
| 512 | 1 | exact | 29.24 | 32.00 | 2.89x | 17,508 | 31.5 | 2.73x | 12,326 |
| 512 | 8 | off | 138.22 | 140.91 | 1.00x | 29,634 | 155.3 | 1.00x | 12,506 |
| 512 | 8 | exact | 107.95 | 113.82 | 1.28x | 37,944 | 123.6 | 1.26x | 12,589 |
| 512 | 32 | off | 517.45 | 522.69 | 1.00x | 31,663 | 626.1 | 1.00x | 13,235 |
| 512 | 32 | exact | 432.61 | 439.60 | 1.20x | 37,873 | 546.3 | 1.15x | 13,499 |
| 1024 | 1 | off | 81.51 | 83.59 | 1.00x | 12,563 | 88.8 | 1.00x | 12,329 |
| 1024 | 1 | exact | 30.52 | 31.39 | 2.67x | 33,554 | 33.9 | 2.62x | 12,369 |
| 1024 | 8 | off | 274.00 | 279.42 | 1.00x | 29,898 | 322.4 | 1.00x | 12,753 |
| 1024 | 8 | exact | 225.55 | 227.72 | 1.21x | 36,320 | 260.1 | 1.24x | 12,896 |
| 1024 | 32 | off | 1056.06 | 1059.69 | 1.00x | 31,029 | 1306.1 | 1.00x | 14,209 |
| 1024 | 32 | exact | 890.60 | 894.63 | 1.19x | 36,793 | 1116.1 | 1.17x | 14,713 |
| 2048 | 1 | off | 92.57 | 93.10 | 1.00x | 22,124 | 101.0 | 1.00x | 12,399 |
| 2048 | 1 | exact | 62.55 | 67.75 | 1.48x | 32,743 | 70.1 | 1.44x | 12,453 |
| 2048 | 8 | off | 575.66 | 578.54 | 1.00x | 28,461 | 665.5 | 1.00x | 13,247 |
| 2048 | 8 | exact | 485.52 | 489.43 | 1.19x | 33,745 | 578.0 | 1.15x | 13,511 |
| 2048 | 32 | off | 2259.85 | 2264.27 | 1.00x | 29,000 | 2719.9 | 1.00x | 16,156 |
| 2048 | 32 | exact | 1931.18 | 1937.96 | 1.17x | 33,936 | 2464.5 | 1.10x | 17,140 |

Kit lines (exact):

    [esmc-opt] ACTIVE mode=exact variant=6b regime=batched levers=pipe,fused levers_out=none esm=3.4.0 torch=2.11.0 flash_attn=2.7.4.post1 transformer_engine=2.15.0 gpu=NVIDIA_H100_80GB_HBM3(sm90) package=0.4.0
    [esmc-opt] APPLIED model#1 variant=6b regime=batched levers_applied=pipe,fused levers_fallback=none t_apply=12.45
