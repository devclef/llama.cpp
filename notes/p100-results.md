# Dual Tesla P100 (R740): measured results and recommended setup

Hardware: 2x Tesla P100-PCIE-16GB (sm_60), 2x Xeon Gold 6126, 512GB DDR4, CUDA 12.4.
Model: `Qwen3.8-27B-UD-Q4_K_XL` (qwen35 hybrid: 16 full-attention + 48 Gated DeltaNet layers, D=256, GQA 6).

## Headline (llama-server, 262k context, same requests)

| scenario (server, ~47k context) | original setup | new setup | change |
|---|---:|---:|---:|
| 46.7k token prompt | 166 t/s | 272 t/s | +64% |
| 784 token follow-up on that context | 87 t/s | 165 t/s | +89% |
| generation at 47k depth | 10.7 t/s | 18.3 t/s | +71% |
| generation (short context, chat) | ~12 t/s | 20.3 t/s | +69% |

The production log on the old setup showed 40-70 t/s prompt and 5-10 t/s generation at 100k+ context.
Part of that was `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`: once the KV cache no longer fits in VRAM it pages over PCIe.

llama-bench, 27B, empty context:

| config | pp512 | pp2048 | tg64 |
|---|---:|---:|---:|
| upstream, `-sm layer`, ub 512 | 146 | 189 | 12.3 |
| upstream, `-sm layer`, ub 2048 | 147 | 233 | 12.3 |
| upstream, `-sm tensor` (butterfly allreduce) | | 305 | 15.4 |
| this branch, `-sm tensor` (internal allreduce) | 230 | 378 | 18.2 |
| this branch, `-sm tensor` + Pascal MMVQ table | 230 | 378 | 20.7 |
| this branch, `-sm layer` + Pascal MMVQ table | 146 | 233 | 15.7 |

## Recommended server command

```
llama-server --model Qwen3.8-27B-UD-Q4_K_XL.gguf -np 1 -ngl 99 --fit off \
  -sm tensor --ctx-size 262144 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
  --host 0.0.0.0 ... (sampling args unchanged)
```

- No `GGML_CUDA_ENABLE_UNIFIED_MEMORY`. With unified memory, CUDA reports ~521GB free per GPU, so `--fit` never shrinks anything and overflow is paged over PCIe.
- `-sm tensor` uses both GPUs at the same time (needs the allreduce fix in this branch to be fast on Pascal).
- `-ctk/-ctv q8_0` so that 262k context fits in VRAM. With 2x16GB it is 15.0GB used per GPU. f16 KV does not fit at 262k.
- `-ub 2048`: best measured for prompt processing (512: -30%, 1024: -10%, 4096: -4%).
- Optional: MTP speculative decoding, `--spec-type draft-mtp --spec-draft-n-max 2 -ctkd q8_0 -ctvd q8_0`, gives +12% generation (84% acceptance).
  It only fits with `--ctx-size 131072`. At 196608 it loads but runs out of memory on long prompts. At 229376 and 262144 it does not load.

## Code changes in this branch

1. `ggml-cuda/allreduce.cu`: allow the internal 2-GPU AllReduce on Pascal.
   It required Volta only for `__nanosleep` in a spin loop, so Pascal now busy-waits instead.
   This gives `-sm tensor` +24% pp and +18% tg over the butterfly fallback. Perplexity is unchanged (3.3267 layer vs 3.3285 tensor, ±0.079).
2. `ggml-cuda/mmvq.cu`: Pascal parameter table. ncols 1 uses 4 rows per block, and ncols 2-4 use 2 warps x 4 rows.
   Batch-1 mat-vec gets 14-28% faster (q4_K 149->117us, q5_K 175->141, q6_K 204->175, iq4_xs 122->88, q8_0 200->170 for 4096x14336).
   test-backend-ops MUL_MAT 1104/1104 and MUL_MAT_ID 933/933 pass.
3. `ggml-cuda/ggml-cuda.cu`: `GGML_CUDA_FORCE_GRAPHS=1` opts into CUDA graphs on pre-Volta GPUs. That is +35% tg for the Flash-Next MoE (CPU experts) and neutral for the 27B.

## Things measured that did NOT help (don't retry)

- Chunked Gated DeltaNet graph instead of the fused kernel: 210 vs 290 t/s pp8192, so the fused kernel is better. The GDN kernel is only ~5% of prefill time anyway.
- Flash-attention tile kernel configs for D=256: the default (256 threads, occupancy 2, nbatch_fa 64, nbatch_K 64) is the best of 11 variants.
- Accumulating KQ in half2 inside the FA tile kernel: correct (4040/4040 tests) but 15% slower.
- `-fa off` at 32k depth: 99 vs 108 t/s.
- Pipeline parallelism (`-b` > `-ub` with `-sm layer`): no measurable gain. It also needs 6GB extra compute buffers at 262k.
- Faster `dp4a` emulation (PRMT + 16-bit XMAD): correct but no speedup, so MMVQ is not ALU-bound there.
- AllReduce env knobs (BF16 threshold, copy threshold): the defaults are best.

## Profile (27B prefill at ~16k average depth, `-sm layer`)

| kernel | share |
|---|---:|
| cuBLAS FP16 GEMM (maxwell_hgemm) | 60% (~15 TFLOPS, ~80% of P100 FP16 peak) |
| flash_attn_tile<256,256> | 20% (~5.5 TFLOPS, grows with depth) |
| gated_delta_net | 5% |
| dequant / convert | ~5% |

Flash attention is the biggest remaining kernel-level opportunity for long-context prefill. It runs at ~30% of FP16 peak versus 80% for the GEMMs.

## Hardware notes

- GPU0 (slot 1, bus 3b:00.0) is an x16 card running at **x8 (downgraded)**. With `-sm tensor`, the allreduce stages data through host memory, so an x16 link should help both pp and tg.
  Reseat the card or move it to a true x16 slot.
- The GPUs are on different sockets (GPU0 on NUMA 0, GPU1 on NUMA 1, topology SYS), so there is no P2P.

## Qwen3.8 Flash-Next (qwen4exp, 177B total / A3B, UD-Q4_K_XL = 104GB)

It does not fit in VRAM, so the experts live in system RAM. Best measured:

| config | pp4096 | tg |
|---|---:|---:|
| `-ncmoe 99`, mmap (default) | ~10 (pp512) | 4.1 |
| + `-lm none` (no mmap, pinned host buffer) | 42 (ub 512) | 5.1 |
| + `-ub 4096` (amortizes expert upload over PCIe) | 138 | 5.1 |
| + `GGML_CUDA_FORCE_GRAPHS=1` | 138 | 6.9 |
| + `-ncmoe 44` (4 expert layers on GPU) | 140 | 7.2 |

Use `numactl --interleave=all`, `-t 48`. `--no-host` (repacked CPU experts) gives the same tg but pp drops to 50.
Generation is limited by CPU expert compute plus CPU<->GPU handoffs in all 48 layers (OpenMP threads idle ~73% of the time).
**Conclusion: on this box the 27B (272 t/s pp, 18-20 t/s tg) clearly beats Flash-Next (140 t/s pp, 7 t/s tg) at 262k context.**

## Round 2: live-server analysis (server left running)

Real workload from the server log: agent loop at ~92k context, short appended prompts (110-460 tokens) then generation.
Generation there is 73.5 ms/token (13.6 t/s) vs ~49 ms at empty context. Lifetime average from `/metrics`: 12.4 t/s.

- **BIOS power profile is the biggest config problem.** `IA32_MISC_ENABLE` bit 38 = 1 (turbo disabled), HWP max = 26 (2.6 GHz),
  EPP = 255 and EPB = 15 (max power saving), C6 enabled, `intel_pstate: P-states controlled by the platform` (Dell DAPC).
  The busy launch thread runs at exactly 2600 MHz, idle cores at 1000 MHz. Turbo ratios available: 3.7 (1 core), 3.5 (2), 3.3 GHz (all).
  Fix in BIOS / iDRAC: System Profile = Performance (turbo on, C-states off, max performance). Needs a reboot.
- `perf` on the live main thread during generation: 55% libcuda + 27% libc (`pthread_mutex_lock` inside kernel launch), ~5% in synchronize.
  Kernel launch microbenchmark (2.6 GHz): 7.1 us/launch on one GPU, 8.5 us alternating two GPUs, 5.3 us effective with one thread per GPU,
  **1.6 us per kernel inside a CUDA graph**. So `GGML_CUDA_FORCE_GRAPHS=1` should be re-tested with `-sm tensor` (previous "neutral" result was not
  measured with tensor split). Needs a server restart.
- Decode attention with q8_0 KV (D=256, GQA 6, per GPU 12 Q / 2 KV heads, 64k context, one layer):
  | kernel | us |
  |---|---:|
  | VEC, q8_0 (before) | 953 |
  | VEC, q8_0, one block per KV head for all 6 Q heads (tried, reverted) | 1365 |
  | TILE, q8_0 converted to f16 first (existing path) | 1755 |
  | TILE, f16 KV | 387 |
  | TILE, q8_0 read directly, 64 threads/block | 518 |
  | **TILE, q8_0 read directly, 128 threads/block, occupancy 3 (this branch)** | **431** |
  | TILE, f16 KV, 128 threads/block, occupancy 3 (this branch) | 361 |
  The VEC kernel is compute bound on P100 (no native dp4a, 32-lane reductions per key); L2 already absorbs the GQA re-reads.
  The q8_0 tile loader has each thread dequantize one whole block from aligned 32 bit loads (blocks are only 2 byte aligned).
  At 92k depth this saves ~11 ms/token per GPU (expected ~13.6 -> ~16 t/s generation, not yet measured end to end).
- Mixed KV types (f16/q8_0) are slower than both (1280 us). Thread split variants of the VEC kernel: at best -5%.
- Flash-Next fully on CPU (`-ngl 0`): 1.0-1.4 t/s. Experts are mostly `q5_1` (`ffn_down_exps`, 27 GB), which has no CPU repack path;
  libgomp barrier spinning is 18% of CPU time. The hybrid setup (7 t/s) is limited by per-layer GPU<->CPU handoffs, which the power-saving
  BIOS profile makes worse (thread wake-up from C6, cores ramping from 1.0 GHz).

## Remaining ideas (not done)

- Pascal-specific flash attention for prompt batches (the tile kernel reaches 30% of FP16 peak).
- MMVQ still reaches only ~280-350GB/s of 732GB/s on P100 even after the table change. A dequant-to-half2 GEMV might do better, but the dp4a experiment suggests the limit is memory-access latency rather than ALU.
- Put the AllReduce pinned buffers on the NUMA node of each GPU.
