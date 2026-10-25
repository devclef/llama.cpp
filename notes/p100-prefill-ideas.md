# Prompt-processing (prefill) ideas: dual Tesla P100 (sm_60), Qwen3.x hybrid (qwen35 / qwen35moe)

Target: Qwen 27B dense (Q4) and the Qwen "Next/Flash" MoE, 262k ctx, 2x P100 16GB, R740, 512GB DDR4.
Current: ~150 t/s decode, ~12 t/s prefill. These are notes to try later, not implemented changes.

## Hardware facts that drive everything

- P100 = compute capability 6.0. **No DP4A** (that starts at 6.1, `GGML_CUDA_CC_DP4A` in `ggml/src/ggml-cuda/common.cuh:51`), no tensor cores, no CUDA graphs (disabled `< Volta`, `ggml-cuda.cu:4901`).
- It does have fast FP16 (about 19 TFLOPS FP16 vs about 9.5 FP32) and 732 GB/s HBM2. **FP16 FMA is the only fast math path on this GPU.**
- Rough ceiling for dense 27B prefill: about 54 GFLOP/token, so FP16 HGEMM at 50% efficiency is about 170 t/s per GPU. 12 t/s is more than 10x below that, so the main bottleneck is not the dense GEMMs. Find it before tuning anything else.
- Sanity check: 150 t/s decode on a dense 27B Q4 (about 16GB of weights) would need more than 2 TB/s. That number is probably from the MoE model, or from a different measurement. Re-measure both models separately.

## Step 0: measure first (cheap, do this before writing any code)

1. `llama-bench -m <model> -p 512,2048 -n 0 -ub 512,1024,2048 -fa 1 -sm layer` at depth 0, then add `-d 8192,65536`. If prefill is fine at depth 0 and collapses at depth, attention or the KV cache is the bottleneck. If it is slow at depth 0, look at GDN or the matmuls.
2. `nsys profile --stats=true llama-bench ... -p 2048 -n 0` and rank kernels by time. Expected suspects, in order: `gated_delta_net_cuda`, `flash_attn_tile`, `dequantize_*` / cublas `hgemm`, `ssm_conv`.
3. Check that nothing runs on the CPU: run with `GGML_SCHED_DEBUG=2` / `-v` and look for ops assigned to CPU (unsupported op or type on CUDA, for example an exotic KV type or a GDN shape). One CPU op per layer kills prefill. Also confirm `-ngl 99` and no `-ot`/`--cpu-moe` for the dense model.
4. Try `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` vs `f32` (`ggml-cuda.cu:1715`) and `-DGGML_CUDA_FORCE_MMQ=ON` vs default. Both are A/B knobs that need no code changes.

## Idea 1 (top suspect): Gated DeltaNet prefill runs a per-token sequential kernel on Pascal

- The `qwen35` linear-attention layers build `ggml_gated_delta_net` because `cparams.fused_gdn_ch = true` is hardcoded (`src/llama-context.cpp:326`, `src/models/delta-net-base.cpp:439`).
- On CUDA, the chunked/MMA kernel (`gated-delta-net-mma.cu:386 select_gdn_mma_path`) is enabled only on Ampere GA10x, Ada and Blackwell. **P100 falls back to `gdn_path::ar`**: the autoregressive kernel in `gated_delta_net.cu:75`, which loops `for t in n_tokens` serially. Its grid is only H x n_seqs blocks (for example 32-64 blocks on 56 SMs), so prefill gets almost no token parallelism. That would explain why decode is fine while prefill is terrible.
- Try A (no kernel work): set `fused_gdn_ch = false` (add an env var or CLI flag) so the graph uses `build_delta_net_chunking` (`delta-net-base.cpp:16`). That path is made of ordinary matmul and elementwise ops that go through cuBLAS FP16. Benchmark it at `-ub 512/1024/2048`.
- Try B: port the chunked GDN schedule to a non-MMA FP16 variant for sm_60, using `half2` FMAs in shared-memory tiles (the same structure as `gdn_single`, without `mma.sync`). Chunk size 64; WY/UT transform via small triangular solves.
- Try C: even in the AR kernel, the state update `S += k (v - S^T k)^T` is rank-1 per token. Batching the q/k/v loads for many tokens into shared memory (instead of `k_t[i]` global loads per token) and using `half2` math on the state may give 2-3x. Note that the FP32 state has accuracy implications.

## Idea 2: dense matmuls on sm_60

- `ggml_cuda_should_use_mmq` (`mmq.cu:443`): without DP4A, MMQ is used only for MoE (`n_experts > 0`). Dense Q4_K therefore goes through dequant to an FP16 buffer, then cublasGemmEx (CUBLAS_COMPUTE_16F). That is reasonable, but:
  - The dequant to FP16 temp is redone per ubatch. **Bigger `-ub` (1024-4096) amortizes it**; VRAM permitting, try `-ub 2048 -b 4096`.
  - FP16 accumulate (`COMPUTE_16F`) is used on Pascal because `prefer_f32_output` is false (`ggml-cuda.cu:1582`). Benchmark the FP16-out vs FP32-out variants; P100 HGEMM in cuBLAS is mature, so expect the GEMMs themselves to be near peak.
- An "MMQ-without-DP4A" FP16 kernel (dequant Q4_K tiles to `half2` in shared memory, then `hfma2`) could beat dequant+cuBLAS by skipping the full-size FP16 temp buffer. The upstream TODO at `mmq.cu:445` hints that nobody has benchmarked pre-DP4A cards. Medium effort; only worth it if the profile shows `dequantize_block_*` taking a significant share.
- Quant choice: on sm_60, Q4_0 / Q8_0 dequantize much cheaper than K-quants and IQ types (IQ needs table lookups). If dequant shows up in the profile, compare Q4_0 or Q4_K_S vs Q4_K_M for prefill.

## Idea 3: flash attention at 262k context

- Pascal gets `BEST_FATTN_KERNEL_TILE` for prefill (no MMA path, `fattn.cu:637-691`). It uses FP16 when `fast_fp16_available` (`fattn-tile.cuh:322`); verify that this compiled for sm_60 (`-DCMAKE_CUDA_ARCHITECTURES=60`, and **do not build for 61** only).
- A quantized KV (`-ctk q8_0 -ctv q8_0`) gets converted to F16 for the tile kernel on every ubatch, which costs O(context) per ubatch. At large depth this dominates. Test F16 KV vs q8_0 at depth 64k. The hybrid model has only about 1/4 full-attention layers, so F16 KV may fit.
- Tune tile config for sm_60 (smem 64KB/SM, 48KB/block): `fattn-tile.cuh` configs are tuned on newer GPUs. Try different `ncols` and `nwarps` for head dim 256 (Qwen3.5/Next full-attn uses D=256, which is heavy on registers for Pascal).

## Idea 4: multi-GPU split

- `-sm layer` (default) serializes the GPUs during prefill. Enable pipeline parallelism: `-b` larger than `-ub` (for example `-b 4096 -ub 1024`), plus a build with `GGML_SCHED_MAX_COPIES=4` (default), so both P100s work on different ubatches at the same time. Confirm with nsys that the two devices overlap.
- Try `-sm row` / `-sm tensor` (`include/llama.h:203`). Without NVLink the P100 PCIe P2P on an R740 (both in CPU1 x16 risers?) may be OK for prefill, where the compute-to-transfer ratio is high. Check `nvidia-smi topo -m`, and confirm P2P is enabled (`GGML_CUDA_P2P`, `CUDA_VISIBLE_DEVICES` order).
- Put both cards on the same CPU socket (R740 riser layout) to avoid the UPI hop.

## Idea 5: MoE ("Next/Flash") specifics

- MMQ is used for MoE on sm_60 with the software `ggml_cuda_dp4a` fallback (`common.cuh:712`), which does 4 integer multiply-adds per call. A `half2` path (dequantize to FP16 inside MMQ tiles) might be faster on P100 than emulated int8 dot products. Benchmark MMQ vs `GGML_CUDA_FORCE_CUBLAS` for MoE prefill.
- With 512GB RAM, offloading experts to CPU (`--cpu-moe` / `-ot exps=CPU`) makes prefill CPU-bound. llama.cpp can move expert weights to the GPU for big batches (op offload, `--no-op-offload` toggles it). Make sure it is on and that `-ub` is large (>= 2048) so the PCIe 3.0 x16 transfer (about 12 GB/s) is amortized.
- R740 CPU-side: set `--numa distribute` or `numactl --interleave=all`, threads = physical cores, check that AVX-512 is enabled in the build (Skylake/Cascade Lake Xeon).

## Idea 6: misc low-effort knobs

- Build: `-DGGML_CUDA_F16=ON` (FP16 intermediate paths), `-DCMAKE_CUDA_ARCHITECTURES=60`, latest CUDA that still supports sm_60 (CUDA 12.x; CUDA 13 dropped Pascal).
- `ssm_conv` (`ggml-cuda/ssm-conv.cu`) for conv_dim of about 10k with d_conv=4 should be cheap. Confirm it is in the profile and is not a hidden CPU fallback.
- Prompt caching (`--cache-reuse`, slot save/restore) so long-context prefill is not redone. With hybrid models, the recurrent state checkpoints (`K` snapshot slots in GDN) matter for reuse.

## Priority order

1. Step 0 profile → 2. GDN chunked path (Idea 1A, a flag flip) → 3. larger ubatch + pipeline parallel (Ideas 2 and 4) → 4. KV type and FA tile tuning (Idea 3) → 5. custom sm_60 kernels (Idea 1B/C, Idea 2 MMQ-FP16).
