# Qwen3.5 on Jetson Orin NX — llama.cpp

Qwen3.5 (2B/4B/9B, Q4_K_M) inference setup using llama.cpp with CUDA backend.

## Hardware

- NVIDIA Jetson Orin NX (16GB unified memory, sm_87)
- CUDA 12.6, aarch64 Linux

## Quick Start

### 1. Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && git checkout 2850bc6
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 2. Download Models

GGUF models from [Unsloth](https://huggingface.co/unsloth) on HuggingFace:

```bash
# Pick the model size you need
# 2B (fastest, ~1.2GB)
huggingface-cli download unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q4_K_M.gguf --local-dir models/
huggingface-cli download unsloth/Qwen3.5-2B-GGUF mmproj-2B-F16.gguf --local-dir models/

# 4B (balanced, ~2.6GB)
huggingface-cli download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir models/
huggingface-cli download unsloth/Qwen3.5-4B-GGUF mmproj-F16.gguf --local-dir models/

# 9B (best quality, ~5.3GB)
huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir models/
huggingface-cli download unsloth/Qwen3.5-9B-GGUF mmproj-9B-F16.gguf --local-dir models/
```

To convert mmproj to Q8_0 (optional, for VL encoding optimization):

```bash
cd llama.cpp && python3 convert_hf_to_gguf.py --mmproj --outtype q8_0 --remote Qwen/Qwen3.5-2B
# Requires torch and transformers in a venv
```

### 3. Text Inference

```bash
./llama.cpp/build/bin/llama-cli \
  -m models/Qwen3.5-4B-Q4_K_M.gguf \
  -ngl 99 -c 4096 -n 64 \
  -p "<|im_start|>user
Your question here<|im_end|>
<|im_start|>assistant
<think>

</think>

" \
  --single-turn --temp 0
```

### 4. Vision-Language (VL) Inference

```bash
./llama.cpp/build/bin/llama-mtmd-cli \
  -m models/Qwen3.5-4B-Q4_K_M.gguf \
  --mmproj models/mmproj-F16.gguf \
  --image data/test.jpg \
  -ngl 99 -c 4096 -n 256 \
  -p "<|im_start|>user
Describe this image in detail.<|im_end|>
<|im_start|>assistant
<think>

</think>

" \
  --temp 0
```

### 5. Run Benchmarks

```bash
cd scripts/
python3 benchmark_gguf.py          # 4B
python3 benchmark_all_sizes.py     # 2B + 9B
python3 benchmark_mmproj.py        # mmproj optimization (2B)
python3 benchmark_mmproj.py 4B     # mmproj optimization (4B)
```

## Benchmark Results

### Model Size Comparison (llama.cpp, Q4_K_M)

| Model | Text (tok/s) | VL (tok/s) | VRAM | Notes |
|-------|-------------|----------|------|-------|
| **2B** | **23.3** | **24.3** | ~1,771 MiB | Fastest, good quality |
| **4B** | **11.5** | **11.5** | ~2,604 MiB | Balanced choice |
| **9B** | **8.5** | **8.2** | ~5,540 MiB | Best quality, tight on memory for VL |

- 9B VL may OOM intermittently on unified memory (model 5.3GB + mmproj 876MB + compute)
- Expected to run stable in production (without dev tools consuming memory)

### Backend Comparison (4B)

| Backend | Quantization | Text (tok/s) | VL (tok/s) | VRAM |
|---------|-------------|-------------|----------|------|
| **llama.cpp** | **Q4_K_M** | **11.5** | **11.5** | **2,604 MiB** |
| transformers | FP16 | 5.4 | 5.0 | ~8,700 MiB |
| transformers | NF4 | 4.2 | 3.9 | ~5,200 MiB |

llama.cpp is **2-3x faster with 1/3 VRAM** compared to transformers.

### VL Image Encoding Optimization

Combined mmproj quantization (F16→Q8_0) and image token limit (`--image-max-tokens 256`):

**2B model:**

| Config | Image Tokens | Encoding | Prompt Processing | Generation | Encoding Change |
|--------|-------------|----------|-------------------|------------|----------------|
| F16, no limit (baseline) | 405 | 674ms | 1,337ms | 23.8 tok/s | — |
| F16 + maxtok256 | 231 | 421ms | 878ms | 23.9 tok/s | **-37%** |
| Q8_0 | 405 | 554ms | 1,204ms | 23.7 tok/s | **-18%** |
| **Q8_0 + maxtok256** | **231** | **336ms** | **780ms** | **23.8 tok/s** | **-50%** |

**4B model:**

| Config | Image Tokens | Encoding | Prompt Processing | Generation | Encoding Change |
|--------|-------------|----------|-------------------|------------|----------------|
| F16, no limit (baseline) | 405 | 670ms | 2,142ms | 11.3 tok/s | — |
| F16 + maxtok256 | 231 | 423ms | 1,413ms | 11.3 tok/s | **-37%** |
| Q8_0 | 405 | 539ms | 2,017ms | 11.4 tok/s | **-20%** |
| **Q8_0 + maxtok256** | **231** | **338ms** | **1,339ms** | **11.4 tok/s** | **-50%** |

- Encoding time depends on mmproj size, not text model size (2B: 638MB, 4B: 642MB → nearly identical encoding times)
- Generation speed is unaffected by these optimizations (depends only on model size)
- No quality degradation — scene recognition remains accurate at 256 tokens
- Below Q8_0 (e.g. Q4_0) causes vision quality collapse — not recommended

## Notes

- **`-c 4096` is required**: Without it, defaults to 262,144 (model's n_ctx_train), causing OOM or hang
- **Use `llama-cli --single-turn`** for text: `llama-simple` does not support chat templates properly
- **Non-thinking mode is default**: Prompt includes empty `<think>\n\n</think>\n\n` block (official Qwen3.5 method)
- Full GPU offload: `-ngl 99`
- **Standard optimizations tested, no effect** (GPU-bound):
  - Flash Attention, thread count, KV cache quantization, batch size — all ±0.1 tok/s (noise)

## Directory Structure

```
Qwen3_5/
├── llama.cpp/                # Inference engine (commit 2850bc6, CUDA build)
├── models/
│   ├── Qwen3.5-2B-Q4_K_M.gguf    # 2B model (1.2GB)
│   ├── Qwen3.5-4B-Q4_K_M.gguf    # 4B model (2.6GB)
│   ├── Qwen3.5-9B-Q4_K_M.gguf    # 9B model (5.3GB)
│   ├── mmproj-2B-F16.gguf         # 2B vision encoder (638MB)
│   ├── mmproj-2B-Q8_0.gguf        # 2B vision encoder quantized (348MB)
│   ├── mmproj-F16.gguf            # 4B vision encoder (642MB)
│   ├── mmproj-4B-Q8_0.gguf        # 4B vision encoder quantized (350MB)
│   └── mmproj-9B-F16.gguf         # 9B vision encoder (876MB)
├── data/
│   └── test.jpg                    # VL test image
├── results/
│   ├── qwen35_2b_gguf_q4km_benchmark.json
│   ├── qwen35_4b_gguf_q4km_benchmark.json
│   ├── qwen35_9b_gguf_q4km_benchmark.json
│   ├── mmproj_optimization_2b.json
│   ├── mmproj_optimization_4b.json
│   ├── qwen35_4b_full_benchmark.json   # transformers FP16 (for comparison)
│   └── qwen35_4b_nf4_benchmark.json    # transformers NF4 (for comparison)
└── scripts/
    ├── utils.py                  # Shared utilities
    ├── benchmark_gguf.py         # 4B benchmark
    ├── benchmark_all_sizes.py    # 2B/9B benchmark
    └── benchmark_mmproj.py       # mmproj optimization benchmark
```

## TODO

### 1. Speculative Decoding — SSM State Rollback Fix (PR #20075)

- **Status**: Pending (PR submitted 2026-03-03, open)
- **Difficulty**: **Low** — 2 files, +188/-22 lines, our code (commit 2850bc6) shares the same base as the PR
- **Problem**: Qwen3.5 uses DeltaNet (recurrent) architecture → speculative decoding cannot roll back SSM state after draft token rejection, causing severe quality degradation
  - Symptoms: `"LargeLargeLargeLarge..."`, `"achingachingaching..."` repetition loops
  - Speed improves to ~15.5 tok/s but output quality is unusable
- **Fix**: [PR #20075](https://github.com/ggml-org/llama.cpp/pull/20075) — SSM state checkpoint/restore (rolling buffer, depth 8)
  - Verified on Qwen3.5-122B + 0.8B draft: accept rate 63-89%, repetition loops resolved, +15-45% speed improvement

#### PR #20075 Code Changes

Target files: `src/llama-memory-recurrent.cpp` (+184/-22), `src/llama-memory-recurrent.h` (+4)

| Function | Existing Issue | Patch |
|----------|---------------|-------|
| `seq_rm()` | Returns `false` on partial intersection → cannot remove rejected tokens | Finds checkpoint cell at p0-1 and rolls back tail |
| `seq_cp()` | Shares same cell (only inserts seq_id) → mutable SSM state corruption | Physically copies SSM tensors via `ggml_backend_tensor_copy` to empty cell |
| `find_slot()` | `empty_cell.src = orig_cell.src` (wrong reference), no `copy_cell` call | `src = seq_meta.tail` (correct reference) + `copy_cell()` + checkpoint history (max 8 cells per sequence) |
| **New** `copy_cell()` | — | Physical copy of r_l/s_l tensors per layer via `ggml_backend_tensor_copy` |
| **New** `get_cell_count()` | — | Count checkpoint cells for a sequence |

#### Expected Performance in Our Environment

| Metric | Estimate | Rationale |
|--------|----------|-----------|
| Accept rate | **50-70%** | 4B↔0.8B capability gap is larger than 122B↔0.8B, lower draft accuracy |
| --draft-max | 2-4 | Conservative operation needed |
| **Expected speed** | **~14-16 tok/s** | +25-40% from current 11.4 (author's +45% unlikely for us) |
| Additional VRAM | ~600-800 MiB | 0.8B Q4_K_M (~500MB) + checkpoint cells overhead |
| Total VRAM | ~3.4-3.5 GB | Current 2.6GB + draft + checkpoint |

- **Draft model**: Qwen3.5-0.8B-Q4_K_M (~0.6GB additional VRAM)
  - 0.8B is also a VL model, enabling image+text speculative decoding
- **Ref**: [PR #19493](https://github.com/ggml-org/llama.cpp/pull/19493) (2026-02-10) — server-level checkpoint approach, alternative. Explicitly does not support mmproj

### 2. Visual Token Reduction Fine-tuning

- **Status**: Research — based on [ARGUSVLM paper](https://arxiv.org/abs/2603.16987) (2026-03-17, Sony AI)
- **Insight**: Paper showed 64 visual tokens (r=4 pixel-unshuffle) sufficient for general VQA (GQA 55.3 on 256M model). Our binary classification task (bicycle yes/no) likely needs even fewer
- **Problem**: Current `--image-max-tokens 256` truncates at inference time → accuracy drops (exp3: 96%→92% recall). Model was trained for ~400 tokens, doesn't know how to use fewer
- **Approach**: Add learned compression between ViT output and LLM input, then fine-tune
  - **Option A**: Average Pooling (simpler) — pool ViT output to 64 tokens, dimension unchanged, MLP fine-tune only
  - **Option B**: Pixel-Unshuffle (better quality) — r=2 spatial merge, but Qwen3.5 dynamic tiling makes grid handling complex
  - Average Pooling recommended as first attempt for our task
- **Training**: ViT frozen, LLM frozen (or LoRA), only MLP/connector retrained on public VQA data + task-specific data
- **Expected TTFT impact** (2B, Q8_0 mmproj, Jetson):
  - 231 tokens (current): Encoding 336ms + Prefill 780ms = **~1,116ms**
  - 64 tokens (after FT): Encoding 336ms + Prefill ~215ms = **~551ms** (-50%)
- **Caveat**: Requires mmproj GGUF regeneration (`convert_hf_to_gguf.py --mmproj` may need modification for new layer)

### 3. Additional Insights from ARGUSVLM Paper

- **Quantization caution**: Paper found W8A8 **slower** than BF16 on compact VLMs (256M). Activation quantization overhead > matmul savings. Validates our finding that mmproj below Q8_0 breaks vision quality — current Q4_K_M + mmproj Q8_0 is the sweet spot
- **Bottleneck shift**: As model shrinks, non-GPU operations dominate latency. In our llama.cpp setup, the analog is process restart/model reload (not Python CPU preprocessing). Reinforces priority of llama-server mode
- **Profile-first methodology**: Paper's systematic approach (austin for CPU, Nsight for GPU) aligns with our CUDA event profiling in `profile_forward.py`

### 4. Native MTP (Multi-Token Prediction)

- **Status**: Not implemented in llama.cpp (MTP tensors intentionally skipped during GGUF conversion)
- Qwen3.5 HF models contain MTP weights (`mtp_num_hidden_layers: 1`, `mtp.layers.0.*`, `mtp.fc.weight`, etc.)
- `convert_hf_to_gguf.py` line 4497: `if name.startswith("mtp"): return` — tensors dropped
- `llama-arch.cpp` has no NEXTN tensor mapping for Qwen3.5 (only GLM4 family registered)
- Once implemented, GGUF reconversion + MTP activation could further improve speed
- No PR or implementation roadmap currently — long-term goal

## License

MIT
