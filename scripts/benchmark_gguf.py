"""Qwen3.5-4B Q4_K_M benchmark via llama.cpp.
Text: llama-cli --single-turn | VL: llama-mtmd-cli

Usage:
    python3 benchmark_gguf.py
"""
import json
import os

from utils import PROJECT_ROOT, IMAGE_PATH, OUT_DIR, NGL, run_benchmark

MODEL_PATH = str(PROJECT_ROOT / "models/Qwen3.5-4B-Q4_K_M.gguf")
MMPROJ_PATH = str(PROJECT_ROOT / "models/mmproj-F16.gguf")


def main():
    print(f"Model: {MODEL_PATH}")
    print(f"mmproj: {MMPROJ_PATH}")
    print(f"Backend: llama.cpp (CUDA, ngl={NGL})")

    benchmarks = []

    # Text benchmarks
    benchmarks.append(run_benchmark(MODEL_PATH,
        name="text_short_32",
        prompt="Give me a short introduction to large language models.",
        max_tokens=32, runs=5, warmup=2, greedy=False,
    ))
    benchmarks.append(run_benchmark(MODEL_PATH,
        name="text_long_256",
        prompt="Explain the key differences between transformer and recurrent neural network architectures in detail.",
        max_tokens=256, runs=3, warmup=1, greedy=False,
    ))
    benchmarks.append(run_benchmark(MODEL_PATH,
        name="text_greedy_64",
        prompt="Give me a short introduction to large language models.",
        max_tokens=64, runs=3, warmup=1, greedy=True,
    ))

    # VL benchmarks
    benchmarks.append(run_benchmark(MODEL_PATH,
        name="vl_caption_64",
        prompt="Describe this scene in one concise sentence.",
        max_tokens=64, runs=3, warmup=1, greedy=True,
        mmproj_path=MMPROJ_PATH, image_path=IMAGE_PATH,
    ))
    benchmarks.append(run_benchmark(MODEL_PATH,
        name="vl_detail_256",
        prompt="Describe this image in detail. Include information about objects, people, actions, and the environment.",
        max_tokens=256, runs=3, warmup=1, greedy=False,
        mmproj_path=MMPROJ_PATH, image_path=IMAGE_PATH,
    ))

    # Quality benchmarks
    quality_prompts = [
        ("quality_intro", "Give me a short introduction to large language models.", 64),
        ("quality_korean", "Answer in Korean with exactly three bullet points: what is overfitting?", 96),
        ("quality_json", "Return only valid JSON with keys answer and confidence for the question: What does CPU stand for?", 64),
        ("quality_2sent", "Summarize why caching is useful in web services in exactly two sentences.", 80),
    ]
    for qname, qprompt, qmax in quality_prompts:
        benchmarks.append(run_benchmark(MODEL_PATH,
            name=qname, prompt=qprompt, max_tokens=qmax,
            runs=1, warmup=0, greedy=False,
        ))

    # Save
    full_result = {
        "model_path": MODEL_PATH,
        "mmproj_path": MMPROJ_PATH,
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "ngl": int(NGL),
        "benchmarks": benchmarks,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "qwen35_4b_gguf_q4km_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(full_result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Summary table
    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"\n{'Benchmark':<20} {'Tokens':>6} {'Wall(s)':>8} {'EvalTPS':>8} {'PromptTPS':>10} {'CUDA(MiB)':>10}")
    print("-" * 64)
    for b in benchmarks:
        ptps = f"{b['avg_prompt_tps']:.1f}" if b.get('avg_prompt_tps') else "-"
        cuda = f"{b['cuda_model_mib']:.0f}" if b.get('cuda_model_mib') else "-"
        print(f"{b['name']:<20} {b['avg_tokens']:>6.0f} {b['avg_wall_sec']:>8.2f} {b['avg_eval_tps']:>8.2f} {ptps:>10} {cuda:>10}")


if __name__ == "__main__":
    main()
