"""Qwen3.5 2B/9B Q4_K_M benchmark via llama.cpp.
Text: llama-cli --single-turn | VL: llama-mtmd-cli

Usage:
    python3 benchmark_all_sizes.py
"""
import json
import os
import time

from utils import PROJECT_ROOT, IMAGE_PATH, OUT_DIR, NGL, run_benchmark

MODELS = [
    {
        "name": "2B",
        "model_path": str(PROJECT_ROOT / "models/Qwen3.5-2B-Q4_K_M.gguf"),
        "mmproj_path": str(PROJECT_ROOT / "models/mmproj-2B-F16.gguf"),
        "out_file": "qwen35_2b_gguf_q4km_benchmark.json",
    },
    {
        "name": "9B",
        "model_path": str(PROJECT_ROOT / "models/Qwen3.5-9B-Q4_K_M.gguf"),
        "mmproj_path": str(PROJECT_ROOT / "models/mmproj-9B-F16.gguf"),
        "out_file": "qwen35_9b_gguf_q4km_benchmark.json",
    },
]


def run_full_suite(model_cfg):
    model_path = model_cfg["model_path"]
    mmproj_path = model_cfg["mmproj_path"]
    model_name = model_cfg["name"]

    print(f"\n{'#'*60}")
    print(f"# Qwen3.5-{model_name} Q4_K_M Benchmark")
    print(f"# Model: {model_path}")
    print(f"# mmproj: {mmproj_path}")
    print(f"{'#'*60}")

    benchmarks = []

    # Text benchmarks
    benchmarks.append(run_benchmark(model_path,
        name="text_short_32",
        prompt="Give me a short introduction to large language models.",
        max_tokens=32, runs=5, warmup=2, greedy=False,
    ))
    benchmarks.append(run_benchmark(model_path,
        name="text_long_256",
        prompt="Explain the key differences between transformer and recurrent neural network architectures in detail.",
        max_tokens=256, runs=3, warmup=1, greedy=False,
    ))
    benchmarks.append(run_benchmark(model_path,
        name="text_greedy_64",
        prompt="Give me a short introduction to large language models.",
        max_tokens=64, runs=3, warmup=1, greedy=True,
    ))

    # VL benchmarks
    benchmarks.append(run_benchmark(model_path,
        name="vl_caption_64",
        prompt="Describe this scene in one concise sentence.",
        max_tokens=64, runs=3, warmup=1, greedy=True,
        mmproj_path=mmproj_path, image_path=IMAGE_PATH,
    ))
    benchmarks.append(run_benchmark(model_path,
        name="vl_detail_256",
        prompt="Describe this image in detail. Include information about objects, people, actions, and the environment.",
        max_tokens=256, runs=3, warmup=1, greedy=False,
        mmproj_path=mmproj_path, image_path=IMAGE_PATH,
    ))

    # Quality benchmarks
    quality_prompts = [
        ("quality_intro", "Give me a short introduction to large language models.", 64),
        ("quality_korean", "Answer in Korean with exactly three bullet points: what is overfitting?", 96),
        ("quality_json", "Return only valid JSON with keys answer and confidence for the question: What does CPU stand for?", 64),
        ("quality_2sent", "Summarize why caching is useful in web services in exactly two sentences.", 80),
    ]
    for qname, qprompt, qmax in quality_prompts:
        benchmarks.append(run_benchmark(model_path,
            name=qname, prompt=qprompt, max_tokens=qmax,
            runs=1, warmup=0, greedy=False,
        ))

    # Save
    full_result = {
        "model_path": model_path,
        "mmproj_path": mmproj_path,
        "backend": "llama.cpp",
        "quantization": "Q4_K_M",
        "ngl": int(NGL),
        "benchmarks": benchmarks,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, model_cfg["out_file"])
    with open(out_path, "w") as f:
        json.dump(full_result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Summary table
    print(f"\n{'='*60}")
    print(f"Qwen3.5-{model_name} Results saved to {out_path}")
    print(f"{'Benchmark':<20} {'Tokens':>6} {'Wall(s)':>8} {'EvalTPS':>8} {'PromptTPS':>10}")
    print("-" * 54)
    for b in benchmarks:
        ptps = f"{b['avg_prompt_tps']:.1f}" if b.get('avg_prompt_tps') else "-"
        print(f"{b['name']:<20} {b['avg_tokens']:>6.0f} {b['avg_wall_sec']:>8.2f} {b['avg_eval_tps']:>8.2f} {ptps:>10}")

    return full_result


def main():
    for model_cfg in MODELS:
        run_full_suite(model_cfg)
        print(f"\nSleeping 30s before next model to let GPU fully release...")
        time.sleep(30)

    print(f"\n{'#'*60}")
    print("# ALL DONE")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
