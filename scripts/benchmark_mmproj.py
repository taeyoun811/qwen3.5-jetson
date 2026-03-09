"""mmproj optimization benchmark: quantization (F16 vs Q8_0) x image token limit.
Measures image encoding time, prompt processing, and generation speed.

Usage:
    python3 benchmark_mmproj.py          # 2B (default)
    python3 benchmark_mmproj.py 4B       # 4B
"""
import json
import os
import sys
import time

from utils import PROJECT_ROOT, IMAGE_PATH, OUT_DIR, NGL, run_benchmark

MODEL_PRESETS = {
    "2B": {
        "model_path": str(PROJECT_ROOT / "models/Qwen3.5-2B-Q4_K_M.gguf"),
        "mmproj_f16": str(PROJECT_ROOT / "models/mmproj-2B-F16.gguf"),
        "mmproj_q8":  str(PROJECT_ROOT / "models/mmproj-2B-Q8_0.gguf"),
        "out_file": "mmproj_optimization_2b.json",
    },
    "4B": {
        "model_path": str(PROJECT_ROOT / "models/Qwen3.5-4B-Q4_K_M.gguf"),
        "mmproj_f16": str(PROJECT_ROOT / "models/mmproj-F16.gguf"),
        "mmproj_q8":  str(PROJECT_ROOT / "models/mmproj-4B-Q8_0.gguf"),
        "out_file": "mmproj_optimization_4b.json",
    },
}


def get_configs(preset):
    return [
        {"name": "F16",            "mmproj": preset["mmproj_f16"], "image_max_tokens": None},
        {"name": "F16_maxtok256",  "mmproj": preset["mmproj_f16"], "image_max_tokens": 256},
        {"name": "Q8_0",           "mmproj": preset["mmproj_q8"],  "image_max_tokens": None},
        {"name": "Q8_0_maxtok256", "mmproj": preset["mmproj_q8"],  "image_max_tokens": 256},
    ]


VL_TESTS = [
    {
        "name": "vl_caption_64",
        "prompt": "Describe this scene in one concise sentence.",
        "max_tokens": 64,
        "greedy": True,
    },
    {
        "name": "vl_detail_256",
        "prompt": "Describe this image in detail. Include information about objects, people, actions, and the environment.",
        "max_tokens": 256,
        "greedy": False,
    },
]


def main():
    model_size = sys.argv[1] if len(sys.argv) > 1 else "2B"
    if model_size not in MODEL_PRESETS:
        print(f"Unknown model size: {model_size}. Choose from: {list(MODEL_PRESETS.keys())}")
        sys.exit(1)

    preset = MODEL_PRESETS[model_size]
    model_path = preset["model_path"]
    configs = get_configs(preset)

    print(f"Model: {model_path} ({model_size})")
    print(f"Image: {IMAGE_PATH}")
    print(f"Backend: llama.cpp (CUDA, ngl={NGL})")

    all_benchmarks = []

    for config in configs:
        maxtok_str = f", image_max_tokens={config['image_max_tokens']}" if config["image_max_tokens"] else ""
        print(f"\n{'#'*60}")
        print(f"# Config: {config['name']} (mmproj={os.path.basename(config['mmproj'])}{maxtok_str})")
        print(f"{'#'*60}")

        for test in VL_TESTS:
            bench_name = f"{config['name']}_{test['name']}"
            result = run_benchmark(
                model_path, bench_name, test["prompt"], test["max_tokens"],
                mmproj_path=config["mmproj"], image_path=IMAGE_PATH,
                greedy=test["greedy"], runs=3, warmup=1,
                image_max_tokens=config["image_max_tokens"],
                sleep_between_vl=5,
            )
            result["config"] = config["name"]
            result["image_max_tokens"] = config["image_max_tokens"]
            result["test"] = test["name"]
            all_benchmarks.append(result)

        print(f"\nSleeping 10s before next config...")
        time.sleep(10)

    # Save
    full_result = {
        "model_path": model_path,
        "image_path": IMAGE_PATH,
        "experiment": "mmproj_optimization",
        "model_size": model_size,
        "configs": [c["name"] for c in configs],
        "benchmarks": all_benchmarks,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, preset["out_file"])
    with open(out_path, "w") as f:
        json.dump(full_result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Summary table
    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"\n{'Config':<18} {'Test':<18} {'ImgTok':>6} {'Encode':>8} {'Prompt':>8} {'EvalTPS':>8} {'Wall(s)':>8}")
    print("-" * 76)
    for b in all_benchmarks:
        img_tok = f"{b['avg_image_tokens']:.0f}" if b.get('avg_image_tokens') else "-"
        enc = f"{b['avg_image_encode_ms']:.0f}" if b.get('avg_image_encode_ms') else "-"
        prompt_ms = f"{b['avg_prompt_eval_ms']:.0f}" if b.get('avg_prompt_eval_ms') else "-"
        print(f"{b.get('config',''):<18} {b.get('test',''):<18} {img_tok:>6} {enc:>7}ms {prompt_ms:>7}ms {b['avg_eval_tps']:>8.2f} {b['avg_wall_sec']:>8.2f}")


if __name__ == "__main__":
    main()
