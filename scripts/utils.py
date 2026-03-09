"""Common utilities for Qwen3.5 llama.cpp benchmarks."""
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLAMA_CLI = str(PROJECT_ROOT / "llama.cpp/build/bin/llama-cli")
LLAMA_MTMD = str(PROJECT_ROOT / "llama.cpp/build/bin/llama-mtmd-cli")
LLAMA_BIN_DIR = str(PROJECT_ROOT / "llama.cpp/build/bin")
IMAGE_PATH = str(PROJECT_ROOT / "data/test.jpg")
OUT_DIR = str(PROJECT_ROOT / "results")
NGL = "99"
CHAT_TEMPLATE = (
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def parse_perf(text):
    """Parse performance stats from llama.cpp stdout+stderr.
    Supports llama_perf_context_print, single-turn summary, and VL metrics.
    """
    r = {}

    m = re.search(r"load time\s*=\s*([\d.]+)\s*ms", text)
    if m:
        r["load_time_ms"] = float(m.group(1))

    m = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
        r"\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)", text)
    if m:
        r["prompt_eval_time_ms"] = float(m.group(1))
        r["prompt_tokens"] = int(m.group(2))
        r["prompt_ms_per_token"] = float(m.group(3))
        r["prompt_tps"] = float(m.group(4))

    m = re.search(
        r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs?"
        r"\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)", text)
    if m:
        r["eval_time_ms"] = float(m.group(1))
        r["eval_tokens"] = int(m.group(2))
        r["eval_ms_per_token"] = float(m.group(3))
        r["eval_tps"] = float(m.group(4))

    m = re.search(r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", text)
    if m:
        r["total_time_ms"] = float(m.group(1))
        r["total_tokens"] = int(m.group(2))

    # llama-cli --single-turn summary
    if "eval_tps" not in r:
        m = re.search(
            r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]", text)
        if m:
            r["prompt_tps"] = float(m.group(1))
            r["eval_tps"] = float(m.group(2))

    # VRAM: standard format
    m = re.search(r"CUDA0 model buffer size\s*=\s*([\d.]+)\s*MiB", text)
    if m:
        r["cuda_model_mib"] = float(m.group(1))
    # VRAM: memory breakdown format (single-turn)
    if "cuda_model_mib" not in r:
        m = re.search(r"CUDA0\s*\([^)]*\)\s*\|\s*\d+\s*=\s*\d+\s*\+\s*\((\d+)", text)
        if m:
            r["cuda_model_mib"] = float(m.group(1))

    m = re.search(r"CPU_Mapped model buffer size\s*=\s*([\d.]+)\s*MiB", text)
    if m:
        r["cpu_model_mib"] = float(m.group(1))

    # VL metrics
    m = re.search(r"image slice encoded in (\d+) ms", text)
    if m:
        r["image_encode_ms"] = int(m.group(1))
    m = re.search(r"image decoded.*in (\d+) ms", text)
    if m:
        r["image_decode_ms"] = int(m.group(1))
    m = re.search(r"n_tokens_batch = (\d+)", text)
    if m:
        r["image_tokens"] = int(m.group(1))

    return r


def extract_response(text):
    """Extract model response text from llama.cpp output."""
    m = re.search(
        r'</think>\s*\n\s*\n(.*?)(?:llama_memory_breakdown|llama_perf|\[\s*Prompt:|$)',
        text, re.DOTALL)
    if m:
        resp = m.group(1).strip()
        resp = re.sub(r'\x08.', '', resp)
        resp = re.sub(r'^[|/\\-]+\s*', '', resp)
        return resp
    return ""


def run_inference(model_path, prompt, max_tokens, *, mmproj_path=None,
                  image_path=None, greedy=True, image_max_tokens=None,
                  extra_args=None):
    """Run a single llama.cpp inference and return parsed results."""
    formatted = CHAT_TEMPLATE.format(prompt=prompt)

    if image_path and mmproj_path:
        cmd = [LLAMA_MTMD, "-m", str(model_path), "--mmproj", str(mmproj_path)]
        cmd += ["--image", str(image_path)]
        cmd += ["-p", formatted]
        cmd += ["-ngl", NGL, "-c", "4096", "-n", str(max_tokens)]
        if image_max_tokens:
            cmd += ["--image-max-tokens", str(image_max_tokens)]
    else:
        cmd = [LLAMA_CLI, "-m", str(model_path)]
        cmd += ["-ngl", NGL, "-c", "4096", "-n", str(max_tokens)]
        cmd += ["-p", formatted]
        cmd += ["--single-turn"]

    if greedy:
        cmd += ["--temp", "0"]
    else:
        cmd += ["--temp", "1.0", "--top-k", "20", "--top-p", "1.0"]

    if extra_args:
        cmd += extra_args

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_BIN_DIR + ":" + env.get("LD_LIBRARY_PATH", "")

    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    wall_time = time.perf_counter() - start

    combined = proc.stdout + "\n" + proc.stderr
    perf = parse_perf(combined)
    perf["wall_time_sec"] = round(wall_time, 3)

    response = extract_response(proc.stdout if image_path else combined)
    perf["response_preview"] = response[:300] if response else ""

    if proc.returncode != 0:
        perf["error"] = proc.stderr[-500:] if proc.stderr else "unknown error"

    return perf


def run_benchmark(model_path, name, prompt, max_tokens, *, mmproj_path=None,
                  image_path=None, runs=3, warmup=1, greedy=True,
                  image_max_tokens=None, sleep_between_vl=8):
    """Run multiple inference iterations with warmup and return summary."""
    print(f"\n{'='*60}")
    print(f"[{name}] runs={runs}, warmup={warmup}, max_new_tokens={max_tokens}, greedy={greedy}")
    print(f"  prompt: {prompt[:80]}...")
    if image_path:
        print(f"  image: {image_path}")

    all_results = []
    for i in range(warmup + runs):
        if i > 0 and image_path:
            time.sleep(sleep_between_vl)
        phase = "warmup" if i < warmup else "measured"
        result = run_inference(model_path, prompt, max_tokens,
                               mmproj_path=mmproj_path, image_path=image_path,
                               greedy=greedy, image_max_tokens=image_max_tokens)
        result["phase"] = phase
        result["run"] = i + 1

        tag = "  [W]" if phase == "warmup" else "  [M]"
        tps = result.get("eval_tps", 0)
        tokens = result.get("eval_tokens", 0)
        err = " ERROR" if "error" in result else ""
        parts = [f"run={i+1}"]
        if result.get("image_tokens"):
            parts.append(f"img_tok={result['image_tokens']}")
        if result.get("image_encode_ms"):
            parts.append(f"encode={result['image_encode_ms']}ms")
        parts += [f"tokens={tokens}", f"tps={tps:.2f}",
                  f"wall={result['wall_time_sec']:.2f}s{err}"]
        print(f"{tag} {' '.join(parts)}")

        resp = result.get("response_preview", "")[:60]
        if resp:
            print(f"      => {resp}...")
        all_results.append(result)

    measured = [r for r in all_results if r["phase"] == "measured"]

    def avg(key):
        vals = [r[key] for r in measured if r.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    def std(key):
        vals = [r[key] for r in measured if r.get(key) is not None]
        return round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0

    summary = {
        "name": name,
        "prompt": prompt,
        "image_path": image_path,
        "max_new_tokens": max_tokens,
        "greedy": greedy,
        "runs": runs,
        "warmup": warmup,
        "avg_eval_tps": avg("eval_tps") or 0,
        "std_eval_tps": std("eval_tps"),
        "avg_tokens": avg("eval_tokens") or 0,
        "avg_wall_sec": avg("wall_time_sec") or 0,
        "cuda_model_mib": measured[0].get("cuda_model_mib") if measured else None,
        "avg_prompt_tps": avg("prompt_tps"),
        "avg_image_tokens": avg("image_tokens"),
        "avg_image_encode_ms": avg("image_encode_ms"),
        "avg_image_decode_ms": avg("image_decode_ms"),
        "avg_prompt_eval_ms": avg("prompt_eval_time_ms"),
        "avg_prompt_tokens": avg("prompt_tokens"),
        "avg_total_ms": avg("total_time_ms"),
        "all_runs": all_results,
    }
    print(f"  => avg_tps={summary['avg_eval_tps']:.2f} avg_wall={summary['avg_wall_sec']:.2f}s")
    return summary
