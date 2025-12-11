#!/usr/bin/env python
"""
Quantization Benchmarking with LLaMA

This script runs a small, reproducible benchmark comparing:
- FP16 baseline
- 4-bit bitsandbytes (bnb-4bit)
- 4-bit GPTQ (auto-gptq)

Metrics:
- TTFT proxy (ms): time to generate the first token.
- Latency per prompt (ms).
- Throughput (tokens / second).
- Perplexity on a small slice of WikiText-2.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    HAS_AUTOGPTQ = True
except ImportError:
    HAS_AUTOGPTQ = False


@dataclass
class BenchmarkResult:
    model_id: str
    backend: str
    bits: int
    device: str
    max_new_tokens: int
    max_samples: int
    avg_ttft_ms: float
    avg_latency_ms: float
    throughput_tokens_per_s: float
    perplexity: Optional[float]



# Model loading / quantization



def load_model_and_tokenizer(
    model_id: str,
    backend: str,
    bits: int,
    device: str,
    quantized_model_dir: Optional[str] = None,
    do_quantize: bool = False,
    num_calib_samples: int = 8,
) -> Tuple[Any, Any]:
    """
    Load a LLaMA model and tokenizer with the specified backend.

    backend:
      - "fp16"   : full / half precision baseline
      - "bnb4"   : 4-bit bitsandbytes
      - "gptq4"  : 4-bit GPTQ via auto-gptq
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Ensure we have a pad token for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if backend == "fp16":
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            # CPU fallback
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model = model.to(device)
        return model, tokenizer

    if backend == "bnb4":
        # 4-bit NF4 config as recommended in HF docs
        # https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            if device == "cuda"
            else torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            model = model.to(device)
        return model, tokenizer

    if backend == "gptq4":
        if not HAS_AUTOGPTQ:
            raise RuntimeError(
                "Backend gptq4 requires auto-gptq. "
                "Install it with `pip install auto-gptq` (see README)."
            )

        # If a quantized directory exists and we aren't explicitly re-quantizing,
        # load from it. Otherwise quantize from the full-precision model.
        if (
            quantized_model_dir
            and os.path.isdir(quantized_model_dir)
            and not do_quantize
        ):
            device_str = "cuda:0" if device == "cuda" else "cpu"
            model = AutoGPTQForCausalLM.from_quantized(
                quantized_model_dir,
                device=device_str,
            )
            return model, tokenizer

        # Simple one-shot quantization path based on the auto-gptq quick tour
        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=128,
            desc_act=False,
        )

        model = AutoGPTQForCausalLM.from_pretrained(
            model_id,
            quantize_config,
        )

        # Use a few training samples from WikiText-2 as calibration
        calib_texts = get_calibration_texts(num_calib_samples)
        calib_examples = [tokenizer(t) for t in calib_texts]
        model.quantize(calib_examples)

        if quantized_model_dir:
            os.makedirs(quantized_model_dir, exist_ok=True)
            model.save_quantized(quantized_model_dir, use_safetensors=True)

        device_str = "cuda:0" if device == "cuda" else "cpu"
        model = AutoGPTQForCausalLM.from_quantized(
            quantized_model_dir or model_id, device=device_str
        )

        return model, tokenizer

    raise ValueError(f"Unknown backend: {backend}")


# -----------------------------
# Data / prompts
# -----------------------------


def get_wikitext_prompts(
    max_samples: int,
    split: str = "validation",
    min_chars: int = 64,
) -> List[str]:
    """
    Load a small slice of WikiText-2-raw-v1 as prompts.

    Use the 'Salesforce/wikitext' dataset with the 'wikitext-2-raw-v1' subset.
    """
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split
    )
    texts = [row["text"].strip() for row in ds if row["text"].strip()]
    # Filter out very short lines
    texts = [t for t in texts if len(t) >= min_chars]
    return texts[:max_samples]


def get_calibration_texts(num_samples: int) -> List[str]:
    """
    Very small calibration set, reusing WikiText-2 validation.

    For GPTQ this should be just enough to make the example coherent.
    """
    return get_wikitext_prompts(max_samples=num_samples, split="validation")


# Metrics: TTFT, latency, throughput, perplexity


def _sync_if_cuda(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_ttft_ms(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    warmup: int = 2,
    runs: int = 5,
) -> float:
    """
    Approximate TTFT as the time to generate 1 new token.

    We do a couple of warmup runs and then average over 'runs' prompts.
    """
    model.eval()
    times_ms: List[float] = []

    # Use a small subset of prompts
    subset = prompts[: warmup + runs]

    for i, prompt in enumerate(subset):
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        _sync_if_cuda(device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                **enc,
                max_new_tokens=1,
                do_sample=False,
            )
        _sync_if_cuda(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if i >= warmup:
            times_ms.append(elapsed_ms)

    return float(sum(times_ms) / max(len(times_ms), 1))


def measure_latency_and_throughput(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int = 32,
    warmup: int = 2,
) -> Tuple[float, float]:
    """
    Measure average latency (ms) and tokens/s over a set of prompts.

    We generate up to max_new_tokens for each prompt, ignore warmup runs,
    and compute:
      - avg latency per prompt (ms)
      - throughput = total new tokens / total wall-clock seconds
    """
    model.eval()
    latencies_ms: List[float] = []
    total_new_tokens = 0

    for i, prompt in enumerate(prompts):
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        _sync_if_cuda(device)
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        _sync_if_cuda(device)
        elapsed_sec = time.perf_counter() - start

        # new tokens = generated length - input length
        gen_len = out.shape[-1]
        in_len = enc["input_ids"].shape[-1]
        new_tokens = max(gen_len - in_len, 0)

        if i >= warmup:
            latencies_ms.append(elapsed_sec * 1000.0)
            total_new_tokens += new_tokens

    if not latencies_ms:
        return 0.0, 0.0

    avg_latency_ms = float(sum(latencies_ms) / len(latencies_ms))
    total_time_sec = sum(latencies_ms) / 1000.0
    throughput = (
        float(total_new_tokens) / total_time_sec if total_time_sec > 0 else 0.0
    )
    return avg_latency_ms, throughput


def estimate_perplexity(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int = 512,
) -> float:
    """
    Estimate perplexity using teacher-forcing cross-entropy.

    For each text:
      loss = model(input_ids, labels=input_ids).loss
    Perplexity = exp(mean(loss)).
    """
    model.eval()
    losses = []

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        losses.append(out.loss.item())

    if not losses:
        return float("nan")

    mean_loss = sum(losses) / len(losses)
    ppl = float(torch.exp(torch.tensor(mean_loss)).item())
    return ppl



# Main runner


def run_benchmark(
    model_id: str,
    backend: str,
    bits: int,
    max_samples: int,
    max_new_tokens: int,
    output_dir: str,
    quantized_model_dir: Optional[str] = None,
    do_quantize: bool = False,
) -> BenchmarkResult:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        backend=backend,
        bits=bits,
        device=device,
        quantized_model_dir=quantized_model_dir,
        do_quantize=do_quantize,
    )

    # Prompts for latency / TTFT
    prompts = get_wikitext_prompts(
        max_samples=max_samples, split="validation"
    )

    # 1) TTFT proxy
    ttft_ms = measure_ttft_ms(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
    )
    print(f"[RESULT] TTFT proxy: {ttft_ms:.2f} ms")

    # 2) Latency + throughput
    avg_latency_ms, tps = measure_latency_and_throughput(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    print(f"[RESULT] Avg latency: {avg_latency_ms:.2f} ms")
    print(f"[RESULT] Throughput: {tps:.2f} tokens/s")

    # 3) Perplexity (reuse same texts)
    ppl = estimate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=prompts,
        device=device,
    )
    print(f"[RESULT] Perplexity: {ppl:.3f}")

    result = BenchmarkResult(
        model_id=model_id,
        backend=backend,
        bits=bits,
        device=device,
        max_new_tokens=max_new_tokens,
        max_samples=max_samples,
        avg_ttft_ms=ttft_ms,
        avg_latency_ms=avg_latency_ms,
        throughput_tokens_per_s=tps,
        perplexity=ppl,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"results_{backend}_{bits}bit.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"[INFO] Saved results to {out_path}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantization benchmarking with LLaMA."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help=(
            "Hugging Face model ID (must be a LLaMA-class causal LM). "
            "You must have accepted the model license on Hugging Face."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["fp16", "bnb4", "gptq4"],
        default="fp16",
        help="Quantization backend.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Target weight bit-width (used for GPTQ).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=16,
        help="Number of WikiText prompts to benchmark on.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max new tokens to generate for throughput measurement.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Relative path to directory where JSON results are stored.",
    )
    parser.add_argument(
        "--quantized-model-dir",
        type=str,
        default="quantized_model",
        help=(
            "Directory for GPTQ-quantized weights (for backend=gptq4). "
            "Will be created if do-quantize is set."
        ),
    )
    parser.add_argument(
        "--do-quantize",
        action="store_true",
        help=(
            "If set and backend=gptq4, quantize from full-precision weights "
            "into quantized-model-dir before benchmarking."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        model_id=args.model_id,
        backend=args.backend,
        bits=args.bits,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        quantized_model_dir=args.quantized_model_dir,
        do_quantize=args.do_quantize,
    )


if __name__ == "__main__":
    main()
