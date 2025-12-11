# Quantization Benchmarking with LLaMA

This repository contains the code and artifacts for the **Advanced Computer Architecture I** final project:

> **Quantization Benchmarking with LLaMA**  
> Author: Damir Sarsenov  
> Term: Fall 2025

The goal of this project is to prototype a **mini benchmark harness** that compares **precision regimes** for LLaMA‑class models (FP16 vs 4‑bit quantization) along the metrics proposed in the accompanying survey paper:

- **Quality**: Perplexity on WikiText‑2.
- **Performance**:  
  - TTFT proxy (time‑to‑first‑token, ms)  
  - End‑to‑end latency per prompt (ms)  
  - Throughput (tokens / second)
 
  **External libraries**:
  - transformers
  - datasets
  - bitsandbytes
  - auto-gptq
  
- **Memory (conceptual)**: via bit‑width and external docs in the report.
- **Energy (discussed in paper)**: reasoned from memory and performance, not directly measured.

The implementation is intentionally lightweight but **architecture‑driven**, designed to reflect the benchmark protocol described in the paper. :contentReference[oaicite:4]{index=4}  

---

## 1. Repository layout

```text
.
├── Survey-Quantization_LLaMA.pdf               # Project paper
├── Presentation_Quantization_Benchmarking_LLaMA.pptx
├── quant_bench_llama.py                        # Main benchmark script
├── requirements.txt                            # Python dependencies
└── results/
```                             
2.1 Python & GPU

Python 3.10+ recommended.

CUDA‑capable GPU recommended for realistic runtimes, but the code can fall back to CPU (it will just be slow).

2.2 Install PyTorch

Follow the official instructions for your OS/GPU at the PyTorch website:

https://pytorch.org/get-started/locally/

Example (CUDA 12.x, Linux):

pip install --upgrade "torch==2.3.1+cu121" --index-url https://download.pytorch.org/whl/cu121

2.3 Install Python libraries

From the project root:

pip install -r requirements.txt


This brings in:

transformers
 – model and tokenizer loading, generation.

datasets
 – WikiText‑2 dataset.

bitsandbytes
 – 4‑bit NF4 quantization. 
GitHub
+1

auto-gptq
 – GPTQ 4‑bit quantization and efficient kernels. 
PyPI

Note: auto-gptq currently supports Linux/Windows with NVIDIA GPUs and specific CUDA versions; see the project documentation for details. 
PyPI

3. Model access (LLaMA‑class models)

By default the script uses:

meta-llama/Meta-Llama-3-8B as the --model-id. 
Hugging Face

To use this or any other official LLaMA‑3 / 3.1 model you must:

Log in to Hugging Face.

Visit the model card (e.g. meta-llama/Meta-Llama-3-8B).

Read and accept the Meta LLaMA license.

Configure your HF_TOKEN (either via huggingface-cli login or environment variable).

You may also substitute a smaller, open model for faster experiments, e.g. any OPT or GPT‑J variant supported by transformers and AutoGPTQ. 
PyPI
+1

4. How to run the benchmark

All commands below assume you are in the repository root and have installed dependencies.

4.1 FP16 baseline
python quant_bench_llama.py \
  --model-id meta-llama/Meta-Llama-3-8B \
  --backend fp16 \
  --max-samples 8 \
  --max-new-tokens 32 \
  --output-dir results


This will:

Load the FP16 model (GPU if available, otherwise CPU).

Sample 8 validation prompts from Salesforce/wikitext, subset wikitext-2-raw-v1. 
Hugging Face
+1

Report:

TTFT proxy (ms)

Average latency per prompt (ms)

Throughput (tokens/s)

Perplexity

Write a JSON file:

results/results_fp16_4bit.json (the 4bit suffix corresponds to the bits argument; for FP16 you can ignore it).

4.2 4‑bit bitsandbytes (bnb‑4bit)
python quant_bench_llama.py \
  --model-id meta-llama/Meta-Llama-3-8B \
  --backend bnb4 \
  --max-samples 8 \
  --max-new-tokens 32 \
  --output-dir results


This uses BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
and loads the model with 4‑bit weights, matching the QLoRA‑style configuration described in the transformers docs. 
Hugging Face

4.3 4‑bit GPTQ (AutoGPTQ)

This is more heavyweight and requires auto-gptq to be installed and GPU support.

First run (quantize + benchmark):

python quant_bench_llama.py \
  --model-id meta-llama/Meta-Llama-3-8B \
  --backend gptq4 \
  --bits 4 \
  --max-samples 8 \
  --max-new-tokens 32 \
  --output-dir results \
  --quantized-model-dir quantized_model \
  --do-quantize


Loads full‑precision weights.

Uses a small calibration subset from WikiText‑2 to run AutoGPTQ’s one‑shot PTQ. 
PyPI

Saves quantized weights into quantized_model/.

Benchmarks TTFT, latency, throughput, and perplexity.

Subsequent runs (reuse quantized checkpoint):

python quant_bench_llama.py \
  --model-id meta-llama/Meta-Llama-3-8B \
  --backend gptq4 \
  --bits 4 \
  --max-samples 8 \
  --max-new-tokens 32 \
  --output-dir results \
  --quantized-model-dir quantized_model


Here the script will load from quantized_model directly without re‑quantizing.

5. Output format and reproducibility

Each run writes a JSON file under results/, e.g.:

{
  "model_id": "meta-llama/Meta-Llama-3-8B",
  "backend": "bnb4",
  "bits": 4,
  "device": "cuda",
  "max_new_tokens": 32,
  "max_samples": 8,
  "avg_ttft_ms": 145.23,
  "avg_latency_ms": 612.78,
  "throughput_tokens_per_s": 151.92,
  "perplexity": 8.37
}


This makes it easy to:

Re‑run the same experiment with different precision.

Drop results into the tables/plots in the paper.

Demonstrate reproducibility (code + config → same JSON).
