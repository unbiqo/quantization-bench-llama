# Quantization Benchmarking with LLaMA

This repository contains the code and artifacts for the **Advanced Computer Architecture I** final project:

> **Quantization Benchmarking with LLaMA**  
> Author: Damir Sarsenov  
> Term: Fall 2025

The goal of this project is to prototype a **mini benchmark harness** that compares **precision regimes** for LLaMA-class models (FP16 vs 4-bit quantization) along the metrics proposed in the accompanying survey paper:

- **Quality** – perplexity on WikiText-2  
- **Performance**
  - TTFT proxy (time-to-first-token, ms)  
  - End-to-end latency per prompt (ms)  
  - Throughput (tokens / second)  
- **Memory (conceptual)** – via bit-width and parameter count (discussed in the paper)  
- **Energy (conceptual)** – reasoned qualitatively from memory traffic and performance; not directly measured  

In the paper and code we treat **LLaMA-3 8B and 70B** as the primary “LLaMA-class” targets, since they are representative large open LLMs.

Because this project was developed on a **CPU-only machine**, the sample result shipped in this repository was generated using **TinyLlama-1.1B-Chat**. On a GPU with sufficient memory and a valid Hugging Face token, the same script and flags can be used with `meta-llama/Meta-Llama-3-8B` or `meta-llama/Meta-Llama-3-70B`.

---

## 1. Repository layout

```text
.
├── Survey_Quantization_LLaMA.pdf               # Project paper
├── Presentation_Quantization_Benchmarking_LLaMA.pptx
├── quant_bench_llama.py                        # Main benchmark script
├── requirements.txt                            # Python dependencies
├── results/
│   └── results_fp16_16bit.json                 # Sample TinyLlama FP16 run on CPU
└── .gitignore
```
The paper describes the quantization methods (GPTQ, Tender), the architectural metrics, and the proposed benchmark protocol.

The Python script implements a minimal harness that can:

load a LLaMA-class (or TinyLlama-class) model,

run generation on a WikiText-2 validation subset,

measure TTFT proxy, latency, throughput, and perplexity,

save results as a JSON file under results/.

2. Environment and installation
2.1 Python and hardware
Python 3.10+ is recommended.

A CUDA-capable GPU is recommended for realistic runtimes and for running 8B/70B models.

The code can fall back to CPU, but it will be slow and is mainly useful for small models such as TinyLlama.

2.2 Install PyTorch
Follow the official instructions for your OS/GPU at the PyTorch website:

https://pytorch.org/get-started/locally/

Example (Linux, CUDA 12.x):

bash
Copy code
pip install --upgrade "torch==2.3.1+cu121" --index-url https://download.pytorch.org/whl/cu121
On CPU-only machines you can install the CPU wheel instead.

2.3 Install Python libraries
From the project root:

bash
Copy code
pip install -r requirements.txt
This brings in, among others:

transformers – model and tokenizer loading, generation

datasets – WikiText-2 dataset

bitsandbytes – 4-bit NF4 quantization (bnb-4bit)

auto-gptq – GPTQ 4-bit quantization and kernels

Note. auto-gptq and bitsandbytes typically require Linux/Windows with NVIDIA GPUs and specific CUDA versions. On CPU-only setups the fp16 backend will still run, but the 4-bit backends may not be available.

3. Model access (LLaMA-class and TinyLlama)
The script is written to work with any Hugging Face model that is compatible with transformers and, for GPTQ, auto-gptq.

3.1 LLaMA-3 models (8B / 70B)
Example model IDs:

meta-llama/Meta-Llama-3-8B

meta-llama/Meta-Llama-3-70B

To use these models you must:

Log in to Hugging Face.

Visit the model card (e.g., meta-llama/Meta-Llama-3-8B).

Read and accept the Meta LLaMA license.

Configure HF_TOKEN (either via huggingface-cli login or environment variable).

With a suitable GPU, you can then pass the model ID via --model-id.

3.2 TinyLlama prototype model
For the actual prototype run included in this repository, the following model was used on CPU:

TinyLlama/TinyLlama-1.1B-Chat-v1.0

This is small enough to run on a CPU-only machine and is used to validate the harness and output format.

4. How to run the benchmark
All commands below assume you are in the repository root and have installed dependencies.

Common flags:

--model-id – Hugging Face model ID (TinyLlama or LLaMA-3).

--backend – one of fp16, bnb4, gptq4.

--bits – numerical precision (e.g., 16 for FP16, 4 for 4-bit modes).

--max-samples – number of prompts sampled from WikiText-2 validation split.

--max-new-tokens – maximum tokens generated per prompt.

--output-dir – directory where JSON results are written (default: results/).

4.1 FP16 baseline (TinyLlama, CPU prototype)
This is the configuration that produced the sample JSON file results_fp16_16bit.json on a CPU-only machine:

bash
Copy code
python quant_bench_llama.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --backend fp16 \
  --bits 16 \
  --max-samples 4 \
  --max-new-tokens 32 \
  --output-dir results
This run:

loads TinyLlama in FP16,

samples 4 validation prompts from Salesforce/wikitext, subset wikitext-2-raw-v1,

reports TTFT proxy, average latency, throughput, and perplexity,

saves a JSON file under results/.

4.2 FP16 baseline (LLaMA-3-8B, GPU)
On a GPU with sufficient memory and a valid LLaMA token, the same script can be used with an 8B model:

bash
Copy code
python quant_bench_llama.py \
  --model-id meta-llama/Meta-Llama-3-8B \
  --backend fp16 \
  --bits 16 \
  --max-samples 4 \
  --max-new-tokens 32 \
  --output-dir results
For a 70B model, the command is the same with --model-id meta-llama/Meta-Llama-3-70B, but this typically requires multi-GPU or very high-memory hardware.

4.3 4-bit bitsandbytes (bnb-4bit)
bash
Copy code
python quant_bench_llama.py \
  --model-id meta-llama/Meta-Llama-3-8B \
  --backend bnb4 \
  --bits 4 \
  --max-samples 4 \
  --max-new-tokens 32 \
  --output-dir results
This configuration uses BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4"), which loads the model with 4-bit weights in a QLoRA-style setup.

4.4 4-bit GPTQ (AutoGPTQ)
First run (quantize + benchmark):

bash
Copy code
python quant_bench_llama.py \
  --model-id meta-llama/Meta-Llama-3-8B \
  --backend gptq4 \
  --bits 4 \
  --max-samples 4 \
  --max-new-tokens 32 \
  --output-dir results \
  --quantized-model-dir quantized_model \
  --do-quantize
This run:

loads full-precision weights,

uses a calibration subset from WikiText-2 for one-shot GPTQ,

saves quantized weights into quantized_model/,

benchmarks TTFT, latency, throughput, and perplexity,

writes a JSON result to results/.

Subsequent runs can reuse the quantized checkpoint:

bash
Copy code
python quant_bench_llama.py \
  --model-id meta-llama/Meta-Llama-3-8B \
  --backend gptq4 \
  --bits 4 \
  --max-samples 4 \
  --max-new-tokens 32 \
  --output-dir results \
  --quantized-model-dir quantized_model
5. Output format and example
Each successful run writes a JSON file under results/, for example:

json
Copy code
{
  "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "backend": "fp16",
  "bits": 16,
  "device": "cpu",
  "max_new_tokens": 32,
  "max_samples": 4,
  "avg_ttft_ms": 2673.57,
  "avg_latency_ms": 16552.97,
  "throughput_tokens_per_s": 1.93,
  "perplexity": 10.565
}
These fields are deliberately simple so that they can be:

dropped directly into tables or plots in the report,

compared across different precision regimes (FP16 vs 4-bit),

used to reproduce experiments by re-running the same command.

6. Limitations and future work
For this course project:

Only a single TinyLlama FP16 CPU run is included as an actual measurement.

LLaMA-3 8B/70B models are used conceptually in the paper for:

scaling laws and memory estimates,

organizing reported results from GPTQ and Tender.

Future work (as outlined in the report) includes:

running the harness on real 8B/70B LLaMA checkpoints on GPU,

extending the implementation to activation-aware or group-wise quantization schemes,

adding direct energy measurements (e.g., board power sampling) to complement the qualitative discussion in the paper,

integrating the harness into a full serving stack to study TTFT and throughput under more realistic load.
