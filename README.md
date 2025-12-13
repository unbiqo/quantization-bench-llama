# Quantization Benchmarking with LLaMA

This repository contains the code and artifacts for the **Advanced Computer Architecture I** final project:

> **Quantization Benchmarking with LLaMA**  
> Author: Damir Sarsenov  
> Term: Fall 2025

The goal of this project is to prototype a **mini benchmark harness** that compares **precision regimes** for LLaMA-class models (FP16 vs 4-bit quantization) along the metrics proposed in the accompanying survey paper:

- **Quality**: Perplexity on WikiText-2.
- **Performance**:  
  - TTFT proxy (time-to-first-token, ms)  
  - End-to-end latency per prompt (ms)  
  - Throughput (tokens / second)
- **Memory (conceptual)**: via bit-width and parameter count (discussed in the paper).
- **Energy (discussed in the paper)**: reasoned qualitatively from memory traffic and performance; not directly measured.

In the paper and code we treat **LLaMA-3 8B and 70B** as the primary LLaMA-class targets, since they are representative large open LLMs.  
Due to limited local hardware, the **sample result shipped in this repository** was generated on a CPU-only machine using **TinyLlama-1.1B-Chat** (see Section 4.1). On a GPU with sufficient memory, the same script and flags can be used with `meta-llama/Meta-Llama-3-8B` or `meta-llama/Meta-Llama-3-70B`.

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
