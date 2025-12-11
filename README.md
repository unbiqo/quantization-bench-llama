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
  - 
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
