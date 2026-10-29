# Execution environments

| Stage | Hardware | Software |
|---|---|---|
| Qwen2.5-7B SFT, GRPO, evaluation, latency | Apple M5 Pro, 48 GB unified memory | macOS, Python 3.14, mlx-lm 0.32.2 (`requirements-mlx.txt`); model `mlx-community/Qwen2.5-7B-Instruct-4bit` |
| Qwen3-8B and Qwen3.8-27B SFT, GRPO, evaluation | 1x NVIDIA RTX 5090, 32 GB | venv; torch 2.11.0+cu128, transformers 5.20.2, peft 0.21.2, bitsandbytes 0.50.2 (as recorded in `results/gpu_extra/gpu_info.txt` on the same server), TRL 1.16; models `Qwen/Qwen3-8B`, `Qwen/Qwen3.8-27B` |
| Qwen3.8-27B zero-shot (NVFP4 run) | shared server, vLLM OpenAI-compatible endpoint | NVFP4 weights; `scripts/run_api_base.sh` |
| Re-scoring, tables, figures | any | Python 3.9+, numpy, matplotlib |

The GPU library versions were recorded at the time of the extra evaluations (24 Oct 2026) on the server used for all
GPU training (including the control runs of 25 Oct 2026, TRL 1.16.0); the training runs of 22-23 Oct 2026 used the
same virtual environment. The GPU server's Hugging Face cache holds a single snapshot of each model, which is the one
used: `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218` and `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (pass `revision=` to reproduce). The revision of
`mlx-community/Qwen2.5-7B-Instruct-4bit` on the Apple-silicon machine was not recorded. The LoRA adapters are not
released and are rebuilt by the training scripts. Regenerating the detection matrices uses the physics module in
`cuas-gym/` and the Copernicus tile in `cuas-gym/data/dem/`.

## Learned surrogate baseline
scikit-learn 1.9.1, NumPy 2.5.3, CPU only (2 cores); results identical across reruns with the same seed.
