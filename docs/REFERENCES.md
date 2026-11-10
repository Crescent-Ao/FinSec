# 方法与上游实现

- QLoRA — https://arxiv.org/abs/2305.14314
- Direct Preference Optimization — https://arxiv.org/abs/2305.18290
- HarmBench / R2D2 — https://arxiv.org/abs/2402.04249
- GCG — https://arxiv.org/abs/2307.15043
- PAIR — https://arxiv.org/abs/2310.08419
- FinEval — https://arxiv.org/abs/2308.09975
- Min-k成员推断 — https://arxiv.org/abs/2310.16789
- GRPO / DeepSeekMath — https://arxiv.org/abs/2402.03300
- Safety-Tuned LLaMAs — https://arxiv.org/abs/2309.07875
- 官方HarmBench源码 — https://github.com/centerforaisafety/HarmBench
- Pi Agent框架 — https://github.com/earendil-works/pi
- FinEval数据 — https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval

HarmBench的固定版本见 `configs/harmbench.lock.json`。Pi依赖版本由 `pi_crypto_agent/package-lock.json`锁定，当前原生桥基于 `@mariozechner/pi-agent-core@0.73.1`。
