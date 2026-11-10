# 环境与外部资产

源码检查使用Python3.10+。CPU协议测试使用Python3.11、`requirements.cpu.txt`；Pi运行时使用Node22及package-lock固定依赖。真实执行工具依赖Linux Bubblewrap、Crypto及cryptography。

GPU训练/推理依赖见本目录 `requirements.in` 与 `requirements.lock.txt`，历史主环境包括ms-swift4.6.0、PyTorch2.10.0、Transformers5.19.0、PEFT0.20.0和bitsandbytes0.50.2。按自己的CUDA与驱动准备独立环境。

HarmBench环境单独使用 `requirements.harmbench.in`，避免覆盖训练环境的PyTorch。官方13B分类器及版权规则依赖由运行者准备。源码、基座、分类器、数据和适配器分开管理。

基座通过 `python finsec.py download-base`下载固定通用Qwen3.5-9B；模型文件、环境、实验结果等均由 `.gitignore`排除。教师API凭证通过环境变量或本地受限文件提供，不写入源码。
