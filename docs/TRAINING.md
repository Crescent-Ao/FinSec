# 多阶段后训练

## 统一模型口径

基础模型为通用Qwen3.5-9B指令模型，不直接选择金融专项模型。原实现采用4-bit NF4双重量化、BF16计算、LoRA rank16/alpha32/dropout0.05，冻结视觉与对齐部分，保留未合并适配器的加载与对照。

模型与数据保存在本地Git忽略目录；通过 `--model-dir`、`--parent`、`--data`、`--validation-data`传入实际位置。

## 主线与默认参数

| 阶段 | 目标 | 学习率 | 参考有效batch | 最大长度 |
|---|---|---:|---:|---:|
| 金融SFT | 金融专业知识与推理 | 2e-5 | 24 | 4096 |
| 安全与工具SFT | 行为边界与工具使用 | 5e-6 | 25 | 4096 |
| 长轨迹补充 | 保留完整工具交互 | 2e-6 | 8 | 16384 |
| DPO | 安全且有用的回答偏好 | 2e-6 | 16 | 4096 |
| 安全对抗SFT → FinSec | 真实难例、安全改写与能力回放 | 1e-6 | 48 | 4096 |

DPO的β为0.1，同时采用系数0.2的chosen监督项，固定参考模型与策略模型从安全/工具阶段初始化。随后低学习率SFT进一步学习真实攻击下的安全回答，以能力回放保持正常任务响应。

对抗增强借鉴HarmBench/R2D2的攻击驱动训练与正常指令混合思路；本项目采用DPO之后的离线难例集，并非在线攻击池的原样复刻。参考文献见 `REFERENCES.md`。

## 启动与检查点

```bash
python finsec.py train-plan --stage safety_tools \
  --model-dir /path/to/base --parent /path/to/finance_adapter \
  --data /path/to/safety_train.jsonl --validation-data /path/to/safety_val.jsonl \
  --output runs/safety_tools
```

配置与完整命令写入输出目录。确认数据与父适配器后，加 `--execute` 执行。每个阶段使用独立输出目录，保存checkpoint，并把实际完成的checkpoint传入下一阶段。多卡采用DDP；micro batch、卡数和累计步数必须满足有效batch。

`scripts/run_stage2_posttraining.py`保留历史整链驱动及冻结数据逻辑；`stage2_swift_entry.py`采用类型化Swift参数；`merged_training_audit.py`和`stage2_audit.py`记录实际输入/训练审计；`stage2_reward_plugin.py`提供独立GRPO探索奖励。历史驱动假设已准备本地语料，通用运行入口以 `finsec.py` 为准。

`finance_demo`是入口阶段名，使用者应显式传入自有数据。公开仓库不把任何演示样本规模等同于比赛全量训练。
