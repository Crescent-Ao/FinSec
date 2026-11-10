# FinSec

**面向金融应用的智能体能力增强与安全验证**

FinSec构建从数据蒸馏、多阶段后训练到密码学代码审计Agent的完整工程流程。项目以不超过10B的通用指令模型为基础，采用统一NF4量化口径验证金融能力、越狱鲁棒性、训练数据成员隐私与工具审计效果。

本仓库公开训练、评测、数据工程及Pi-Agent源代码。模型适配器、训练数据、基准题目、实验原始记录和计算环境由运行者另行准备；Git历史保持源码发布。

```mermaid
flowchart LR
    A[公开金融研究数据 / 多教师蒸馏] --> B[任务归一 · 校验 · 问题家族划分]
    B --> C[金融知识SFT]
    C --> D[安全与工具SFT]
    D --> E[DPO安全偏好]
    E --> F[真实攻击难例 + 回放 SFT]
    F --> G[FinSec模型]
    G --> H[Pi-Agent密码学审计]
    G --> I[FinEval / GCG-PAIR / MIA]
```

## 实现内容

- **数据工程**：金融、安全、隐私、密码学审计及工具轨迹蒸馏；SFT、chosen/rejected偏好与可验证奖励数据构建；去重、家族划分及训练评测隔离。
- **后训练**：金融知识SFT → 安全与工具SFT → DPO → 安全对抗SFT，得到最终FinSec。保留GRPO奖励插件和独立探索实现。
- **统一评测**：FinEval选项logits；官方HarmBench GCG/PAIR流程及本地NF4适配；Loss、Min-k、参考模型差异成员推断；Pi-Agent规则与行号匹配。
- **执行型Agent**：基于真正的Pi Agent框架，原生tools schema、assistant工具请求、tool返回进入同一个本地模型上下文。11项工具覆盖源码、规则、工作区读写、Python测试及Bash执行。

## 快速开始

源码检查只需要Python3.10或更新版本：

```bash
git clone https://github.com/Crescent-Ao/FinSec.git
cd FinSec
python3 verify.py
python3 finsec.py --help
```

安装CPU测试及Agent依赖，推荐Linux、Python3.11、Node22：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.cpu.txt
npm ci --prefix pi_crypto_agent --ignore-scripts

python -m unittest tests.test_public_smoke tests.test_stage2_reward_serialization tests.test_privacy_exposure -v
node --test pi_crypto_agent/agent_described.test.mjs
```

`ci/github-actions.example.yml`提供可启用的GitHub Actions配置模板。完整测试集还会检查可选训练依赖与官方HarmBench集成；核心CPU协议测试不需要模型权重。

真实Python/Bash工具执行需要Linux Bubblewrap，并可运行：

```bash
FINSEC_PYTHON="$PWD/.venv/bin/python" bash scripts/check_tools.sh
```

其中最后一项使用脚本化回复连接真正的Pi工具循环与执行沙箱，用于验证接线；它不构成一次模型质量测评。

## 训练

GPU依赖见 `environment/requirements.in` 和锁定版本。训练使用ms-swift、NF4双重量化、BF16计算及LoRA；各阶段明确保存适配器。先提供自己的数据、通用基座和父阶段适配器，再生成训练计划：

```bash
python finsec.py train-plan --stage dpo \
  --model-dir /path/to/Qwen3.5-9B --parent /path/to/safety_adapter \
  --data /path/to/dpo_train.jsonl --validation-data /path/to/dpo_val.jsonl \
  --output runs/dpo --micro-batch 1
```

添加 `--execute` 执行。支持阶段：`finance_demo`、`safety_tools`、`safety_long`、`dpo`、`finsec`。多卡使用 `--gpus 0,1,2`，有效batch约束会在启动前校验。历史整链驱动、训练审计和GRPO实现也在 `scripts/`。

详细生成要求和消息格式见 [数据说明](docs/DATA.md) 及 `data_specs/`；训练方法和参数见 [训练说明](docs/TRAINING.md)。

## 评测与本地审计

```bash
# 固定版本FinEval，仅下载到被Git忽略的本地评测目录
python3 finsec.py fetch-fineval

# 固定官方HarmBench源码，保留官方GCG/PAIR实现
python3 finsec.py setup-harmbench

# 通用基座下载；需要已安装huggingface_hub
python finsec.py download-base

# 提供自己的FinSec适配器，使用本地模型审计示例源码
python finsec.py audit --model-dir models/Qwen3.5-9B \
  --adapter /path/to/FinSec --python "$PWD/.venv/bin/python" \
  --file examples/ecb_demo.py --output runs/audit
```

核心判断由本地增强模型承担，工具提供真实执行证据。NVIDIA A100和RTX4090为本地NF4实现的主要运行设备；国产PPU等异构环境的厂商运行时另行适配。

使用 `python3 finsec.py demo` 打开本地轨迹查看器，加载自己保存的审计JSON。训练数据与留出标签不会被公开演示页面预置。

评测协议和运行方法见 [评测说明](docs/EVALUATION.md)。已有完整实验包可通过 `python3 verify.py --bundle-root /path/to/FinSec_1123` 重算逐样本指标。

## 实验口径

所有效果对比均以同一NF4基础模型为参照。比赛记录中的FinSec在FinEval完整1,151题上达到77.15%，审计Precision达到82.56%；较基础模型分别提高4.69、7.83个百分点。GCG和PAIR的风险增量沿安全训练主线收敛，完整能力与风险指标分别分析。

FinEval题目及答案不进入微调数据。成员推断针对实际金融训练成员和独立非成员；不能用预训练成员假设替代。Agent演示围绕增强模型部署后的密码学误用审计效果展开。

## 目录

| 路径 | 用途 |
|---|---|
| `fineval_pipeline/` | 金融、攻防、隐私和Agent评测核心模块 |
| `scripts/` | 蒸馏、数据校验、训练、奖励与验证实现 |
| `pi_crypto_agent/` | Pi编排、本地模型桥、原生协议和沙箱工具 |
| `submission_tools/` | 本地审计入口、工具接线检查与实验指标重算 |
| `configs/` | 基准版本锁及配置示例 |
| `data_specs/` | SFT、DPO、GRPO生成要求 |
| `tests/` | 自编夹具上的协议、隐私隔离、评分与源码检查 |
| `examples/` | 用于工具验证的原创示例代码 |

## 许可与引用

原创项目代码采用 [Apache-2.0](LICENSE)。HarmBench相关适配、Pi依赖及其他第三方组件沿用其原始许可，见 [NOTICE](NOTICE)。本项目使用官方HarmBench实现，来源与方法文献见 [参考资料](docs/REFERENCES.md)。
