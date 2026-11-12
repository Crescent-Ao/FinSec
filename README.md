# FinSec

[![CI](https://github.com/Crescent-Ao/FinSec/actions/workflows/ci.yml/badge.svg)](https://github.com/Crescent-Ao/FinSec/actions/workflows/ci.yml) · [v0.1.0](https://github.com/Crescent-Ao/FinSec/releases/tag/v0.1.0)

**面向金融应用的智能体能力增强与安全验证**

[English](README.en.md) · [方法框架](docs/METHOD.md) · [五阶段复现](docs/REPRODUCE.md) · [Agent真实案例](docs/DEMO.md) · [实验结果](docs/RESULTS.md) · [ModelScope阶段成果](https://modelscope.cn/models/aiops11/Qwen3.5-9B-FinSec)

FinSec构建从数据蒸馏、多阶段后训练到密码学代码审计Agent的完整工程流程。项目以不超过10B的通用指令模型为基础，采用统一NF4量化口径验证金融能力、越狱鲁棒性、训练数据成员隐私与工具审计效果。

本仓库公开数据工程、多阶段后训练、统一评测和Pi-Agent源代码，附聚合实验结果及三个合成代码上的真实Agent案例回放。模型适配器、完整训练与评测数据由运行者另行准备。

![FinSec数据工程、多阶段后训练与Pi-Agent接入](docs/assets/method/overview.png)

## 实现内容

- **数据工程**：金融、安全、隐私、密码学审计及工具轨迹蒸馏；SFT、chosen/rejected偏好与可验证奖励数据构建；去重、家族划分及训练评测隔离。
- **后训练**：金融知识SFT → 安全与工具SFT → DPO → 安全对抗SFT，得到最终FinSec。保留GRPO奖励插件和独立探索实现。
- **统一评测**：FinEval选项logits；官方HarmBench GCG/PAIR流程及本地NF4适配；Loss、Min-k、参考模型差异成员推断；Pi-Agent规则与行号匹配。
- **执行型Agent**：基于真正的Pi Agent框架，原生tools schema、assistant工具请求、tool返回进入同一个本地模型上下文。11项工具覆盖源码、规则、工作区读写、Python测试及Bash执行。

## 快速体验Agent

```bash
python3 finsec.py demo
```

打开终端中的本机地址，自动加载三组历史真实轨迹，无需GPU或API凭证。模型输出、Python/Bash真实返回和最终结论在同一页面展示；支持导入自己保存的审计结果。

![FinSec Agent实际工具执行案例](docs/assets/demo/agent_walkthrough.gif)

案例源码、逐步截图和修复说明见[Agent真实案例](docs/DEMO.md)。在自己的代码上执行新推理见[复现指南](docs/REPRODUCE.md#4-在自己的代码上实时审计)。

<!-- EXPERIMENT_RESULTS_START -->
## 实验效果与可视化

统一以**同一NF4基础模型**为参照，展示金融知识、安全与工具、DPO及最终FinSec五个阶段，以下为比赛实验的聚合结果。

**FinSec的金融准确率为77.15%，比基础模型提高4.69个百分点；审计Precision为82.56%，提高7.83个百分点；Recall与F1分别提高2.75和4.82个百分点。**

| 模型阶段 | FinEval ↑ | GCG ASR ↓ | PAIR ASR ↓ | 审计Precision ↑ | 审计Recall ↑ | 审计F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 基础模型 | 72.46% | 9.06% | 20.45% | 74.73% | 62.39% | 68.00% |
| 金融模型 | 76.46% | 32.81% | 53.67% | 77.00% | 70.64% | 73.68% |
| 安全与工具模型 | 76.63% | — | — | 78.49% | 66.97% | 72.28% |
| 偏好优化模型 | 76.72% | 18.13% | 28.75% | 79.07% | 62.39% | 69.74% |
| **FinSec模型** | **77.15%** | **16.56%** | **28.43%** | **82.56%** | **65.14%** | **72.82%** |

![五阶段FinEval、GCG和PAIR结果；虚线为NF4基础模型](docs/assets/results/01_capability_safety.png)

### 安全训练如何发挥作用

以基础模型风险为零点，GCG额外ASR沿“金融适配 → 安全与工具SFT及DPO → FinSec”从 **+23.75 → +9.06 → +7.50** 个百分点收敛；PAIR对应为 **+33.23 → +8.31 → +7.99** 个百分点。该路径展示安全偏好与真实难例增强对专业适配后风险的恢复作用，所有增量仍使用同一基础模型零点。

![以同一基础模型为零点的GCG与PAIR风险增量收敛路径](docs/assets/results/02_safety_recovery.png)

### Agent审计质量与工具参与

FinSec在本组主线模型中取得最高审计Precision。图中同时保留Recall、F1和工具执行统计，呈现误报、覆盖与实际验证行为的关系。FinSec主动调用工具 **58次**，其中 **49次**为Python或Bash执行；基础模型分别为3次和2次。统计不包含框架自动准备。

![Pi-Agent精确率、召回率、F1及模型主动工具调用](docs/assets/results/03_agent_metrics.png)

### 成员推断隐私检验

固定实际金融训练成员与独立非成员各2,913条，比较Loss与Min-k20%攻击。AUC越接近0.5，表示当前攻击区分能力越弱；TPR越低，表示在1%误报约束下识别出的成员越少。

![成员推断AUC、问题家族置信区间与低误报TPR](docs/assets/results/04_mia_privacy.png)

完整隐私指标、评测分母和图表复现方式见 **[实验结果与口径](docs/RESULTS.md)**。公开[聚合指标JSON](docs/results/metrics.json)与绘图源码，图片同时提供PNG和SVG；模型权重和逐样本记录另行管理。

<!-- EXPERIMENT_RESULTS_END -->

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

GitHub Actions已启用CPU、协议和公开文档检查，配置见[ci.yml](.github/workflows/ci.yml)。`ci/github-actions.example.yml`保留可复用模板。完整测试集还会检查可选训练依赖与官方HarmBench集成；核心CPU协议测试不需要模型权重。

真实Python/Bash工具执行需要Linux Bubblewrap，并可运行：

```bash
FINSEC_PYTHON="$PWD/.venv/bin/python" bash scripts/check_tools.sh
```

其中最后一项使用脚本化回复连接真正的Pi工具循环与执行沙箱，用于验证接线；它不构成一次模型质量测评。

## 五阶段统一评测

编辑本地配置中的基座和四个阶段适配器路径，默认规划五阶段FinEval：

```bash
cp configs/evaluation_suite.example.json configs/evaluation_suite.local.json
python finsec.py eval-suite --config configs/evaluation_suite.local.json
# 加 --execute 执行；加 --collect 汇总已有输出
```

同一入口支持`fineval`、`privacy`、`agent`、`gcg`、`pair`；攻击搜索和官方判分分别记录。模型按阶段顺序加载，基础模型始终不附加适配器；输出统一JSON与Markdown表，缺项保持“—”。环境、资源与完整命令见[复现指南](docs/REPRODUCE.md)。

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

使用 `python3 finsec.py demo` 打开案例回放及本地轨迹查看器，也可加载自己保存的审计JSON。公开案例不包含隐藏评测标签。

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
| `demo/` | 真实历史Agent案例与本地轨迹查看器 |
| `docs/` | 方法、复现、结果、静态图和公开聚合数据 |

## 许可与引用

原创项目代码采用 [Apache-2.0](LICENSE)。HarmBench相关适配、Pi依赖及其他第三方组件沿用其原始许可，见 [NOTICE](NOTICE)。本项目使用官方HarmBench实现，来源与方法文献见 [参考资料](docs/REFERENCES.md)。

使用本项目请通过[CITATION.cff](CITATION.cff)引用软件版本；改进建议和问题反馈见[贡献指南](CONTRIBUTING.md)，版本变更见[CHANGELOG](CHANGELOG.md)。
