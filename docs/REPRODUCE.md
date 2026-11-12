# 安装、五阶段评测与结果汇总

## 1. CPU检查与Agent案例回放

推荐Linux、Python3.11和Node22。先运行不加载模型的源码检查：

```bash
git clone https://github.com/Crescent-Ao/FinSec.git
cd FinSec
python3 verify.py
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.cpu.txt
npm ci --prefix pi_crypto_agent --ignore-scripts
python -m unittest tests.test_public_smoke tests.test_experiment_suite -v
node --test pi_crypto_agent/agent_described.test.mjs pi_crypto_agent/agent_native.test.mjs
python3 finsec.py demo
```

打开终端显示的本机地址，页面自动加载三个真实历史案例。该步骤为只读回放，不需要GPU、模型或API凭证。真实Python/Bash执行另需Bubblewrap，验证命令见首页。

## 2. 准备模型与可选评测资源

模型权重不放入Git。既有[ModelScope阶段成果](https://modelscope.cn/models/aiops11/Qwen3.5-9B-FinSec)与本仓库属于同一项目；复现本文完整五阶段表，需要取得对应阶段的适配器，不能用同一适配器替代多个阶段。通用基座、各阶段适配器和官方HarmBench分类器独立准备。

GPU环境按[环境说明](../environment/README.md)安装。训练/目标推理与HarmBench分别使用独立虚拟环境。FinEval下载与官方攻击源码准备：

```bash
python3 finsec.py fetch-fineval
python3 finsec.py setup-harmbench
```

MIA还需实际金融训练文件、独立非成员、训练状态及曝光清单，详见[评测说明](EVALUATION.md)。Pi批量评测入口沿用固定82个唯一案例和12轮/4096输出/16384上下文协议，需要自备案例文件；公开三个案例用于演示，不替代完整评测集。

## 3. 规划或执行五阶段实验

```bash
cp configs/evaluation_suite.example.json configs/evaluation_suite.local.json
```

编辑本地配置中的基座、四个适配器、输出目录与GPU编号。路径相对仓库根目录解析，也支持绝对路径。`pipeline_config`指向完整YAML；可从`configs/harmbench_gcg.example.yaml`复制本地版本，补齐数据、分类器及独立HarmBench Python路径。

```bash
# 只生成计划，不加载模型或执行任务
python finsec.py eval-suite --config configs/evaluation_suite.local.json

# 顺序运行五阶段FinEval，释放各阶段目标模型后继续下一阶段
python finsec.py eval-suite --config configs/evaluation_suite.local.json --execute

# 从已有结果重新汇总；缺项保持—
python finsec.py eval-suite --config configs/evaluation_suite.local.json --collect
```

输出包含`suite.yaml`、`plan.json`、执行状态、每阶段原始结果、`comparison.json`和`comparison.md`。模型目录/adapter身份、数据与协议由各评测器记录；FinEval相对基础模型的增量仅在题集、协议与覆盖一致时计算。更换任务或模型配置时使用新的输出目录，防止混入旧缓存。

选择其他任务时，在配置中设置`tasks`，或传入：

```bash
python finsec.py eval-suite --config configs/evaluation_suite.local.json \
  --tasks fineval privacy agent gcg pair
```

以上仍只规划；加`--execute`才运行。GCG/PAIR各自分为攻击生成和官方分类器评分；`agent`调用真正的Pi原生工具批量评测器。默认PAIR使用YAML中的本地攻击器，不会调用DeepSeek。API攻击由独立campaign管理，需另外配置凭证、模型及预算。

汇总表中的攻击ASR保留每次运行的实际分母，解释变化前应对齐BehaviorID、搜索预算和分类器。本文已发布PAIR表采用313个共同有效行为，不能把新运行的任意分母直接套入该历史表。

单阶段适配器也可直接评估：

```bash
python finsec.py eval-fineval --model-dir /path/to/base \
  --variant dpo --adapter /path/to/dpo_adapter --output runs/dpo_fineval
```

`--variant base`不允许附加适配器；其余支持`finance`、`safety_tools`、`dpo`和`FinSec`。

## 4. 在自己的代码上实时审计

```bash
python finsec.py audit --model-dir /path/to/base --adapter /path/to/FinSec \
  --file examples/ecb_demo.py --python "$PWD/.venv/bin/python" --output runs/audit
```

默认规则来自仓库内`knowledge/crypto_rules.json`。加载输出`audit.json`查看真实新推理；它与预置历史案例回放分别标识。模型原始生成、运行时实际工具返回与审计端安全补充相互区分。

## 5. 重新生成公开实验图

```bash
python -m pip install -r environment/requirements.figures.txt
python scripts/plot_results.py
python scripts/update_results_docs.py
```

该步骤读取公开聚合JSON，不运行新模型测评。已有完整比赛实验包时，可另用`python3 verify.py --bundle-root /path/to/FinSec_1123`从逐样本记录重算比赛指标。
