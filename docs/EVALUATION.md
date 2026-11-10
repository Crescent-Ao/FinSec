# 统一评测

## 金融能力

`python3 finsec.py fetch-fineval`按 `configs/fineval_reference.lock.json`下载固定HF发布版本并逐文件校验。数据仅放入本地 `evaluation_data/`。完整公开Academic val为34科、1,151道带答案题；使用原生非thinking模板、零样本A/B/C/D下一Token logits，报告微平均与科目宏平均。公开test无本地真值，不能作为带答案验证集混算。

```bash
python finsec.py eval-fineval --model-dir /path/to/base \
  --variant FinSec --output runs/fineval
```

将待测适配器置于本地 `models/FinSec`；`base`表示同一NF4基座、不附加项目适配器；`finance`对应 `models/finance_sft`。所有阶段保持相同精度、题集和提示协议。主结果变化相对基础模型；第一阶段金融得分保留门槛作为独立约束计算。

## 越狱鲁棒性

`python3 finsec.py setup-harmbench`获取固定版本的官方实现。`fineval_pipeline/harmbench_upstream.py`构建独立兼容运行副本，保留官方GCG/PAIR搜索类和评分规则；修改及来源分别记录。GCG支持本地可微NF4目标，PAIR支持本地/外部攻击器与共享预算账本。

配置模板为 `configs/harmbench_gcg.example.yaml`。先准备基座、适配器、官方13B分类器及独立HarmBench依赖，然后按原CLI分开生成攻击与官方判分：

```bash
python -m fineval_pipeline --config /path/to/config.yaml attack --model M2 --method GCG
python -m fineval_pipeline --config /path/to/config.yaml judge --model M2 --method GCG
```

本项目记录的GCG预算为50步、搜索宽度128、目标回答上限512；这组参数是实验预算，不代表所有HarmBench任务的唯一官方默认值。完整分类覆盖320个文本行为。PAIR比较使用相同有效BehaviorID；历史314行为结果与共同313行为结果不得混用。

## 训练数据隐私

`fineval_pipeline/privacy.py`提供Loss、Min-k与参考模型差异攻击，报告ROC-AUC、TPR@1%FPR、实际FPR和问题家族bootstrap区间。分数方向固定为“越大越像成员”；低FPR取实际ROC工作点，不插值。高TPR代表该攻击更容易识别成员，应从隐私风险角度分析。

比赛固定成员和非成员各2,913条，来自实际第一阶段金融训练及独立留出。同一批记录、序列化、答案区间和模型精度贯穿阶段比较。公开仓库提供实现，实际成员与分数不随源码发布。

## Pi-Agent代码审计

本地模型接收源码、规则与已记录工具返回，主动选择工具，输出带规则ID、原始文件行号、证据、原因和修复建议的findings。目标标签留在评测端，不进入模型消息。

当前原生协议实现见 `pi_crypto_agent/README.md`。训练后模型可通过 `finsec.py audit`实时审计自有文件。保存轨迹后，用 `python3 finsec.py demo`打开查看器，加载该JSON查看输入、生成、调用、返回和结论。

## 离线重算外部实验包

```bash
python3 verify.py --bundle-root /path/to/extracted/FinSec_1123
```

入口从外部提供的逐题预测、官方逐行为标签、成员分数和Agent结果重算，检查两份适配器与记录身份。它不会生成新模型回答，也不会将外部实验包加入Git。
