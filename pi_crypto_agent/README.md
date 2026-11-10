# FinSec Pi-Agent

当前入口是 `agent_described_4096.mjs` 与 `native_worker_described_4096.py`。Pi负责状态、参数校验、工具循环与执行记录；Python桥使用 `fineval_pipeline.core.Backend`加载本地NF4模型及未合并LoRA。工具schema、assistant.tool_calls与role=tool按原生模板保留。

## 工具

| 工具 | 作用 |
|---|---|
| read_source | 读取最初源码和行号 |
| search_code | 在原始代码中进行字面检索 |
| static_scan | 返回待模型核对的静态候选 |
| lookup_rules | 查询本地密码学规则 |
| run_python_test | 导入当前原函数并执行Python测试 |
| workspace_info | 查询当前案例工作区 |
| list_files / read_file | 列举与读取工作区文件 |
| write_file / replace_text | 创建测试或修改候选修复 |
| run_command | 执行实际Bash/Python命令并返回退出码、stdout、stderr |

工具在每案例独立的Bubblewrap工作区运行，无外部网络和宿主项目挂载。最终证据定位始终引用最初输入源码；候选补丁和模型自编测试分别记录。模型自述不替代真实执行返回。

## 安装与检查

```bash
npm ci --prefix pi_crypto_agent --ignore-scripts
node --test pi_crypto_agent/agent_described.test.mjs
FINSEC_PYTHON=/path/to/python3.11 bash scripts/check_tools.sh
```

完整GPU实时审计入口：

```bash
python finsec.py audit --model-dir /path/to/base --adapter /path/to/FinSec \
  --python /path/to/python3.11 --file examples/ecb_demo.py --output runs/audit
```

单轮最多4096输出Token、总上下文16384、最多12次模型调用。实际模型生成、参数、执行返回、格式修复和最终findings写入JSON。若原生桥未提供准确Token计数，记录null，不用占位0冒充实测。

`evaluate_described_4096_pi_only.mjs`保留固定82例评测入口，需要使用者自行提供案例与规则。其余agent/evaluate/native_worker变体保留用于历史协议复核，当前接口以本页指明版本为准。
