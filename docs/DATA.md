# 数据构建与外部数据格式

仓库发布构建逻辑和生成要求。实际训练文件、评测题目、留出标签与成员数据由运行者在本地准备，分别放在Git忽略目录中。

## 数据工程

公开金融研究语料、GPT/Gemini/DeepSeek等教师蒸馏语料通过统一归一、数值/源码核验、去重与问题家族划分，输出不同监督形式：

1. 金融知识与推理SFT。
2. 安全边界、合法请求、隐私行为、密码学审计和工具轨迹SFT。
3. 同一请求下chosen/rejected回答的偏好对。
4. 真实GCG/PAIR训练难例、安全回答改写和能力回放。
5. GRPO policy可见输入与独立奖励真值。

具体生成规范见 `data_specs/`。核心实现包括 `generate_stage2_deepseek.py`、`merge_stage2_safety_data.py`、`validate_stage2_delivery.py`、`prepare_hardening_curriculum.py`、`finalize_hardening_curriculum.py`。教师API价格和预算参数需按使用者账户配置。

## SFT消息

每行一个JSON对象，典型结构：

```json
{"messages":[{"role":"user","content":"自编训练问题"},{"role":"assistant","content":"经过验证的参考回答"}]}
```

工具示范保留assistant.tool_calls与role=tool，`validate_stage2_delivery.native_messages`负责原生格式归一。禁止把工具结果重新拼成含糊的普通user消息。

## DPO消息

```json
{"messages":[{"role":"user","content":"自编训练问题"},{"role":"assistant","content":"chosen回答"}],"rejected_response":"rejected回答"}
```

同一问题及其改写、偏好派生、工具轨迹继承同一家族身份；训练/验证/测试按家族隔离。

## GRPO任务

可见输入与奖励真值分文件保存，奖励端可包含 `private_ground_truth`；不得把答案标签放入policy prompt。实现见 `stage2_reward_plugin.py`、`stage2_verifiers.py`。GRPO是独立探索分支，最终FinSec主线为DPO后的安全对抗SFT。

## 成员推断与基准隔离

成员必须来自实际使用过的金融微调训练输入。非成员不得加入后续SFT、DPO的chosen/rejected任一分支、GRPO或检索库。`privacy_exposure.py`检查完整消息和归一用户提示的重合。FinEval题目和答案保持评测用途，不作为蒸馏或微调数据。

比赛使用的规模为：547,247条金融样本；5,728条常规安全/工具样本及34条长轨迹；8,907对DPO数据；最终1,931行安全对抗训练曝光。最后一个数字含真实难例过采样，不等于1,931个独立攻击。
