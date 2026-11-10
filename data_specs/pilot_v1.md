# GPT-6-Luna 金融候选数据首批规范

目标：500 条中文金融 SFT 候选，用于 Qwen3.5-9B；不是正式基准测试集。
总量：train 300、validation 100、privacy_nonmembers 100。
所有样本尚未专家审核，review_status 必须为 generated_unreviewed。
生成者是明确指定的 gpt-6-luna 子 Agent，不调用外部教师 API。

## 分工与预划分

读取本目录 pilot_v1_manifest.json，按 owner 选择全部 slots，必须逐条覆盖。
finance：金融 150 条；accounting：会计 150 条；economics_professional：经济与从业知识共 200 条。
split、group_id、task_type、id、domain 必须与 manifest 一致。
manifest 在生成前固定；同一 group_id 是一个材料/问题家族，始终只属于一个 split。
相同材料、同一问题仅换数字/换名字的版本，不得跨 group_id，尤其不得跨 split。
每个家族可有多个相关问题，但应考察不同方面，不要用同一句问题批量替换数字充数。

## 内容

1. 全部原创，使用自足题设/原创短材料。每题所需参数、口径、公式假设和规则都在 context 中明确。
2. 使用稳定且正确的金融、经济、会计概念；避免把虚构规则当作真实法规、准则或监管要求。
3. 不写投资荐股建议，不杜撰真实公司数据、文献、网址、法律条款。
4. 从业知识可涉及职业道德、信息保护、适当性、内控与操作核验，但题设规则不能冒充现行法律。
5. 不访问、不复制、不改写 FinEval 或其他考试基准的题目及答案。
6. 答案简洁但说明关键依据、公式、单位或判断条件；禁止冗长思维独白。
7. 每个家族要有独立具体场景；跨家族不要反复粘贴同一段定义。
8. 不同 split 尽量保持主题、难度和字数分布相似，不能把困难题全部留给非成员集。

## JSONL 原始格式

每行一个对象，字段如下。只写自己的 parts 文件，不修改他人的输出。

~~~json
{
  "id": "manifest 中的 id",
  "group_id": "manifest 中的 group_id",
  "split": "train 或 validation 或 privacy_nonmembers",
  "domain": "finance/accounting/economics/professional",
  "task_type": "mcq/qa/calculation/grounded_qa",
  "topic": "具体主题",
  "context": "原创且自足的中文材料或计算题设",
  "question": "具体问题",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "final_answer": "单选题为 A/B/C/D；计算题为不含单位的十进制结果；其余为简短结论",
  "answer": "完整的目标回答，必须包含 final_answer。单选题以 答案：A。解析： 开头",
  "checks": [{"expression": "1000 * 0.03 * 2", "expected": "60.00", "tolerance": "0.01"}],
  "source": {"kind": "synthetic_authored", "reference": "等于 group_id"},
  "producer": {"requested_model": "gpt-6-luna", "agent": "自己的 owner"},
  "review_status": "generated_unreviewed"
}
~~~

非单选题 options 为 {}。单选选项不能重复，须有唯一正确答案，正确字母尽量均衡。
计算题 checks 必须非空，第一项 expected 必须等于 final_answer（十进制数值相等）；answer 要明确单位。
expression 只允许十进制常数、括号、+ - * / **；禁止变量、函数调用、百分号。
表达式需与题设语义一致，不能只填最终答案常数冒充复算。
四舍五入口径写清楚；tolerance 是绝对误差，可为 0 或 0.01 等非负小数。
每条都检查 context、question、answer 三者一致，选择题检查答案字母与选项内容。

## 交付

finance 写 data/pilot_luna_v1/parts/finance.jsonl。
accounting 写 data/pilot_luna_v1/parts/accounting.jsonl。
economics_professional 写 data/pilot_luna_v1/parts/economics_professional.jsonl。
可在同一 parts 目录保存仅以 owner 为前缀的生成/复算 Python 文件与说明。
建议分批编写原创家族，再用代码精确算数和序列化；不要用统一空洞模板扩增到目标条数。
完成前自己检查条数、manifest 覆盖、JSON 可解析、计算正确和答案合理性。
主 Agent 将独立复算、构造 messages、汇总和审查；这些操作不等于专家确认或 FinEval 去污染完成。
