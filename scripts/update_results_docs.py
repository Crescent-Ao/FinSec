"""Synchronize Markdown comparison tables with the public aggregate metrics."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import json,re

ROOT=Path(__file__).resolve().parents[1]
START='<!-- EXPERIMENT_RESULTS_START -->';END='<!-- EXPERIMENT_RESULTS_END -->'
def fmt(v,d=2):return str(Decimal(str(float(v))).quantize(Decimal('1').scaleb(-d),rounding=ROUND_HALF_UP))
def pct(v):return fmt(v*100)+'%'

def tables(stages):
    main=['| 模型阶段 | FinEval ↑ | GCG ASR ↓ | PAIR ASR ↓ | 审计Precision ↑ | 审计Recall ↑ | 审计F1 ↑ |',
          '|---|---:|---:|---:|---:|---:|---:|']
    privacy=['| 模型阶段 | Loss AUC | Loss TPR@1%FPR | Min-k AUC | Min-k TPR@1%FPR |',
             '|---|---:|---:|---:|---:|']
    for s in stages:
        values=[s['label'],pct(s['finance']['accuracy']),pct(s['gcg']['asr']) if s['gcg'] else '—',pct(s['pair']['asr']) if s['pair'] else '—',
          pct(s['agent']['precision']),pct(s['agent']['recall']),pct(s['agent']['f1'])]
        if s['id']=='finsec':values=['**'+v+'**' for v in values]
        main.append('| '+' | '.join(values)+' |')
        a=s['mia'];privacy.append('| '+' | '.join([s['label'],fmt(a['loss_score']['roc_auc'],6),pct4(a['loss_score']['tpr_at_1pct_fpr']),
          fmt(a['min_k_score']['roc_auc'],6),pct4(a['min_k_score']['tpr_at_1pct_fpr'])])+' |')
    return '\n'.join(main),'\n'.join(privacy)
def pct4(v):return fmt(v*100,4)+'%'

def main():
    data=json.loads((ROOT/'docs/results/metrics.json').read_text());stages=data['stages'];main_table,mia_table=tables(stages)
    section='''## 实验效果与可视化

统一以**同一NF4基础模型**为参照，展示金融知识、安全与工具、DPO及最终FinSec五个阶段，以下为比赛实验的聚合结果。

**FinSec的金融准确率为77.15%，比基础模型提高4.69个百分点；审计Precision为82.56%，提高7.83个百分点；Recall与F1分别提高2.75和4.82个百分点。**

'''+main_table+'''

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
'''
    p=ROOT/'README.md';text=p.read_text().replace('及Pi-Agent源代码。','及Pi-Agent源代码与聚合实验结果。',1)
    if START in text:text=re.sub(re.escape(START)+'.*?'+re.escape(END),START+'\n'+section+'\n'+END,text,flags=re.S)
    else:text=text.replace('## 快速开始',START+'\n'+section+'\n'+END+'\n\n## 快速开始',1)
    p.write_text(text)
    doc='''# 多阶段实验结果与评测口径

数据日期：2026年11月10日。基线为通用Qwen3.5-9B以NF4双重量化、BF16计算加载，不附加项目LoRA。最终FinSec是DPO后继续安全对抗SFT得到的模型。

## 1. 金融能力、安全与审计质量

'''+main_table+'''

![金融能力与越狱安全](assets/results/01_capability_safety.png)

FinSec相对基础模型的FinEval增益为+4.69个百分点，审计Precision、Recall、F1分别为+7.83、+2.75、+4.82个百分点。不同指标反映不同取舍，表格保留全部阶段的实际值。

## 2. 安全恢复路径

![GCG与PAIR基础风险增量](assets/results/02_safety_recovery.png)

DPO点包含前序安全与工具SFT；其后真实GCG/PAIR训练难例、教师安全改写和能力回放共同构成FinSec增强。金融适配到DPO阶段，GCG与PAIR ASR分别下降14.69和24.92个百分点；FinSec进一步下降1.56和0.32个百分点。图中柱高始终相对同一基础模型计算：最终额外风险收敛至7.50和7.99个百分点，同时金融能力保持增长。

## 3. Pi-Agent

![Pi-Agent指标与工具调用](assets/results/03_agent_metrics.png)

固定82个文件，其中32个良性文件。规则ID与原始行号匹配后计算TP/FP/FN及Precision、Recall、F1；格式失败保留在评测中。工具调用只统计模型主动请求，每模型164次框架准备单独处理。所有阶段使用同一Pi工具协议、12次模型调用上限、4096输出Token上限和16384上下文。

## 4. 成员推断

'''+mia_table+'''

![MIA隐私画像](assets/results/04_mia_privacy.png)

所有阶段使用相同实际训练成员2,913条及独立非成员2,913条，方向固定为高分更像成员。Min-k比例为20%；低误报工作点不插值。图中误差线为问题家族bootstrap 95%区间。参考模型差异攻击及各工作点的实际FPR也在聚合JSON中保存。

当前结果形成一致的跨阶段隐私风险画像，并未显示相对基础模型的隐私风险下降。

## 5. 固定协议

| 项目 | 口径 |
|---|---|
| FinEval | 固定HF Academic完整公开val，34科、1,151题；原生非thinking模板、零样本选项logits；非官方test成绩 |
| GCG | 官方实现，320行为；本项目预算50搜索步、宽度128、目标回答最多512新Token |
| PAIR | 官方流程；四模型共同313个有效BehaviorID，单独攻击各目标模型 |
| HarmBench评分 | 相同官方HarmBench-Llama-2-13b-cls及版权规则 |
| MIA | 实际第一阶段金融训练成员与留出非成员；Loss、Min-k及参考模型差异；相同序列化与答案区间 |
| Agent | 固定82文件、32良性；同一Pi原生工具协议，核心判断由本地模型承担 |

安全与工具中间阶段没有同协议完整GCG/PAIR结果，表内保持“—”，不记为0。PAIR历史314行为的结果先对齐共同313个行为再进入本表。所有主结果变化均相对NF4基础模型；第一阶段金融保留门槛另行计算。

## 6. 复现图表

```bash
python -m pip install -r environment/requirements.figures.txt
python scripts/plot_results.py
python scripts/update_results_docs.py
```

中文字体优先使用Noto Sans CJK，也支持Microsoft YaHei/SimHei。PNG用于GitHub直接展示，SVG使用字形轮廓保证跨设备显示。数据源为 [metrics.json](results/metrics.json)，仅含聚合统计、分母、协议与区间，不含训练文本、成员ID或逐样本回答。

已有完整实验包时，可用 `python3 verify.py --bundle-root /path/to/FinSec_1123`重算逐样本结果；这与本页聚合结果绘图分别管理。
'''
    (ROOT/'docs/RESULTS.md').write_text(doc)
    print('Updated README and docs/RESULTS.md from aggregate metrics.')

if __name__=='__main__':main()
