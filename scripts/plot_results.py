"""Render the published aggregate experiment metrics as static PNG and SVG."""
from pathlib import Path
import json
from decimal import Decimal,ROUND_HALF_UP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/results/metrics.json'
OUT=ROOT/'docs/assets/results'
BLUE='#005BAC';INK='#163E60';MUTED='#6A8398';TEAL='#16867A';CYAN='#23A3C4';RED='#C62828';GREEN='#15803D'
COLORS=['#809BB0','#D7A061','#9BBFD0','#3793C0',BLUE]

def fmt(v,digits=2):
    return str(Decimal(str(float(v))).quantize(Decimal('1').scaleb(-digits),rounding=ROUND_HALF_UP))

def setup():
    names={f.name for f in font_manager.fontManager.ttflist}
    font=next((n for n in ['Noto Sans CJK SC','Noto Sans CJK JP','Microsoft YaHei','SimHei'] if n in names),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'font.size':12,'axes.titlesize':15,'axes.titleweight':'bold',
      'axes.labelsize':12,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':MUTED,
      'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':False,
      'axes.unicode_minus':False,'grid.color':'#E6EDF3','grid.linewidth':.7,'svg.fonttype':'path',
      'savefig.facecolor':'white','figure.facecolor':'white'})

def axes_style(ax,labels):
    ax.set_xticks(range(len(labels)),labels);ax.tick_params(axis='both',length=0,pad=9)
    ax.grid(axis='y',zorder=0);ax.set_axisbelow(True)
    ax.set_xlim(-.55,len(labels)-.45)

def title(fig,heading,sub):
    fig.suptitle(heading,x=.5,y=.985,fontsize=20,fontweight='bold',color=INK)
    fig.text(.5,.89,sub,ha='center',color=MUTED,fontsize=11.5)

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.2)
    svg=OUT/(name+'.svg')
    fig.savefig(svg,bbox_inches='tight',pad_inches=.2)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)

def overview(stages):
    labels=['基础\nNF4','金融\nSFT','安全与\n工具','DPO','FinSec'];fig,axs=plt.subplots(1,3,figsize=(14.8,5.5))
    title(fig,'金融能力与越狱安全：五阶段统一对照','基线为同一NF4通用模型；数值为准确率或攻击成功率（%），虚线为基础模型实测值。')
    fig.subplots_adjust(top=.78,bottom=.18,wspace=.30)
    for ax,key,heading in zip(axs,['finance','gcg','pair'],['FinEval ↑','GCG ASR ↓','PAIR ASR ↓']):
        axes_style(ax,labels);values=[x[key]['accuracy' if key=='finance' else 'asr']*100 if x[key] else None for x in stages]
        ax.set_title(heading,pad=17,color=BLUE)
        ax.axhline(values[0],color=MUTED,ls=(0,(4,4)),lw=1.2,zorder=1)
        if key=='finance':
            ax.plot(range(5),values,color=BLUE,lw=2.6,zorder=3)
            ax.scatter(range(5),values,c=COLORS,s=70,zorder=4,edgecolor='white',linewidth=1)
            ax.set_ylim(70,80);ax.set_yticks([70,72,74,76,78,80])
        else:
            ax.set_ylim(0,62);ax.set_yticks([0,15,30,45,60])
            for i,v in enumerate(values):
                if v is None:
                    ax.text(i,2.4,'未评测',ha='center',fontsize=10,color=MUTED)
                else:ax.bar(i,v,.62,color=COLORS[i],zorder=2)
        for i,v in enumerate(values):
            if v is not None:ax.annotate(fmt(v),(i,v),xytext=(0,9),textcoords='offset points',ha='center',fontsize=12,fontweight='bold' if i==4 else 'normal',color=BLUE if i==4 else INK)
        ax.set_ylabel('准确率（%）' if key=='finance' else '攻击成功率（%）')
    fig.text(.5,.025,'FinEval：1,151题 · GCG：320行为 · PAIR：313个共同有效行为；安全与工具中间模型的完整攻击结果保持缺项。',ha='center',fontsize=10.5,color=MUTED)
    save(fig,'01_capability_safety')

def recovery(stages):
    chosen=[stages[i] for i in [0,1,3,4]];labels=[x['plot_label'] for x in chosen]
    fig,axs=plt.subplots(1,2,figsize=(14.8,5.4));title(fig,'安全训练的恢复路径：相对基础模型的ASR增量','各阶段沿用同一零点；更接近0表示更接近基础模型风险水平。')
    fig.subplots_adjust(top=.77,bottom=.18,wspace=.25)
    for ax,key in zip(axs,['gcg','pair']):
        base=chosen[0][key]['asr'];values=[(x[key]['asr']-base)*100 for x in chosen]
        axes_style(ax,labels);ax.set_ylim(-2,43);ax.axhline(0,color=INK,lw=1.2)
        ax.set_title(key.upper()+' 风险增量',color=BLUE,pad=14);ax.set_ylabel('高于NF4基础模型的ASR（百分点）')
        for i,v in enumerate(values):
            ax.bar(i,v,.55,color=[COLORS[0],COLORS[1],COLORS[3],COLORS[4]][i],zorder=3)
            ax.text(i,v+.9,'0.00' if i==0 else '+'+fmt(v),ha='center',color=MUTED if i==0 else RED,fontsize=13,fontweight='bold')
        for a,b in [(1,2),(2,3)]:ax.annotate('',xy=(b-.27,values[b]+4),xytext=(a+.27,values[a]+4),arrowprops={'arrowstyle':'->','color':GREEN,'lw':2.2})
        ax.text(1.45,42.3,'安全与工具SFT + DPO\n−'+fmt(values[1]-values[2])+' pp',fontsize=10.5,color=GREEN,ha='center',va='top')
        ax.text(2.8,34,'真实难例 + 回放\n−'+fmt(values[2]-values[3])+' pp',fontsize=10.5,color=GREEN,ha='center',va='top')
    fig.text(.5,.025,'DPO点包含前序安全与工具训练；FinSec为DPO后的安全对抗SFT。最终ASR仍高于基础模型，完整值与能力收益分别报告。',ha='center',fontsize=10.5,color=MUTED)
    save(fig,'02_safety_recovery')

def agent(stages):
    fig,axs=plt.subplots(2,2,figsize=(14.8,8.5));title(fig,'Pi-Agent：审计质量与真实主动工具调用','固定82个文件、32个良性案例；所有阶段使用相同Pi模式与工具协议。')
    fig.subplots_adjust(top=.79,bottom=.12,hspace=.56,wspace=.24)
    labels=[x['plot_label'] for x in stages]
    for ax,key,heading in zip(axs.flat[:3],['precision','recall','f1'],['Precision ↑','Recall ↑','F1 ↑']):
        values=[x['agent'][key]*100 for x in stages];axes_style(ax,labels)
        ax.set_title(heading,color=BLUE,pad=12);ax.set_ylim(0,101);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('百分比（%）')
        ax.axhline(values[0],color=MUTED,ls=(0,(4,4)),lw=1.2,zorder=1)
        ax.bar(range(5),values,.58,color=COLORS,zorder=2)
        for i,v in enumerate(values):ax.text(i,v+2.2,fmt(v),ha='center',fontsize=12,color=BLUE if i==4 else INK,fontweight='bold' if i==4 else 'normal')
        ax.text(.98,.96,'FinSec '+('+' if values[-1]>=values[0] else '−')+fmt(abs(values[-1]-values[0]))+' pp',transform=ax.transAxes,ha='right',va='top',color=RED,fontsize=11.5)
    ax=axs.flat[3];axes_style(ax,labels);ax.set_title('模型主动工具调用',color=BLUE,pad=12)
    total=[x['agent']['active_tool_calls'] for x in stages];execution=[x['agent']['execution_tool_calls'] for x in stages]
    ax.bar(range(5),execution,.58,color=BLUE,label='Python / Bash执行',zorder=2)
    ax.bar(range(5),np.array(total)-execution,.58,bottom=execution,color='#AFCBDD',label='其他主动工具',zorder=2)
    for i,v in enumerate(total):ax.text(i,v+2,fmt(v,0),ha='center',fontsize=12,fontweight='bold')
    ax.set_ylim(0,86);ax.set_yticks([0,20,40,60,80]);ax.set_ylabel('调用次数');ax.legend(loc='upper left',frameon=False,fontsize=10)
    fig.text(.5,.025,'工具数不含每阶段164次框架准备；FinSec共58次主动调用，其中49次执行类调用。Precision、Recall、F1分别反映误报、覆盖和综合平衡。',ha='center',fontsize=10.5,color=MUTED)
    save(fig,'03_agent_metrics')

def privacy(stages):
    fig,axs=plt.subplots(1,2,figsize=(14.8,5.6));title(fig,'训练数据隐私：成员推断攻击的跨阶段画像','固定实际金融训练成员2,913条与独立非成员2,913条，分数方向不事后翻转。')
    fig.subplots_adjust(top=.77,bottom=.19,wspace=.24);labels=[x['plot_label'] for x in stages]
    for ax in axs:axes_style(ax,labels)
    for key,name,offset,color,marker in [('loss_score','Loss',-.08,BLUE,'o'),('min_k_score','Min-k20%',.08,TEAL,'s')]:
        x=np.arange(5)+offset;y=np.array([s['mia'][key]['roc_auc'] for s in stages])
        bounds=np.array([s['mia'][key]['family_bootstrap_95pct_ci']['roc_auc'] for s in stages])
        axs[0].errorbar(x,y,yerr=np.array([y-bounds[:,0],bounds[:,1]-y]),fmt=marker+'-',color=color,label=name,lw=1.8,capsize=4,ms=6)
        tpr=[s['mia'][key]['tpr_at_1pct_fpr']*100 for s in stages]
        axs[1].plot(x,tpr,marker=marker,color=color,lw=2,label=name,ms=6)
    axs[0].axhline(.5,color=MUTED,ls=(0,(4,4)),lw=1.2,label='随机区分 0.5')
    axs[0].set_ylim(.475,.54);axs[0].set_title('ROC-AUC与95%区间',color=BLUE,pad=14);axs[0].set_ylabel('ROC-AUC')
    axs[1].set_ylim(0,1.25);axs[1].set_title('TPR @ 1% FPR',color=BLUE,pad=14);axs[1].set_ylabel('低误报约束下成员识别比例（%）')
    axs[0].legend(loc='upper left',frameon=False,fontsize=10);axs[1].legend(loc='lower left',frameon=False,fontsize=10)
    fig.text(.5,.025,'AUC越接近0.5，当前攻击区分能力越弱；TPR越低，1%误报约束下识别到的成员越少。误差线为问题家族bootstrap区间。',ha='center',fontsize=10.5,color=MUTED)
    save(fig,'04_mia_privacy')

def main():
    setup();data=json.loads(DATA.read_text());stages=data['stages']
    assert data['baseline_id']=='base_nf4' and len(stages)==5
    assert stages[0]['finance']['correct']==834 and stages[-1]['finance']['correct']==888
    for s in stages:
        assert s['finance']['total']==1151
        if s['gcg']:assert s['gcg']['behaviors']==320
        if s['pair']:assert s['pair']['behaviors']==313
    overview(stages);recovery(stages);agent(stages);privacy(stages)
    print('Rendered 4 comparison figures as PNG + editable SVG:',OUT)

if __name__=='__main__':main()
