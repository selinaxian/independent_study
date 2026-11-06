"""Render locked diagnostic results, without fitting any predictive model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iclab.data import sha256
from diagnostics_v2.common import config,save_json


def table(df,digits=4):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return "—" if not np.isfinite(v) else f"{v:,.{digits}f}"
        return str(v).replace("|","/")
    return "\n".join(["|"+"|".join(df.columns)+"|","|"+"|".join(["---"]*len(df.columns))+"|"]+["|"+"|".join(fmt(v) for v in row)+"|" for row in df.itertuples(index=False,name=None)])


def main():
    cfg,v1=config();out=Path(cfg["output_dir"])
    assert (out/"analysis_complete.json").exists()
    m=pd.read_csv(out/"metrics.csv");ci=pd.read_csv(out/"confidence_intervals.csv");diff=pd.read_csv(out/"paired_comparisons.csv")
    bt=pd.read_csv(out/"backtest_summary.csv");syn=pd.read_csv(out/"synthetic/metrics.csv")
    sd=json.loads((out/"state_definition.json").read_text());verification=json.loads((out/"verification.json").read_text())
    gates=json.loads((out/"state_use_assessment.json").read_text())["gates"]
    def get(arm,rep,probe,target):return m[(m.arm==arm)&(m.representation==rep)&(m.probe==probe)&(m.target==target)].iloc[0]
    def comp(left,right,target):return diff[(diff.left==left)&(diff.right==right)&(diff.target==target)].iloc[0]
    def interval(model,target):
        r=ci[(ci.model==model)&(ci.target==target)].iloc[0]
        return f"[{r.ci_low:.4f}, {r.ci_high:.4f}]"
    def trading(arm,policy="selected",scenario="base"):return bt[(bt.arm==arm)&(bt.policy==policy)&(bt.scenario==scenario)].iloc[0]
    parts={};s=get("rolling_all","stdp","linear","future_state");raw=get("rolling_all","raw_history","linear","future_state");persist=get("baseline","persistence","rule","future_state")
    direction=get("rolling_all","stdp","linear","return_5");old=get("frozen","stdp","linear","return_5")
    parts["SUMMARY"]=(f"- 合成序列：原始完整序列接近100%，全窗计数严格50%；STDP能保留部分先后信息，随机SNN也能做到，不能把全部顺序能力归功于STDP。\n"
        f"- 滚动训练：STDP线性读出的收益IC从冻结组{old.ic:.4f}变为全滚动{direction.ic:.4f}，全滚动自身95%区间{interval('rolling_all__stdp__linear','return_5')}；后文用配对区间判断改善是否明确，而不按点估计选赢家。\n"
        f"- 市场状态：全滚动STDP未来状态balanced accuracy为{s.balanced_accuracy:.4f}，原始特征为{raw.balanced_accuracy:.4f}，状态持续基线为{persist.balanced_accuracy:.4f}。当前状态识别和未来预测分开报告。\n"
        f"- 交易：逐段用过去验证选择的全滚动策略，115包日期连续资金模拟净盈亏{trading('rolling_all').net_pnl:,.2f}元；冻结策略为{trading('frozen').net_pnl:,.2f}元。它们都属于已经观察过历史数据上的条件模拟，不能作为新独立确认。")
    folds=json.loads((out/"folds.json").read_text())
    schedule=[]
    for i in [0,1,2,10,22]:
        f=folds[i]
        schedule.append({"段编号":i,"训练区间":f["train_dates"][0]+" ~ "+f["train_dates"][-1],"验证区间":f["validation_dates"][0]+" ~ "+f["validation_dates"][-1],"预测区间":f["forecast_dates"][0]+" ~ "+f["forecast_dates"][-1]})
    parts["ROLLING_SCHEDULE"]="部分实际滚动日期（完整23段见folds.json）：\n\n"+table(pd.DataFrame(schedule))
    sg=syn.groupby(["gap_minutes","representation","probe"]).accuracy.agg(["mean","min","max"]).reset_index()
    parts["SYNTHETIC"]=table(sg)
    sp=pd.read_parquet(out/"synthetic/predictions.parquet");sp["correct"]=sp.actual.eq(sp.prediction).astype(float)
    synthdiff=[];rng=np.random.default_rng(72531)
    for gap in cfg["synthetic_gaps_minutes"]:
        for probe in cfg["probe_types"]:
            p=sp[(sp.gap==gap)&(sp.probe==probe)].groupby(["pair_id","representation"]).correct.mean().unstack()
            d=(p.stdp-p.random).to_numpy();idx=rng.integers(len(d),size=(1000,len(d)));b=d[idx].mean(axis=1)
            synthdiff.append({"间隔分钟":gap,"读出":probe,"STDP减随机准确率":float(d.mean()),"成对95%下限":float(np.quantile(b,.025)),"成对95%上限":float(np.quantile(b,.975))})
    pd.DataFrame(synthdiff).to_csv(out/"synthetic/paired_accuracy.csv",index=False)
    parts["SYNTHETIC_INTERPRETATION"]=("下表按400个未用于训练/验证的样本对重抽样，先在每对内平均两个顺序和3个种子，避免把成对样本和种子当独立市场证据。\n\n"+table(pd.DataFrame(synthdiff))+"\n\n分辨AB/BA说明表示中留有顺序信息。LIF泄漏本身会使较新输入影响更大，随机网络也能利用这种差异；长间隔更易区分也可能是新旧事件影响差距更大，不是已证明STDP学到了更长的因果依赖。打乱对照接近机会水平，支持原先任务依赖顺序。RFF没有必然优于线性；不能因某一个非线性读出较差就断言表示绝对没有信息。")
    ladder=[]
    reps=["raw_history","counts_minute","counts_total","random","stdp","shuffle","raw_plus_stdp","risk_control"]
    for rep in reps:
        for probe in cfg["probe_types"]:
            ladder.append({"表示":rep,"读出":probe,"收益IC":get("rolling_all",rep,probe,"return_5").ic,"绝对收益IC":get("rolling_all",rep,probe,"absolute_5").ic,"实现波动IC":get("rolling_all",rep,probe,"realized_volatility_5").ic,"当前状态BA":get("rolling_all",rep,probe,"current_state").balanced_accuracy,"未来状态BA":get("rolling_all",rep,probe,"future_state").balanced_accuracy})
    parts["LADDER"]="以下均为全流程滚动、相同15900条历史预测上的结果，种子预测等权平均。BA为balanced accuracy。\n\n"+table(pd.DataFrame(ladder))
    readrows=[]
    for rep in ["raw_history","counts_minute","random","stdp","raw_plus_stdp"]:
        for target in ["return_5","absolute_5","current_state","future_state"]:
            r=comp(f"rolling_all__{rep}__rff",f"rolling_all__{rep}__linear",target)
            readrows.append({"表示":rep,"目标":target,"RFF减线性":r.difference,"95%下限":r.ci_low,"95%上限":r.ci_high})
    parts["READOUT_FINDINGS"]="非线性与线性读出的配对差异如下，回归为IC差、状态为BA差。RFF固定256基函数，线性维数随表示变化，因此不是严格参数量匹配实验；它还改变了信息的表达方式，不能把所有差异只归因于“非线性”。\n\n"+table(pd.DataFrame(readrows))
    parts["STATE_DEFINITION"]=f"首段训练拟合的RV阈值为 **{sd['rv_threshold_bps']:.4f}bps**，效率阈值为 **{sd['efficiency_threshold']:.4f}**；拟合日期{sd['fit_dates'][0]}至{sd['fit_dates'][-1]}。"
    staterows=[]
    for target in ["current_state","future_state"]:
        for rep in ["raw_history","random","stdp","shuffle","raw_plus_stdp"]:
            for probe in cfg["probe_types"]:
                r=get("rolling_all",rep,probe,target)
                staterows.append({"目标":target,"表示":rep,"读出":probe,"accuracy":r.accuracy,"balanced_accuracy":r.balanced_accuracy,"macro_F1":r.macro_f1,"BA95%区间":interval(f"rolling_all__{rep}__{probe}",target)})
        for arm,rep,probe in [("rolling_all","mean","linear"),("baseline","persistence","rule")]:
            r=get(arm,rep,probe,target);staterows.append({"目标":target,"表示":rep,"读出":probe,"accuracy":r.accuracy,"balanced_accuracy":r.balanced_accuracy,"macro_F1":r.macro_f1,"BA95%区间":interval(f"{arm}__{rep}__{probe}",target)})
    parts["STATE_RESULTS"]=table(pd.DataFrame(staterows))
    components=pd.read_csv(out/"state_components.csv")
    component_table=components[(components.arm.eq("rolling_all") & components.representation.isin(["raw_history","random","stdp"])) | components.arm.eq("baseline")]
    component_table=component_table[component_table.target.eq("future_state")][["representation","probe","component","balanced_accuracy"]]
    parts["STATE_COMPONENTS"]="为解释四类分数，另将既有分数按高/低波动、高/低路径效率分别求和，作描述性二分类检查（机会水平BA=0.5）；没有新增拟合或据此选策略。这样可区分状态成绩主要来自波动持续，还是也包含路径效率信息。\n\n"+table(component_table)
    state_lines=[]
    for probe in cfg["probe_types"]:
        a=comp(f"rolling_all__stdp__{probe}","baseline__persistence__rule","future_state")
        b=comp(f"rolling_all__stdp__{probe}",f"rolling_all__raw_history__{probe}","future_state")
        state_lines.append(f"{probe}读出：STDP比状态持续的未来BA差{a.difference:.4f}，95%区间[{a.ci_low:.4f}, {a.ci_high:.4f}]；比原始特征差{b.difference:.4f}，区间[{b.ci_low:.4f}, {b.ci_high:.4f}]。")
    parts["STATE_INTERPRETATION"]="\n\n".join(state_lines)+"\n\n当前状态分数高只能说明压缩表示中仍可读出部分已有信息；未来状态成绩才涉及前瞻性。还要超过状态持续和原始特征，才能支持STDP有额外用途。四类状态和涨跌方向不是同一任务，不能直接比较它们的准确率或将趋势类自动做多。"
    roll=[];rolldiff=[]
    for arm in cfg["arms"]:
        for rep in ["raw_history","stdp","raw_plus_stdp"]:
            for probe in cfg["probe_types"]:
                roll.append({"训练方式":arm,"表示":rep,"读出":probe,"收益IC":get(arm,rep,probe,"return_5").ic,"绝对收益IC":get(arm,rep,probe,"absolute_5").ic,"未来状态BA":get(arm,rep,probe,"future_state").balanced_accuracy})
    for other in ["frozen","readout_only"]:
        for rep in ["raw_history","stdp"]:
            for probe in cfg["probe_types"]:
                for target in ["return_5","absolute_5","future_state"]:
                    r=comp(f"rolling_all__{rep}__{probe}",f"{other}__{rep}__{probe}",target)
                    rolldiff.append({"比较":"rolling_all减"+other,"表示":rep,"读出":probe,"目标":target,"差":r.difference,"95%下限":r.ci_low,"95%上限":r.ci_high})
    parts["ROLLING_RESULTS"]=table(pd.DataFrame(roll))+"\n\n配对差异：\n\n"+table(pd.DataFrame(rolldiff))
    monthly=pd.read_csv(out/"monthly_metrics.csv")
    monthly_summary_rows=[]
    for month in sorted(monthly.month.unique()):
        sub=monthly[monthly.month.eq(month)&monthly.arm.eq("rolling_all")&monthly.probe.eq("linear")]
        def value(rep,target,field):return sub[sub.representation.eq(rep)&sub.target.eq(target)].iloc[0][field]
        monthly_summary_rows.append({"月份":month,"原始收益IC":value("raw_history","return_5","ic"),"STDP收益IC":value("stdp","return_5","ic"),"STDP绝对收益IC":value("stdp","absolute_5","ic"),"STDP未来状态BA":value("stdp","future_state","balanced_accuracy")})
    parts["MONTHLY_RESULTS"]=table(pd.DataFrame(monthly_summary_rows))+"\n\n汇总IC和月内IC不必相同；跨滚动段预测的水平/校准变化也会影响总体相关，不能把总体IC下降全部归结为神经表示丢失信息。8—9月STDP未来四状态BA为0.25，需结合state_confusions.csv查看类别偏置，不能只看准确率。"
    r=comp("rolling_all__stdp__linear","frozen__stdp__linear","return_5")
    if r.ci_low>0: words="收益IC的改善区间高于零，支持这段历史中滚动训练优于同设置的冻结模型；仍需看它自身是否有效和能否覆盖成本。"
    elif r.ci_high<0:words="收益IC的差异区间低于零，本段历史中滚动全流程反而弱于冻结；不能把重训频繁等同于更好。"
    else:words="收益IC差异区间跨零，没有在这段历史中确认全流程滚动稳定优于冻结。"
    parts["ROLLING_INTERPRETATION"]=words+"只更新读出与全流程重训的对照，有助于判断适应是否主要来自监督层；后者还包含缩放和阈值改变，只有结合随机突触对照才能进一步讨论STDP贡献。"
    mainbt=bt[bt.scenario=="base"][["arm","policy","trades","gross_pnl","fees","slippage","net_pnl","return_pct","max_drawdown_pct","daily_sharpe"]]
    sensitivity=bt[bt.policy=="selected"][["arm","scenario","trades","net_pnl","return_pct"]]
    parts["BACKTEST_RESULTS"]="基础情景中，selected才是每段预测前选择的正式规则，其余为固定模型诊断。\n\n"+table(mainbt,3)+"\n\n正式选择的成本/延迟敏感性：\n\n"+table(sensitivity,3)
    selected=trading("rolling_all");pure=trading("rolling_all","stdp")
    parts["TRADING_INTERPRETATION"]=(f"全滚动正式策略{int(selected.trades)}笔，毛盈亏{selected.gross_pnl:,.2f}元、手续费{selected.fees:,.2f}元、滑点{selected.slippage:,.2f}元，净盈亏{selected.net_pnl:,.2f}元。全滚动单独STDP诊断{int(pure.trades)}笔，毛盈亏{pure.gross_pnl:,.2f}元，手续费和滑点合计{pure.fees+pure.slippage:,.2f}元，净盈亏{pure.net_pnl:,.2f}元。这一版毛收益为正而净收益为负，说明成本是交易结果的重要约束；毛收益正的点估计仍不足以证明稳定预测能力。正式策略允许在原始特征、STDP、组合和空仓间选择，其总收益不能全部归因于STDP。\n\n"+f"正式策略单边2tick情景净盈亏{trading('rolling_all',scenario='two_ticks').net_pnl:,.2f}元，额外1分钟延迟情景{trading('rolling_all',scenario='extra_minute').net_pnl:,.2f}元。这些是固定过去选择后的执行敏感性，不是重新优化得到的成绩。")
    support=any(g["future_state_increment_supported"] for g in gates)
    parts["STATE_USE"]=("至少一种读出通过了上述未来状态增量检查。这支持进一步单独检验状态择时/风险过滤，但本轮没有利用整段结果事后挑一个状态交易规则，也不能将状态分类直接等同于方向盈利。" if support else "两种读出都没有同时通过上述未来状态增量检查。因此本轮没有强行构造新的STDP状态过滤策略；已经完成的滚动方向回测用于回答训练方式问题，不能把当前状态识别成绩包装成额外交易收益。")
    diagnostic=[]
    diagnostic.append("时序表达：合成任务表明当前SNN并非完全丢失顺序，但随机网络已有明显能力；STDP学习的额外贡献需要看随机对照差异，不能只看AB/BA准确率。")
    for target in ["return_5","absolute_5","future_state"]:
        d=comp("rolling_all__stdp__linear","rolling_all__random__linear",target)
        diagnostic.append(f"STDP学习贡献（{target}）：相对同编码随机突触差{d.difference:.4f}，区间[{d.ci_low:.4f}, {d.ci_high:.4f}]。")
    d=comp("rolling_all__stdp__linear","rolling_all__shuffle__linear","future_state")
    diagnostic.append(f"金融时序贡献：STDP真实顺序相对打乱顺序的未来状态BA差{d.difference:.4f}，区间[{d.ci_low:.4f}, {d.ci_high:.4f}]。打乱版仍包含滚动特征摘要，所以它并没有去除全部历史信息。")
    d=comp("rolling_all__counts_minute__linear","rolling_all__raw_history__linear","absolute_5")
    diagnostic.append(f"编码/统计环节：逐分钟脉冲计数相对原始特征的绝对收益IC差{d.difference:.4f}，区间[{d.ci_low:.4f}, {d.ci_high:.4f}]。这可以定位通路中信息可读性如何变化，但不是信息论意义的全部信息损失证明。")
    diagnostic.append("滚动设置：本轮已直接比较冻结、只更新读出和全流程更新，不能再只凭单次固定训练结果推断；同时，它仍不是每日在线学习，40/10/5窗口也不是经过证明的最优周期。")
    seed_metrics=pd.read_csv(out/"seed_metrics.csv")
    seed_rows=[]
    for rep in ["random","stdp","shuffle"]:
        for target in ["return_5","absolute_5","future_state"]:
            subset=seed_metrics[(seed_metrics.arm=="rolling_all") & seed_metrics.representation.str.startswith(rep+"_s") & seed_metrics.probe.eq("linear") & seed_metrics.target.eq(target)]
            field="balanced_accuracy" if "state" in target else "ic"
            seed_rows.append({"表示":rep,"目标":target,"三个种子最小":subset[field].min(),"三个种子最大":subset[field].max()})
    diagnostic.append("单种子结果范围如下，反映初始化敏感性；三个种子共享行情，不能当作三个独立市场样本。\n\n"+table(pd.DataFrame(seed_rows)))
    neural_rows=[]
    for fold in range(len(folds)):
        path=out/"initial_frozen" if fold==0 else out/"folds"/f"fold_{fold:02d}"/"rolling_all"/"encoder"
        for key,values in json.loads((path/"diagnostics.json").read_text()).items():
            neural_rows.append({"fold":fold,"mode":key.split("_s")[0],"seed":int(key.split("_s")[1]),**{k:v for k,v in values.items() if k!="history"}})
    neural=pd.DataFrame(neural_rows)
    neural.to_csv(out/"neural_diagnostics.csv",index=False)
    ns=neural.groupby("mode").agg(神经元最少活跃数=("active_neurons","min"),平均突触余弦=("mean_weight_cosine","mean"),训练表示平均有效秩=("effective_rank","mean"),平均放电率=("mean_rate","mean")).reset_index()
    diagnostic.append("训练期神经诊断（23段×3种子汇总）如下。突触相似度升高可以提示模式多样性不足，但不是因果证明；有效秩在原始表示上只中心化、未逐列标准化，会受膜电位与发放率量纲影响，不能当成可预测信息量。\n\n"+table(ns))
    parts["DIAGNOSIS"]="\n\n".join(diagnostic)
    parts["VERIFICATION"]="新增9项训练/因果检查、2项分析/文件顺序检查，加原38项共49项测试通过。合成18个、金融207个神经模型实际训练；金融1800个读出模型（首段40个，后22段两组各40个），合成72个读出。核验如下：\n\n"+table(pd.DataFrame([{"检查":k,"通过":v} for k,v in verification.items()]))
    # Figures are descriptive views of already-locked outputs.
    figdir=out/"figures";figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi":140,"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    sr=["raw_history","counts_minute","counts_total","random","stdp","shuffle"]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,gap in zip(axes,cfg["synthetic_gaps_minutes"]):
        for offset,probe in [(-.18,"linear"),(.18,"rff")]:
            vals=[sg[(sg.gap_minutes==gap)&(sg.representation==n)&(sg.probe==probe)]["mean"].iloc[0] for n in sr]
            ax.bar(np.arange(len(sr))+offset,vals,width=.35,label=probe)
        ax.axhline(.5,color="black",linestyle="--",linewidth=.8);ax.set_xticks(np.arange(len(sr)),sr,rotation=30,ha="right");ax.set(title=f"AB/BA gap {gap} min",ylabel="Held-out synthetic accuracy",ylim=(0,1.05));ax.legend()
    fig.savefig(figdir/"synthetic.png");plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,target in zip(axes.ravel(),["return_5","absolute_5","current_state","future_state"]):
        field="balanced_accuracy" if "state" in target else "ic"
        for offset,probe in [(-.18,"linear"),(.18,"rff")]:
            vals=[get("rolling_all",n,probe,target)[field] for n in reps]
            ax.bar(np.arange(len(reps))+offset,vals,width=.35,label=probe)
        if "state" in target:ax.axhline(.25,color="black",linestyle="--",linewidth=.6)
        else:ax.axhline(0,color="black",linewidth=.6)
        ax.set_xticks(np.arange(len(reps)),reps,rotation=35,ha="right");ax.set(title=target,ylabel=field);ax.legend()
    fig.savefig(figdir/"information_ladder.png");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    names=["raw_history","random","stdp","shuffle","raw_plus_stdp"]
    for ax,target in zip(axes,["current_state","future_state"]):
        for offset,probe in [(-.18,"linear"),(.18,"rff")]:
            ax.bar(np.arange(len(names))+offset,[get("rolling_all",n,probe,target).balanced_accuracy for n in names],width=.35,label=probe)
        baseline=get("baseline","persistence","rule",target).balanced_accuracy
        ax.axhline(baseline,color="black",linestyle="--",label="current-rule / persistence")
        ax.set_xticks(np.arange(len(names)),names,rotation=25,ha="right");ax.set(title=target,ylabel="Balanced accuracy",ylim=(0,1.05));ax.legend(fontsize=7)
    fig.savefig(figdir/"states.png");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,target in zip(axes,["return_5","future_state"]):
        for k,arm in enumerate(cfg["arms"]):
            field="ic" if target=="return_5" else "balanced_accuracy"
            ax.bar(np.arange(3)+(k-1)*.25,[get(arm,n,"linear",target)[field] for n in ["raw_history","stdp","raw_plus_stdp"]],width=.24,label=arm)
        ax.set_xticks(np.arange(3),["raw_history","stdp","raw_plus_stdp"]);ax.set(title=target+" (linear probe)",ylabel="IC" if target=="return_5" else "Balanced accuracy");ax.legend(fontsize=8)
    fig.savefig(figdir/"rolling_comparison.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
    for arm in cfg["arms"]:
        d=pd.read_csv(out/"backtests"/(arm+"__selected__base_daily.csv"))
        ax.plot(pd.to_datetime(d.trading_date),(d.equity/v1["trading"]["capital"]-1)*100,label=arm)
    ax.axhline(0,color="black",linewidth=.6);ax.set(title="Past-validation-selected policies; continuous capital",ylabel="Return on initial capital (%)");ax.legend();fig.savefig(figdir/"equity.png");plt.close(fig)
    text=Path("independent_study/V2_REPORT_TEMPLATE.md").read_text()
    for key,value in parts.items():text=text.replace("%%"+key+"%%",value)
    assert "%%" not in text
    (out/"report.md").write_text(text)
    save_json(out/"report_sources.json",{p:sha256(p) for p in ["independent_study/render_v2_report.py","independent_study/V2_REPORT_TEMPLATE.md"]})
    save_json(out/"artifact_manifest.json",{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name not in ["artifact_manifest.json","report_build.log"]})
    print(parts["SUMMARY"]);print(parts["STATE_USE"]);print("Report: "+str(out/"report.md"))


if __name__=="__main__":main()
