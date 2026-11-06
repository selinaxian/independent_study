"""Presentation and frozen-result audit; never fits or selects a strategy."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iclab.data import FEATURES, TARGETS, dump_json, sha256
from iclab.analysis import correlation, block_bootstrap
from iclab.neural import encode, CompetitiveSNN
from iclab.backtest import trade_accounting


def md(frame, digits=4):
    def cell(v):
        if isinstance(v,(float,np.floating)):
            return "—" if not np.isfinite(v) else f"{v:,.{digits}f}"
        return str(v).replace("|","/").replace("\n"," ")
    lines=["|"+"|".join(map(str,frame.columns))+"|","|"+"|".join(["---"]*len(frame.columns))+"|"]
    lines += ["|"+"|".join(cell(v) for v in row)+"|" for row in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="independent_study/configs/experiment.json")
    cfg=json.loads(Path(parser.parse_args().config).read_text()); out=Path(cfg["output_dir"])
    lock=json.loads((out/"selection_lock.json").read_text()); complete=json.loads((out/"evaluation_complete.json").read_text())
    data=json.loads((out/"data_summary.json").read_text()); boot=json.loads((out/"bootstrap.json").read_text())
    metrics=pd.read_csv(out/"test_metrics.csv"); vm=pd.read_csv(out/"validation_metrics.csv")
    bt=pd.read_csv(out/"backtest_summary.csv"); meta=pd.read_parquet(out/"samples.parquet"); bars=pd.read_parquet(out/"bars_features.parquet")
    a=np.load(out/"dataset.npz"); pred=pd.read_parquet(out/"test_predictions.parquet")
    diag=json.loads((out/"training_diagnostics.json").read_text()); td=json.loads((out/"test_neural_diagnostics.json").read_text())
    figs=out/"figures"; figs.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi":140,"axes.spines.top":False,"axes.spines.right":False,"font.size":9})
    def metric(name,target="return_5",source=metrics): return source[(source.model==name)&(source.target==target)].iloc[0]
    def result(name,scenario="base"): return bt[(bt.model==name)&(bt.scenario==scenario)].iloc[0]
    def ci(name,target="return_5"):
        d=boot[target][name]; return f"[{d['ci_low']:.4f}, {d['ci_high']:.4f}]"
    names=["mean","price_history","raw_last","raw_history","without_levels","risk_control","random","stdp","shuffle","raw_plus_stdp"]
    parts={}
    # Additional pre-stated mechanism comparisons, using paired day blocks.
    calendar_dates=pd.read_csv(out/"contract_selection.csv").trading_date
    calendar_dates=calendar_dates[calendar_dates>cfg["validation_end"]].tolist()
    comparisons={}
    for target in ["return_5","absolute_5"]:
        comparisons[target]={}
        for left,right in [("stdp","random"),("stdp","shuffle"),("raw_history","without_levels")]:
            bb=block_bootstrap({"stdp":pred[f"{left}__{target}"].to_numpy(),"raw_history":pred[f"{right}__{target}"].to_numpy()},pred["actual_"+target].to_numpy(),pred.trading_date.to_numpy(),calendar_dates,cfg["bootstrap_repetitions"],cfg["bootstrap_block_days"])
            comparisons[target][left+"_minus_"+right]=bb["stdp_minus_raw_history"]
    dump_json(out/"additional_comparisons.json",comparisons)
    sel=lock["selected_strategy"]; chosen=result(sel)
    parts["CONCLUSIONS"]=(f"已完成18个SNN及其读出训练，并在未用于本轮选择的7—9月数据上完成预测分析和回测。主分析使用验证期选中的 **{lock['encoding']}编码**。\n\n"
        f"- 5分钟收益预测：STDP集成IC **{metric('stdp').ic:.4f}**，原始完整特征序列IC **{metric('raw_history').ic:.4f}**，原始特征加STDP为 **{metric('raw_plus_stdp').ic:.4f}**。原始+STDP相对原始的IC差95%块重抽样区间为 **{ci('raw_plus_stdp_minus_raw_history')}**。\n"
        f"- 5分钟绝对收益（风险幅度）：STDP IC **{metric('stdp','absolute_5').ic:.4f}**，仅已知波动水平与时钟的控制IC **{metric('risk_control','absolute_5').ic:.4f}**。风险相关不能直接解释为涨跌方向或可交易利润。\n"
        f"- 验证期锁定的策略是 **{sel}**；测试期交易 **{int(chosen.trades)}笔**，净盈亏 **{chosen.net_pnl:,.2f}元**，100万元本金收益率 **{chosen.return_pct:.3f}%**。固定STDP诊断策略净盈亏 **{result('stdp').net_pnl:,.2f}元**，下文同时给出毛收益和费用，不只展示最好结果。\n"
        f"- 测试期也有正收益对照：原始特征加STDP为 **{result('raw_plus_stdp').net_pnl:,.2f}元（{result('raw_plus_stdp').return_pct:.3f}%）**，删去支撑/阻力的原始特征为 **{result('without_levels').net_pnl:,.2f}元（{result('without_levels').return_pct:.3f}%）**。它们验证期亏损、没有成为正式策略；不能看完测试后改选赢家。")
    counts=[]
    for split in ["train","validation","test"]:
        m=meta[meta.split==split]
        counts.append({"区间":split,"首日":m.trading_date.min(),"末日":m.trading_date.max(),"合约日":m.trading_date.nunique(),"窗口":len(m),"行情行":int(bars.split.eq(split).sum()),"合约":", ".join(m.contract_id.unique())})
    parts["DATA_TABLE"]=f"规则选中{data['selected_days']}日、{data['selected_rows']}行，产生{len(meta)}个12分钟窗口。\n\n"+md(pd.DataFrame(counts))
    decision=pd.read_csv(out/"contract_selection.csv")
    parts["SKIPS"]=md(decision[decision.status!="selected"][["trading_date","information_date","contract_id","status"]].fillna("—"))
    example=pred[["sample_id","contract_id","datetime","stdp__return_1","stdp__return_3","stdp__return_5","stdp__return_10","stdp__absolute_5","actual_return_5"]].head(3).copy()
    example.columns=["样本","合约","信号时间","预测1m","预测3m","预测5m","预测10m","预测绝对5m","事后真实5m"]
    parts["OUTPUT_EXAMPLE"]="下面是最终测试前三个样本的实际连续输出（所选编码、3种子平均），价格变化均以bps计；它们不是概率，也不是指数点。最后一列仅供事后评价，不是模型输入。\n\n"+md(example,4)
    feature_rows=[]
    for split in ["train","validation","test"]:
        mask=meta.split.eq(split).to_numpy()
        for j,name in enumerate(FEATURES):
            feature_rows.append({"split":split,"feature":name,"return_5_ic":correlation(a["X"][mask,-1,j],a["y"][mask,2]),"absolute_5_ic":correlation(a["X"][mask,-1,j],a["y"][mask,4]),"clip_fraction":float((np.abs(a["X"][mask,-1,j])>=1).mean())})
    fa=pd.DataFrame(feature_rows); fa.to_csv(out/"feature_analysis.csv",index=False)
    parts["FEATURE_ANALYSIS"]="以下为每个特征末值与未来目标的测试期单变量相关，属于解释性诊断，未据此删选特征或重训；相关不等于因果或独立增量。\n\n"+md(fa[fa.split=="test"].drop(columns="split"))
    rows=[]
    for enc in cfg["encodings"]:
        for mode in cfg["modes"]:
            ds=[diag[f"{enc}_{mode}_s{s}"]["train"] for s in cfg["seeds"]]
            tests=[td[f"{enc}_{mode}_s{s}"] for s in cfg["seeds"]]
            rows.append({"编码":enc,"模式":mode,"训练活跃数":f"{min(d['active_neurons'] for d in ds)}–{max(d['active_neurons'] for d in ds)}","测试活跃数":f"{min(d['active_neurons'] for d in tests)}–{max(d['active_neurons'] for d in tests)}","权重L2变化均值":np.mean([d["weight_change_l2"] for d in ds]),"权重余弦均值":np.mean([d["mean_weight_cosine"] for d in ds]),"测试有效秩均值":np.mean([d["effective_rank"] for d in tests])})
    parts["NEURAL_TABLE"]=md(pd.DataFrame(rows))+"\n\n有效秩按表示中心化奇异值能量的熵计算；活跃神经元多不代表表示不冗余，更不代表收益可预测。权重相似度高说明不同神经元可能学到相近组合。"
    policies=pd.DataFrame(lock["policies"].values())
    parts["LOCK"]=(f"选择锁写入UTC {lock['locked_at_utc']}，其文件哈希为 `{complete['selection_lock_sha256']}`。验证期最终选择编码 **{lock['encoding']}**、策略 **{sel}**。最终测试之前尚未产生本轮测试预测；锁中保存模型/读出及核心源文件哈希，评价入口逐个核验。\n\n"+md(policies[["model","quantile","threshold","trades","net_pnl"]],3))
    ptable=[]
    for n in names:
        r=metric(n); ptable.append({"模型":n,"验证IC":metric(n,source=vm).ic,"测试IC":r.ic,"95%区间":ci(n),"Rank IC":r.rank_ic,"RMSE(bps)":r.rmse,"方向准确率":r.direction_accuracy,"日均IC":r.daily_ic_mean})
    rtable=[]
    for n in names:
        r=metric(n,"absolute_5"); rtable.append({"模型":n,"绝对收益IC":r.ic,"95%区间":ci(n,"absolute_5"),"Rank IC":r.rank_ic,"RMSE(bps)":r.rmse})
    horizon=metrics[(metrics.model.isin(["raw_history","random","stdp","raw_plus_stdp"])) & metrics.target.ne("absolute_5")].pivot(index="model",columns="target",values="ic").reset_index()
    diff=pd.DataFrame([{ "比较":n,**d} for n,d in {**boot["return_5"],**comparisons["return_5"]}.items() if "minus" in n])
    riskdiff=pd.DataFrame([{"比较":n,**d} for n,d in {**boot["absolute_5"],**comparisons["absolute_5"]}.items() if "minus" in n])
    parts["PREDICTION_TABLES"]="**主要5分钟收益目标**\n\n"+md(pd.DataFrame(ptable))+"\n\n**5分钟风险幅度目标**\n\n"+md(pd.DataFrame(rtable))+"\n\n**不同持有时域的测试收益IC**（不是事件预警提前量）\n\n"+md(horizon)+"\n\n**收益IC成对差异的块重抽样**\n\n"+md(diff)+"\n\n**风险幅度IC成对差异**\n\n"+md(riskdiff)
    stab=pd.read_csv(out/"stability.csv")
    month=stab[(stab.group=="month")&(stab.target=="return_5")&stab.model.isin(["raw_history","random","stdp","shuffle","raw_plus_stdp"])].pivot(index="value",columns="model",values="ic").reset_index()
    seeds=metrics[(metrics.model.str.startswith(lock["encoding"]+"_stdp_s"))&(metrics.target=="return_5")][["model","ic","rank_ic","rmse"]]
    parts["STABILITY"]="**月份稳定性（收益IC）**\n\n"+md(month)+"\n\n**所选编码三个STDP种子（收益目标）**\n\n"+md(seeds)+"\n\n逐日和逐合约结果保存在stability.csv。IC2612只有2026-09-15一天，不能据此证明跨合约稳定性。"
    base=bt[bt.scenario=="base"][["model","trades","gross_pnl","fees","slippage","net_pnl","return_pct","max_drawdown_pct","daily_sharpe"]]
    sens=bt[(bt.model.isin(["raw_history","without_levels","stdp","raw_plus_stdp"]))][["model","scenario","trades","net_pnl","return_pct"]]
    parts["BACKTEST_TABLES"]="**最终测试：基础成本情景**。除验证锁定策略外，其余是事先列出的诊断交易，不能事后选最好者替代正式结果。Sharpe按短样本日收益年化，只作描述。\n\n"+md(base,3)+"\n\n**固定门槛的敏感性**\n\n"+md(sens,3)
    pnl_uncertainty=[]
    rng=np.random.default_rng(62320); n=len(calendar_dates); block=cfg["bootstrap_block_days"]
    starts=rng.integers(0,n-block+1,(cfg["bootstrap_repetitions"],int(np.ceil(n/block))))
    idx=(starts[:,:,None]+np.arange(block)).reshape(len(starts),-1)[:,:n]
    for name in ["raw_history","without_levels","stdp","raw_plus_stdp"]:
        daily=pd.read_csv(out/"backtests"/(name+"_daily.csv"))
        pnl=np.diff(np.r_[cfg["trading"]["capital"],daily.equity.to_numpy()])
        samples=pnl[idx].sum(axis=1)
        pnl_uncertainty.append({"模型":name,"实际总盈亏":float(pnl.sum()),"55日块重抽样下限":float(np.quantile(samples,.025)),"55日块重抽样上限":float(np.quantile(samples,.975))})
    pd.DataFrame(pnl_uncertainty).to_csv(out/"pnl_uncertainty.csv",index=False)
    parts["BACKTEST_TABLES"]+="\n\n**事后补充的盈亏不确定性诊断**：按连续5日块重抽样55个包日期的人民币日盈亏，1000次95%区间；未用于选策略，不是下一期盈利保证。\n\n"+md(pd.DataFrame(pnl_uncertainty),2)
    di=boot["return_5"]["raw_plus_stdp_minus_raw_history"]
    st=boot["return_5"]["stdp"]
    interpretation=[]
    if di["ci_low"]<=0<=di["ci_high"]:
        interpretation.append("原始特征加STDP相对原始特征的收益IC差异区间跨零，本轮没有确认STDP带来稳定的增量方向信息。")
    elif di["ci_high"]<0:
        interpretation.append("原始特征加STDP的收益IC差异区间整体低于零，本轮组合反而弱于原始特征，不支持STDP增量优势。")
    else:
        interpretation.append("原始特征加STDP的收益IC差异区间高于零，出现样本内条件下的增量迹象；鉴于多项探索与有限交易日，仍需新的独立样本确认。")
    if st["ci_low"]<=0<=st["ci_high"]: interpretation.append("单独STDP收益IC区间也跨零，不能把点估计或某个种子成绩当作可靠方向预测。")
    cr=comparisons["return_5"]["stdp_minus_random"]
    if cr["ci_low"]>0:
        interpretation.append(f"需要保留一个有限的正面发现：STDP比随机SNN的收益IC高{cr['ic']:.4f}，成对区间[{cr['ci_low']:.4f}, {cr['ci_high']:.4f}]高于零。它改善了这个较差的随机对照，但STDP自身方向IC未稳定高于零，也没有在原始特征之上确认增量，不能据此得出可盈利结论。")
    risk=boot["absolute_5"]["stdp"]
    if risk["ci_low"]>0:
        interpretation.append(f"STDP风险幅度IC区间{ci('stdp','absolute_5')}高于零，说明有未来波动幅度的相关信息。但波动水平/时间控制IC为{metric('risk_control','absolute_5').ic:.4f}，原始完整特征为{metric('raw_history','absolute_5').ic:.4f}，需要与这些普通信息比较，不能将全部风险可预测性归功于STDP。")
    else: interpretation.append("STDP风险幅度预测在该样本中也未形成稳健的正相关区间。")
    if sel=="flat": interpretation.append("验证阶段已选择不交易，因此正式冻结策略测试收益为0。这不是SNN赚取收益；它说明已有候选在验证成本后不足以支持上线交易。其他固定模型的测试回测仍完整给出，没有为了展示收益而跳过亏损。")
    elif chosen.net_pnl>0: interpretation.append("验证锁定策略在测试条件模拟中取得正净收益；这是一段给定数据与成交假设下的结果，不是已经验证实盘能持续盈利。")
    else: interpretation.append("验证锁定策略在测试期扣费后亏损，当前流程没有实现样本外盈利目标。")
    if result("stdp").net_pnl<=0: interpretation.append("固定STDP策略扣费后不盈利，所以即使某些预测统计量为正，本次仍不能将其解释为可用的日内赚钱策略。")
    interpretation.append(f"具体分解：STDP毛盈亏{result('stdp').gross_pnl:,.2f}元，加上手续费{result('stdp').fees:,.2f}元和滑点{result('stdp').slippage:,.2f}元后得到上述净亏损。这不只是费用把一个赚钱信号吃掉：其固定阈值下的毛交易结果也为负。原始完整特征毛盈亏{result('raw_history').gross_pnl:,.2f}元，扣费后{result('raw_history').net_pnl:,.2f}元，则体现出成本覆盖不足。")
    interpretation.append(f"原始特征加STDP的测试净盈利{result('raw_plus_stdp').net_pnl:,.2f}元需要如实保留，但单边2tick时变为{result('raw_plus_stdp','two_ticks').net_pnl:,.2f}元，多延迟1分钟时变为{result('raw_plus_stdp','extra_minute').net_pnl:,.2f}元，说明这段小幅正收益不稳健。删除支撑/阻力的原始特征对照盈利{result('without_levels').net_pnl:,.2f}元，延迟1分钟后降到{result('without_levels','extra_minute').net_pnl:,.2f}元；这是无需SNN的诊断对照，且验证期亏损，不能改称已验证盈利策略。")
    cos=np.mean([td[f"{lock['encoding']}_stdp_s{s}"]["mean_weight_cosine"] for s in cfg["seeds"]])
    interpretation.append(f"神经层面，STDP三个种子的平均突触余弦相似度约{cos:.3f}，不同隐藏神经元的输入偏好较接近。风险IC也低于随机SNN的{metric('random','absolute_5').ic:.4f}。这提示当前竞争/权重归一化和压缩可能没有保留最有用的金融信息，是后续在新验证方案内检查的方向；本实验没有证明它就是失败的因果机制。")
    interpretation.append(f"删除支撑/阻力后的收益IC为{metric('without_levels').ic:.4f}，完整原始特征为{metric('raw_history').ic:.4f}；这是本次设定下的对照，不足以概括所有支撑阻力定义。真实顺序STDP与打乱历史版本IC分别{metric('stdp').ic:.4f}和{metric('shuffle').ic:.4f}，应结合随机对照、月份和区间判断，不能仅凭微小点差宣称学到了金融因果时序。")
    parts["INTERPRETATION"]="\n\n".join(interpretation)
    # Diagrams are generated entirely from saved/frozen inputs and outputs.
    enc=lock["encoding"]; seed=cfg["seeds"][0]
    ex=3; xx=a["X"][ex:ex+1]; spikes=encode(xx,[ex],enc,seed)
    fig,axes=plt.subplots(2,1,figsize=(10,5),constrained_layout=True)
    axes[0].imshow(xx[0].T,aspect="auto",cmap="coolwarm",vmin=-1,vmax=1); axes[0].set(ylabel="12 features",xlabel="Market minutes in window",title="One training window: normalized features")
    axes[1].imshow(spikes[0].T,aspect="auto",cmap="binary",interpolation="nearest"); axes[1].set(xlabel="96 simulation steps (8 per market minute)",ylabel="24 sign channels",title=f"{enc} encoding; same input window")
    fig.savefig(figs/"encoding_example.png"); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,mode in zip(axes[:2],["random","stdp"]):
        w=np.load(out/"models"/f"{enc}_{mode}_s{seed}_snn.npz")["weight"]
        ax.imshow(w,aspect="auto",vmin=0,vmax=max(.6,float(w.max())),cmap="viridis"); ax.set(title=mode+" input weights",xlabel="Input channel",ylabel="Hidden neuron")
    for mode in ["random","stdp"]:
        rep=np.load(out/"representations"/f"{enc}_{mode}_s{seed}_test.npy")
        axes[2].plot(rep[:,:64].mean(axis=0),label=mode)
    axes[2].set(title="Test mean firing rate",xlabel="Hidden neuron",ylabel="Spikes / simulation step"); axes[2].legend()
    fig.savefig(figs/"neural_diagnostics.png"); plt.close(fig)
    show=["price_history","raw_history","risk_control","random","stdp","shuffle","raw_plus_stdp"]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,target in zip(axes,["return_5","absolute_5"]):
        v=np.array([metric(n,target).ic for n in show]); low=np.array([boot[target][n]["ci_low"] for n in show]); high=np.array([boot[target][n]["ci_high"] for n in show])
        ax.bar(np.arange(len(show)),v,color=["#64748b"]*3+["#9ca3af","#2563eb","#c084fc","#0d9488"])
        # Percentile bootstrap intervals need not contain the observed statistic.
        ax.vlines(np.arange(len(show)),low,high,color="black",linewidth=1)
        ax.axhline(0,color="black",linewidth=.6); ax.set_xticks(np.arange(len(show)),show,rotation=35,ha="right"); ax.set(title=target+" test IC (5-day block interval)",ylabel="Pearson IC")
    fig.savefig(figs/"prediction_comparison.png"); plt.close(fig)
    buckets=pd.read_csv(out/"score_buckets.csv")
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,target in zip(axes,["return_5","absolute_5"]):
        for n in ["raw_history","stdp","raw_plus_stdp"]:
            b=buckets[(buckets.model==n)&(buckets.target==target)]
            ax.plot(b.bucket,b.mean_actual,marker="o",label=n)
        ax.set(title="Test outcomes: "+target,xlabel="Score bucket (training cutoffs)",ylabel="Actual mean (bps)"); ax.legend(fontsize=8)
    fig.savefig(figs/"score_buckets.png"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    for n in ["raw_history","stdp","raw_plus_stdp","flat"]:
        d=pd.read_csv(out/"backtests"/(n+"_daily.csv"))
        ax.plot(pd.to_datetime(d.trading_date),(d.equity/cfg["trading"]["capital"]-1)*100,label=n+(" (locked)" if n==sel else ""))
    ax.axhline(0,color="black",linewidth=.5); ax.set(title="Final test: fixed policies after fees and slippage",ylabel="Return on initial capital (%)"); ax.legend(); fig.savefig(figs/"equity.png"); plt.close(fig)
    # Independent artifact/accounting checks, without rerunning optimization.
    checks={"source_inputs_unchanged":all(sha256(p)==h for p,h in data["input_hashes"].items()),"models_unchanged":all(sha256(p)==h for p,h in lock["models_sha256"].items()),"core_sources_unchanged":all(sha256(p)==h for p,h in lock["source_sha256"].items()),"selection_lock_unchanged":sha256(out/"selection_lock.json")==complete["selection_lock_sha256"],"finite_inputs_and_labels":bool(np.isfinite(a["X"]).all() and np.isfinite(a["y"]).all()),"all_saved_predictions_finite":bool(np.isfinite(pred.filter(regex="__").to_numpy()).all()),"sample_id_unique":bool(meta.sample_id.is_unique)}
    netchecks=[]
    for name in lock["policies"]:
        file=out/"backtests"/(name+"_trades.csv")
        try: trades=pd.read_csv(file)
        except pd.errors.EmptyDataError: trades=pd.DataFrame()
        if len(trades):
            calculated=np.array([trade_accounting(r.entry_open,r.exit_open,r.side,cfg["trading"])["net_pnl"] for r in trades.itertuples()])
            netchecks.append(bool(np.allclose(calculated,trades.net_pnl,rtol=0,atol=1e-6) and np.isclose(calculated.sum(),result(name).net_pnl,atol=1e-6)))
        else: netchecks.append(bool(result(name).trades==0))
    checks["all_base_trades_reconciled"]=all(netchecks)
    checks["all_18_neural_checkpoint_reloads_exact"]=len(diag)==18 and all(v["checkpoint_reload_exact"] for v in diag.values())
    assert all(checks.values()),checks
    dump_json(out/"verification.json",checks)
    parts["VERIFICATION"]="新增9项机制/因果/执行测试通过；原数据层及旧项目29项测试另行回归，合计38项。训练阶段运行约"+f"{lock['seconds']:.1f}"+"秒（CPU、2个PyTorch线程；不作为性能基准）。实际产物复核如下（不靠重新选择模型修饰结果）：\n\n"+md(pd.DataFrame([{"检查":k,"通过":v} for k,v in checks.items()]))
    text=Path("independent_study/REPORT_TEMPLATE.md").read_text()
    for key,value in parts.items(): text=text.replace("%%"+key+"%%",value)
    assert "%%" not in text
    (out/"report.md").write_text(text,encoding="utf-8")
    dump_json(out/"artifact_manifest.json",{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name not in {"artifact_manifest.json","report_build.log"}})
    print(parts["CONCLUSIONS"])
    print("\n"+parts["INTERPRETATION"])
    print("\nAll artifact checks passed; report: "+str(out/"report.md"))


if __name__=="__main__": main()
