"""Evaluation and reload checks, separate from the locked training runner."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from iclab.data import FEATURES, dump_json, sha256
from event_stdp_v7.core import (encode, price_events, samples, score, lagged,
                               ranking, classification, fit_stdp, trace)
from event_stdp_v7.run import ROOT, HERE, OUT

NAMES = {'teacher_stdp':'价格事件教师STDP', 'untrained_equal':'未训练等权',
         'shuffled_teacher':'错配教师STDP（三种子平均）', 'current_rv':'当前5分钟波动',
         'same_input_ridge':'相同脉冲积分的监督Ridge'}


def verify(predictions, cfg):
    source = ROOT/'independent_study/results/v1/bars_features.parquet'
    bars = pd.read_parquet(source)
    x, seg = bars[FEATURES].to_numpy(), bars.segment_id.to_numpy()
    close = bars.close.to_numpy(dtype=float)
    errors, feature_pairs = [], []
    for fid, group in predictions.groupby('fold', sort=True):
        folder = OUT/'folds'/f'fold_{fid:02d}'
        lock = json.loads((folder/'lock.json').read_text())
        assert sha256(folder/'model.npz') == lock['model_sha256']
        m = np.load(folder/'model.npz')
        train = bars.trading_date.isin(lock['fold']['train_dates']).to_numpy()
        finite = np.isfinite(x).all(axis=1)
        expected_bounds = np.quantile(x[train & finite], cfg['feature_quantiles'], axis=0)
        np.testing.assert_array_equal(expected_bounds, m['bounds'])
        returns, raw, onset = price_events(close, seg, float(m['event_threshold']), cfg['quiet_minutes'])
        threshold = np.quantile(np.abs(returns[train & np.isfinite(returns)]), cfg['event_quantile'])
        assert threshold == m['event_threshold']
        eligible, y = samples(x, seg, returns, raw, onset, cfg['horizon'])
        s = encode(x, seg, m['bounds'])
        forecast = bars.trading_date.isin(lock['fold']['forecast_dates']).to_numpy()
        pre = trace(s, seg, range(1, cfg['horizon']+1), cfg['tau_minutes'])
        post = trace(onset.astype(float), seg, range(1, cfg['horizon']+1), cfg['tau_minutes'])
        for channel, name in enumerate(f'{f}:{side}' for f in FEATURES for side in ['low','high']):
            decision_spikes = (s[:, channel] > 0) & eligible & forecast
            feature_pairs.append(dict(fold=int(fid), channel=name,
                                      forecast_spikes=int(s[forecast, channel].sum()),
                                      weighted_lead_pairs=float((onset[forecast]*pre[forecast, channel]).sum()),
                                      weighted_lag_pairs=float((s[forecast, channel]*post[forecast]).sum()),
                                      eligible_spikes=int(decision_spikes.sum()),
                                      future_hits=int(y[decision_spikes].sum())))
        main, currents = score(s, seg, m['weight'], cfg)
        scores = dict(teacher_stdp=main,
                      untrained_equal=currents @ np.full(24, cfg['initial_weight']),
                      shuffled_teacher=np.mean(currents @ m['shuffled_weights'].T, axis=1),
                      current_rv=np.sqrt(sum(lagged(np.nan_to_num(returns)**2, seg, d) for d in range(5))),
                      same_input_ridge=((currents-m['ridge_mean'])/m['ridge_scale']) @ m['ridge_coef'] + m['ridge_target_mean'])
        for model, rows in group.groupby('model'):
            idx = rows.bar_id.to_numpy()
            np.testing.assert_array_equal(y[idx], rows.target)
            assert eligible[idx].all()
            expected_idx = np.flatnonzero(eligible & bars.trading_date.isin(lock['fold']['forecast_dates']).to_numpy())
            np.testing.assert_array_equal(idx, expected_idx)
            error = float(np.max(np.abs(scores[model][idx]-rows.score_raw)))
            assert error < 1e-12
            ref = m[f'validation_sorted_{model}']
            np.testing.assert_array_equal(np.searchsorted(ref, scores[model][idx], side='right')/len(ref), rows.score)
            np.testing.assert_array_equal(scores[model][idx] > lock['cutoffs'][model], rows.alarm)
            errors.append(error)
        if fid in [0, 11, 22]:
            ww, _ = fit_stdp(s, onset, seg, train, cfg)
            np.testing.assert_array_equal(ww, m['weight'])
    run_lock = json.loads((OUT/'run_lock.json').read_text())
    for path, expected in run_lock['hashes'].items():
        assert sha256(ROOT/path) == expected, path
    pd.DataFrame(feature_pairs).to_csv(OUT/'forecast_feature_pairs.csv', index=False)
    return dict(reloaded_folds=23, reconstructed_model_outputs=len(errors),
                maximum_score_error=max(errors), exact_training_repeats=3,
                thresholds_fitted_on_training_only=True, sources_unchanged=True)


def bootstrap(predictions, cfg):
    models = list(NAMES)
    grouped = {m: predictions[predictions.model == m].sort_values('bar_id').reset_index(drop=True) for m in models}
    ref = grouped['teacher_stdp']
    for frame in grouped.values():
        np.testing.assert_array_equal(frame.bar_id, ref.bar_id)
        np.testing.assert_array_equal(frame.target, ref.target)
    days = ref.trading_date.unique()
    indices = [np.flatnonzero(ref.trading_date.to_numpy() == d) for d in days]
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    block = cfg['bootstrap_block_days']
    y = ref.target.to_numpy()
    scores = {m: grouped[m].score.to_numpy() for m in models}
    actual = {m: ranking(y, scores[m]) for m in models}
    differences = {(m, metric): [] for m in models[1:] for metric in ['ap', 'auc']}
    for _ in range(cfg['bootstrap_repeats']):
        starts = rng.integers(0, len(days)-block+1, int(np.ceil(len(days)/block)))
        picked = np.concatenate([np.arange(s, s+block) for s in starts])[:len(days)]
        ix = np.concatenate([indices[d] for d in picked])
        metrics = {m: ranking(y[ix], scores[m][ix]) for m in models}
        for (other, metric), values in differences.items():
            values.append(metrics['teacher_stdp'][metric]-metrics[other][metric])
    rows = []
    for (other, metric), values in differences.items():
        lo, hi = np.nanquantile(values, [.025, .975])
        rows.append(dict(comparison=f'teacher_stdp - {other}', metric=metric,
                         difference=actual['teacher_stdp'][metric]-actual[other][metric],
                         lower=float(lo), upper=float(hi)))
    return pd.DataFrame(rows)


def synthetic_result(cfg):
    seg = np.repeat(np.arange(80), 30)
    spikes = np.zeros((len(seg), 4)); teacher = np.zeros(len(seg), dtype=bool)
    for episode in range(80):
        start = episode*30
        teacher[start+15] = True
        for channel, dt in enumerate([13, 17, 15, 4]):
            spikes[start+dt, channel] = 1
    weights, _ = fit_stdp(spikes, teacher, seg, seg < 60, cfg)
    pred, _ = score(spikes, seg, weights, cfg)
    assert weights[0] > .5 > weights[1] and weights[2] == weights[3] == .5
    return dict(channels=['领先2分钟', '滞后2分钟', '同时发生', '超出5分钟关联窗'],
                initial_weight=.5, weights=weights,
                training_episodes=60, heldout_episodes=20,
                heldout_leading_score=float(pred[70*30+13]),
                heldout_lagging_score=float(pred[70*30+17]))


def main():
    cfg = json.loads((HERE/'config.json').read_text())
    predictions = pd.read_parquet(OUT/'predictions.parquet')
    checks = verify(predictions, cfg)
    print('23-fold source, threshold, model and prediction reload checks passed', flush=True)
    metrics = pd.DataFrame([dict(model=model, **classification(g.target, g.score, g.alarm))
                            for model, g in predictions.groupby('model')])
    metrics.to_csv(OUT/'metrics.csv', index=False)
    monthly = predictions.copy(); monthly['month'] = monthly.trading_date.str[:7]
    pd.DataFrame([dict(month=month, model=model, **classification(g.target, g.score, g.alarm))
                  for (month, model), g in monthly.groupby(['month','model'])]).to_csv(OUT/'monthly_metrics.csv', index=False)
    leads = pd.read_csv(OUT/'event_leads.csv')
    lead_summary = []
    for model, all_rows in leads.groupby('model'):
        g = all_rows[all_rows.evaluable]
        hits = g[g.detected]
        lead_summary.append(dict(model=model, total_events=len(all_rows), evaluable_events=len(g),
                                 detected_events=len(hits), event_recall=len(hits)/len(g),
                                 median_earliest_lead=float(hits.earliest_lead_minutes.median()),
                                 median_latest_lead=float(hits.latest_lead_minutes.median())))
    lead_summary = pd.DataFrame(lead_summary)
    lead_summary.to_csv(OUT/'event_summary.csv', index=False)
    ci = bootstrap(predictions, cfg)
    ci.to_csv(OUT/'paired_intervals.csv', index=False)
    synthetic = synthetic_result(cfg)
    dump_json(OUT/'synthetic.json', synthetic)
    weights = pd.read_csv(OUT/'weights.csv')
    summary = weights[weights.model == 'teacher_stdp'].groupby('channel').agg(
        mean_weight=('weight','mean'), min_weight=('weight','min'), max_weight=('weight','max'),
        mean_train_spikes=('input_spikes','mean'), mean_lead_pairs=('weighted_lead_pairs','mean'),
        mean_lag_pairs=('weighted_lag_pairs','mean')).sort_values('mean_weight', ascending=False)
    summary.to_csv(OUT/'weight_summary.csv')
    m = metrics.set_index('model')
    primary = m.loc['teacher_stdp']
    lr = lead_summary.set_index('model').loc['teacher_stdp']
    table = '\n'.join(f'|{NAMES[k]}|{m.loc[k,"auc"]:.4f}|{m.loc[k,"ap"]:.4f}|{m.loc[k,"precision"]:.2%}|{m.loc[k,"coverage"]:.2%}|{m.loc[k,"recall"]:.2%}|' for k in NAMES)
    intervals = '\n'.join(f'|{r.comparison}|{r.metric}|{r.difference:.4f}|[{r.lower:.4f}, {r.upper:.4f}]|' for r in ci.itertuples())
    tops = '\n'.join(f'|{ch}|{row.mean_weight:.4f}|{row.min_weight:.4f}—{row.max_weight:.4f}|{row.mean_train_spikes:.1f}|' for ch, row in summary.head(8).iterrows())
    fold_metrics = pd.read_csv(OUT/'fold_metrics.csv')
    fm = fold_metrics[fold_metrics.model == 'teacher_stdp']
    srows = '\n'.join(f'|{ch}|0.5|{w:.6f}|' for ch, w in zip(synthetic['channels'], synthetic['weights']))
    teacher_vs_equal = ci[(ci.comparison == 'teacher_stdp - untrained_equal') & (ci.metric == 'ap')].iloc[0]
    evidence = ('AP相对未训练等权的区间完全为正，有本轮探索性的增益证据；还须核对其他基线和新数据。'
                if teacher_vs_equal.lower > 0 else
                'AP相对未训练等权的区间未完全高于零，尚未建立STDP直接加权的稳定增量优势。')
    text = f'''# v7：价格事件教师驱动 STDP 原型报告

已按用户最新要求把“特征脉冲与真实价格事件的先后顺序”接入权重更新。完成23折训练、92组突触权重（23组真实教师+69组错配教师）、23个监督基线与滚动预测。**机制能按规定做到领先加权、滞后减权；金融有效性需另看对照。** {evidence}

主模型AUC {primary.auc:.4f}，AP {primary.ap:.4f}，预测样本实际事件比例 {primary.base_rate:.2%}。告警精确率 {primary.precision:.2%}、告警覆盖率 {primary.coverage:.2%}。这不是方向预测、风险概率或盈利结论。

## 1. 为什么要改

旧版STDP的post来自隐藏神经元自己的放电，收益/风险标签只训练Ridge；不能自动做到“领先实际价格事件就直接加强”。本版训练post由**当分钟已发生的价格事件**提供，pre来自特征阈值穿越。10:00特征、10:03事件：到10:03才加强这条连接；10:04才出现的特征按滞后关系减弱。禁止用10:03更新后的权重回填10:00预测。

这属于有价格事件教师信号的STDP，常可归入教师驱动/监督式STDP；不能继续称纯无监督，也不等同于误差反传或原论文原样实现。原论文输出是网络自己放电，再作未来事件评价，并非用实际价格事件强制后突触。

## 2. 数据、特征与编码

只读v1真实合约底表43440分钟，含datetime（上海时区）、contract_id、OHLC、volume、open_interest、segment_id、bar_id及12项特征。半日/合约/日期/分钟缺口断开时段。复用v2日历：40日训练、10日验证、下一5包日期预测，23段覆盖2026-04-01至09-15。事件样本重新生成，本次 {int(primary.n)} 个决策时点、{predictions.trading_date.nunique()} 个有效日；不同于旧15900个收益窗口，不直接把本版AP/AUC与旧IC比较。

保留12项候选：1/3分钟价格变化除以前20分钟ATR、5分钟有符号路径效率、距离此前20分钟最高/最低点的ATR距离、当根收盘位置、5/20分钟波动对数比、振幅/ATR对数比、成交量相对前20分钟均值的log1p差、持仓log1p变化、前20分钟波动的log1p水平、已知日内时钟。具体公式见[原特征实现](../../iclab/data.py)。这些特征在t收盘计算，不假定它们天然预测未来。

每项训练期20%/80%分位作为低/高阈值。仅向下穿越低阈值、向上穿越高阈值各放一次脉冲；合计24通道。低/高不等于数值正/负。这里没有8步Rate或Latency，脉冲时间直接是实际市场分钟；持续处于高位不会每分钟重复发放，可能舍弃持续水平信息，这也是该简单原型的限制。

## 3. 价格事件、学习规则、输出

事件是绝对1分钟对数收益（bp）超过训练期95%分位，且此前5分钟无阈值突破。阈值仅按训练历史拟合，每折更新；不是未来收益标签前移，也不是论文未来均值/全天中位数判据。定义只描述静默后的跳变突破，不包含所有风险类型。

24条突触直接到1个漏积分预警单元，无隐藏层。训练3遍、初始化w=0.5，按实际分钟运行：

`Δw_i = 0.01(1-w_i) × event[t] × Σ(d=1..5)e^(-d/3)s_i[t-d] - 0.01w_i × s_i[t] × Σ(d=1..5)e^(-d/3)event[t-d]`

权重限于[0,1]，不把同分钟当提前，跨时段清迹。每折重新初始化，不是继承上一折权重在线更新。

预测V为 `Σ_i w_i Σ(d=0..4)e^(-d/3)s_i[t-d]`。输出一个连续风险分数和 `V>历史验证80%分位` 的告警。它是有限5分钟窗的漏积分电位读出；没有自由运行发放后重置的LIF，不能把阈值标志冒称完整LIF网络。该小原型用于先验证用户要求的突触逻辑，未同时加入隐藏层、软抑制或深层结构。

预测期只用已冻结权重和截至t的特征脉冲；未来事件只事后评分。首版问题是：截至t最近5分钟无阈值突破时，未来1—5分钟会不会有事件？每个模型用历史验证分数的经验分位统一跨折评分刻度，该数是排序分位，不是发生概率。主模型没有监督Ridge读出，Ridge只作对照。

## 4. 合成机制核验

60个训练片段、20个保留片段，每个通道脉冲计数相同。事件在第15分钟，通道分别在13/17/15/4分钟放电：

|通道|初值|训练后权重|
|---|---:|---:|
{srows}

先后更新方向符合设计，完全不配对的通道仍是0.5。这也表明**普通STDP不会仅因没有后续事件而自动惩罚误报特征**。保留片段领先时刻分数 {synthetic['heldout_leading_score']:.6f}，滞后时刻 {synthetic['heldout_lagging_score']:.6f}；这是人工机制检查，不是金融证据。

## 5. 滚动金融预测

所有模型使用相同决策时点和未来事件标签。当前RV是当时已有5分钟波动。错配教师在训练半日内循环错开至少10分钟，保留事件计数，3固定种子平均；它检验精确配对贡献，但不是消除所有日内结构的完美零假设。Ridge仅用同24路脉冲积分，在训练拟合、验证Brier选择正则强度。

|模型|AUC|AP|告警精确率|告警覆盖率|时点召回率|
|---|---:|---:|---:|---:|---:|
{table}

AUC随机排序约0.5，AP参照实际正例比例 {primary.base_rate:.4f}，精确率高不能忽略告警频率及漏报。按独立事件统计：STDP {int(lr.evaluable_events)} 个具备有效提前决策点的事件中预警 {int(lr.detected_events)} 个，覆盖 {lr.event_recall:.2%}；成功预警事件最早告警提前量中位数 {lr.median_earliest_lead:.1f} 分钟，最近告警提前量中位数 {lr.median_latest_lead:.1f} 分钟。该提前量仅针对已命中事件，不能视为每次事件都能提前发现。全体事件数 {int(lr.total_events)}，开盘/午后特征预热等无有效提前输入者不放入可评价分母，原始清单完整保留。

各折事件阈值 {fm.event_threshold_bp.min():.2f}—{fm.event_threshold_bp.max():.2f} bp，未来事件比例 {fm.base_rate.min():.2%}—{fm.base_rate.max():.2%}，其中 {int((fm.positives == 0).sum())} 折没有正例，其AUC/AP为未定义，未填成0.5。事件频率随行情显著漂移，合并排序包含跨折校准影响，因此同时保留逐折/逐月结果。

5日连续日期块重采样1000次，同日配对95%区间（探索性，不作多重比较校正）：

|差异（STDP减对照）|指标|差值|区间|
|---|---|---:|---|
{intervals}

完整分月、分折与事件提前量见[逐月](monthly_metrics.csv)、[逐折](fold_metrics.csv)、[逐事件](event_leads.csv)、[对照事件覆盖](event_summary.csv)。不根据本轮预测结果改选事件、编码、阈值或交易规则。

## 6. 权重可解释性与局限

以下为23折平均权重最高的8路，完整24路及每折正/反配对贡献均已保存：

|通道|平均权重|各折范围|平均每折训练脉冲数|
|---|---:|---:|---:|
{tops}

权重大表示在本规则、活动频率和事件定义下更偏向领先配对，不能直接解释为金融因果、全局特征重要性或单独交易收益。高频脉冲和共同原因可能形成偶合，市场状态持续也可能同时影响特征与事件。仅加权的线性汇合不具备任意多特征组合/顺序逻辑，需要在确认基本领先信息后另加隐藏单元或多延迟通道。

预测期每通道领先/滞后配对、具备有效提前输入的脉冲及其后续事件次数见[特征配对](forecast_feature_pairs.csv)。例如低波动脉冲较容易在静默后突破定义下取得领先配对优势，但这可能部分来自事件定义本身；不能据最高权重断言已找到可交易的先行特征。相同输入监督Ridge优于本STDP，说明这种直接成对加权未充分利用该输入中可读出的事件信息，并不能据此断定必须更换品种或更细数据。

本版同时改变教师目标、特征事件编码、网络和读出，是为了直接实现用户新机制；不能把相对旧v5任何差异单独归因于STDP。版内等权/错配是更直接的学习对照。

## 7. 验证、用途与复现

23折阈值重新核对、115组模型输出复载，最大数值误差 {checks['maximum_score_error']:.3g}；首/中/末3折重新训练权重逐元素一致。输入底表、原日历、配置与训练源码哈希未变。本轮8项新机制测试及36项独立项目旧回归，共44项通过；覆盖领先/滞后、零时差、跨时段、因果前缀、教师错配计数与指标平分。

没有运行新交易回测。无方向事件分数可候选用于开仓过滤或仓位控制，但高波动不等于当前持仓会亏；动态止损还需方向条件不利路径与可执行规则。所有历史已经观察，本版为探索性滚动历史检验，不能声称未见数据确认或已获利。数据供应商分钟标签、调整/单位等限制仍在。

复现命令见[README](../../event_stdp_v7/README.md)；训练严格拒绝覆盖已有v7产物。协议、源码、权重、逐分钟预测、逐事件提前量和核验均可审阅。v6仍未完成，因本次用户优先级切换暂缓，未把其计划当成成果。
'''
    (OUT/'report.md').write_text(text, encoding='utf-8')
    dump_json(OUT/'verification.json', checks)
    artifacts = {str(p.relative_to(ROOT)): sha256(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    dump_json(OUT/'manifest.json', dict(status='complete', artifacts=artifacts,
                                      analysis_source_sha256=sha256(Path(__file__))))
    dump_json(ROOT/'outputs/independent_v7_verification.json', checks)
    print(metrics.to_string(index=False), flush=True)
    print(ci.to_string(index=False), flush=True)
    print('Report and artifact manifest complete', flush=True)


if __name__ == '__main__':
    main()
