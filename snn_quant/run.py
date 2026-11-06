"""Run the complete first experiment: python -m snn_quant.run --config ..."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

# Limit this small experiment's CPU use on the shared research machine.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch

from .evaluation import evaluate, select_ridge
from .features import (FEATURE_DESCRIPTIONS, FEATURE_NAMES, build_features,
                       build_windows, fit_scales, load_bars, normalize)
from .network import CompetitiveSTDP, NetworkConfig


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def make_plots(out, histories, trained_networks, metrics, model_order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, records in histories.items():
        if not name.startswith("stdp_"):
            continue
        epoch = [r["epoch"] for r in records]
        axes[0].plot(epoch, [r["weight_relative_change"] for r in records], marker="o", label=name)
        axes[1].plot(epoch, [r["mean_spikes_per_sample"] for r in records], marker="o", label=name)
        axes[2].plot(epoch, [r["active_neurons"] for r in records], marker="o", label=name)
    axes[0].set_title("STDP weight change from initialization")
    axes[1].set_title("Spikes per 10-minute window")
    axes[2].set_title("Active hidden neurons during training")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training_diagnostics.png", dpi=160)
    plt.close(fig)

    first = trained_networks[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    channels = [f"{name}{sign}" for name in FEATURE_NAMES for sign in ("+", "-")]
    im = axes[0].imshow(first.weight.numpy(), aspect="auto", vmin=0, cmap="viridis")
    axes[0].set_title(f"Trained synapses (seed {first.seed})")
    fig.colorbar(im, ax=axes[0], shrink=0.6)
    delta = (first.weight - first.initial_weight).numpy()
    bound = max(float(np.abs(delta).max()), 1e-6)
    im = axes[1].imshow(delta, aspect="auto", vmin=-bound, vmax=bound, cmap="coolwarm")
    axes[1].set_title("Weight change from initialization")
    fig.colorbar(im, ax=axes[1], shrink=0.6)
    for axis in axes:
        axis.set_xticks(range(len(channels)), channels, rotation=80, fontsize=7)
        axis.set_ylabel("Hidden LIF neuron")
    fig.tight_layout()
    fig.savefig(out / "synaptic_weights.png", dpi=160)
    plt.close(fig)

    names = [name for name in model_order if name != "train_mean"]
    fig, ax = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(names))
    for split, offset, color in [("validation", -0.18, "#5470c6"), ("test", 0.18, "#ee6666")]:
        scores = [metrics[name][split]["pearson_ic"] or 0 for name in names]
        ax.bar(positions + offset, scores, width=0.36, label=split, color=color)
    ax.set_xticks(positions, names, rotation=25, ha="right")
    ax.set_ylabel("Pearson IC with future 5-minute return")
    ax.set_title("Forecast evaluation; not executable trading P&L")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "forecast_comparison.png", dpi=160)
    plt.close(fig)


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def write_report(out, cfg, metadata, metrics, histories, model_order, manifest):
    split_lines = []
    for split in ("train", "validation", "test"):
        part = metadata[metadata.split == split]
        split_lines.append(f"| {split} | {part.trading_date.min()} 至 {part.trading_date.max()} | "
                           f"{part.trading_date.nunique()} | {len(part)} |")
    metric_lines = []
    for name in model_order:
        validation, test = metrics[name]["validation"], metrics[name]["test"]
        metric_lines.append(f"| {name} | {fmt(validation['pearson_ic'])} | {fmt(test['pearson_ic'])} | "
                            f"{fmt(test['spearman_ic'])} | {fmt(test['rmse_bps'], 3)} | "
                            f"{fmt(test['direction_accuracy'])} | {fmt(test['daily_ic_mean'])} |")
    train_lines = []
    for name, records in histories.items():
        if name.startswith("stdp_"):
            r = records[-1]
            train_lines.append(f"| {name} | {r['epoch']} | {r['active_neurons']} / {cfg['network']['hidden']} | "
                               f"{r['mean_spikes_per_sample']:.2f} | {r['weight_relative_change']:.4f} | "
                               f"{r['mean_pairwise_weight_cosine']:.4f} |")
    feature_lines = [f"| {name} | {description} |" for name, description in FEATURE_DESCRIPTIONS.items()]
    stdp_ic = metrics["stdp_ensemble"]["test"]["pearson_ic"]
    raw_ic = metrics["raw_history"]["test"]["pearson_ic"]
    random_ic = metrics["random_ensemble"]["test"]["pearson_ic"]
    augmented_ic = metrics["raw_plus_stdp_ensemble"]["test"]["pearson_ic"]
    text = f"""# IC 分钟特征与 STDP-SNN 首版实验

训练已完成。保存了 {len(cfg['seeds'])} 个随机种子的 STDP 模型及各自的随机权重对照。
这是**无监督 STDP 表示 + 有监督线性读出**实验；未来收益没有参与 STDP 更新。

## 本轮结果

- STDP 表示三种子平均预测的测试 IC：{fmt(stdp_ic)}。
- 原始特征完整序列线性读出的测试 IC：{fmt(raw_ic)}。
- 随机权重 SNN 对照的测试 IC：{fmt(random_ic)}。
- 原始特征序列加 STDP 表示的测试 IC：{fmt(augmented_ic)}。
- 数值高低仅是这次固定切分的观察；增量价值应结合随机对照、种子差异、每日表现和以后更多时期验证。
- 本轮未进行可执行策略回测，未计算或承诺交易盈利。

## 数据与信息边界

输入 `{cfg['data_path']}`，SHA256 `{manifest['data_sha256']}`。
原始数据仍为合约身份未知的主连，换月、价格调整、量单位和分钟时间标签尚待确认。
本实验暂将每行视为已结束分钟，在该行数据全部可得后生成信号。
上午/下午、跨日、缺分钟及已知合约切换分别分段；所有特征历史、输入序列和未来目标均不跨段。
这降低跨日拼接影响，但不能证明供应商在段内未切换或未调整价格。

| 集合 | 日期范围 | 交易日数 | 有效窗口数 |
| --- | --- | ---: | ---: |
{chr(10).join(split_lines)}

目标为 `10000 * log(close[t+5] / close[t])`，单位基点；它是未来收益预测目标，不是实际成交盈亏。
每个样本使用过去 {cfg['sequence_minutes']} 个已完成分钟的特征。特征尺度仅使用训练期有效分钟的绝对值 {cfg['scale_quantile']:.0%} 分位数，截断到 [-1,1]，不减均值，保存零点含义。
STDP 和阈值自适应仅在训练期运行；验证和测试冻结全部参数。Ridge 正则系数仅按验证 MSE 选择；没有使用测试结果选择网络或特征。

## 特征

历史价位是候选支撑/阻力，不预设有效。所有参考高低点和历史尺度排除当前分钟；本分钟 OHLCV/OI 在分钟结束后才使用。

| 特征 | 定义 |
| --- | --- |
{chr(10).join(feature_lines)}

持仓变化不等于净资金流入或净多头；没有成交额/盘口输入，因此未构造真实 VWAP、订单簿或主动买卖量特征。
开盘和午休后的历史不足窗口不补零，零分母/缺失使相应窗口无效；平坦分钟收盘位置及平坦路径效率定义为0。

## 编码与网络

10 个数值特征拆成20个正负通道。每个市场分钟对应 {cfg['network']['steps_per_minute']} 个模拟时间步，按强度做 Bernoulli 时间离散的 Poisson 风格速率编码。
每个样本共 {cfg['sequence_minutes'] * cfg['network']['steps_per_minute']} 步；模拟步不是行情 tick。
20 输入 → {cfg['network']['hidden']} 个竞争 LIF 神经元。每步每个样本最多一个神经元发放，带侧向抑制和训练期阈值自适应。
STDP 用衰减的 pre/post 迹：先前 pre 与当前 post 配对增强，先前 post 与当前 pre 配对减弱；同一步的配对不更新。权重有界，并在每个 batch 后作入连接总量归一化。
状态在一个样本的分钟之间保留、在样本边界复位，已学习权重不复位。时间常数以模拟步计，pre/post 迹约跨2个市场分钟。
输出表示为每个隐藏神经元的窗口发放率及最后归一化膜电位，共128维；单个 spike 没有预先赋予买卖含义。

## 训练检查

| 模型 | 轮数 | 训练中活跃神经元 | 每窗口总发放 | 权重相对变化 | 平均权重余弦相似度 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(train_lines)}

STDP 没有以未来收益定义的训练 loss。上表检查网络确实发放、突触发生变化及权重是否过于相似。
“活跃”仅说明该轮训练有过发放，不证明神经元有预测能力。
随机权重对照保持初始突触不变，但用相同训练数据/轮数校准阈值；两者使用相同的初始权重和编码随机种子。

![训练诊断](training_diagnostics.png)
![突触权重](synaptic_weights.png)

## 预测对照

| 模型 | 验证 IC | 测试 IC | 测试 Rank IC | 测试 RMSE / bps | 测试方向准确率 | 测试每日 IC 均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(metric_lines)}

- `price_history`：仅近1分钟价格变化的10分钟序列。
- `raw_last` / `raw_history`：最后一分钟10特征 / 完整10分钟特征序列，使用同样训练期尺度。
- `random_ensemble` / `stdp_ensemble`：随机突触 / STDP 突触产生的表示，经各自训练的 Ridge 读出，再对3个种子预测取均值。
- `raw_plus_stdp_ensemble`：原始特征序列与 STDP 表示拼接，用于检查增量信息。
- `shuffled_history_ensemble`：固定最后一分钟，打乱前9分钟的编码块；使用已训练 STDP 网络并单独拟合读出。这是顺序敏感性诊断，不是“重新打乱训练”的完整因果消融；滚动特征自身仍包含历史信息。
- 方向准确率排除目标恰好为0的样本；IC 对常数预测不定义。
- 未来5分钟窗口重叠，样本非独立；没有把分钟数量当成独立样本数作显著性声明。

![预测比较](forecast_comparison.png)

## 保存内容与复现

- `features.csv`、`windows_metadata.csv`、`dataset.npz`：因果特征、窗口索引和归一化输入；未来标签单独存放。
- `normalization.json`、`config.json`、`manifest.json`：训练期尺度、全部参数、输入与源码哈希及依赖版本。
- `models/`：STDP 与随机对照的 PyTorch 检查点、Ridge 读出参数。
- `representations_seed*.npz`：固定神经表示。
- `training_history.json`、`metrics.json`、`predictions.csv`、`ridge_selection.json`：训练与评价记录。

运行：`OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m snn_quant.run --config configs/baseline.json --output outputs/new_run`
结果目录不得覆盖旧实验；中途故障可用同一配置加 `--resume` 复用已完成的神经模型检查点。
真实合约数据取得后，应重新核对并运行同一流程；本报告不替代真实合约、手续费与滑点下的交易回测。
"""
    (out / "report.md").write_text(text, encoding="utf-8")


def run(config_path: Path, out: Path, resume=False):
    started = time.monotonic()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    source_paths = sorted(Path("snn_quant").glob("*.py"))
    manifest = {
        "data_sha256": file_hash(cfg["data_path"]),
        "config_sha256": file_hash(config_path),
        "source_sha256": {str(p): file_hash(p) for p in source_paths},
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "torch": torch.__version__,
        "device": "cpu", "torch_threads": cfg["torch_threads"],
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    if out.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {out}; use a new output directory")
        previous = json.loads((out / "manifest.json").read_text())
        for key in ("data_sha256", "config_sha256", "source_sha256"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Cannot resume changed experiment: {key}")
    else:
        out.mkdir(parents=True)
    (out / "models").mkdir(exist_ok=True)
    dump_json(out / "manifest.json", manifest)
    dump_json(out / "config.json", cfg)
    torch.set_num_threads(cfg["torch_threads"])
    torch.use_deterministic_algorithms(True)
    bars = load_bars(cfg["data_path"])
    features = build_features(bars, cfg["feature_lookback"], cfg["horizon_minutes"])
    data = build_windows(features, cfg["sequence_minutes"], cfg["train_end"], cfg["validation_end"])
    scale = fit_scales(features, cfg["train_end"], cfg["scale_quantile"])
    windows = normalize(data.x, scale)
    meta = data.metadata
    masks = {name: (meta.split == name).to_numpy() for name in ("train", "validation", "test")}
    if any(mask.sum() < 100 for mask in masks.values()):
        raise ValueError("Each chronological split must have at least 100 valid windows")
    y = meta.future_return_bps.to_numpy(dtype=np.float64)
    features.to_csv(out / "features.csv", index=False)
    meta.to_csv(out / "windows_metadata.csv", index=False)
    np.savez_compressed(out / "dataset.npz", x=windows, y=y,
                        split=meta.split.to_numpy(dtype=str), feature_names=np.array(FEATURE_NAMES))
    dump_json(out / "normalization.json", {"fit_end": cfg["train_end"], "quantile": cfg["scale_quantile"],
              "scales": dict(zip(FEATURE_NAMES, scale.tolist())), "descriptions": FEATURE_DESCRIPTIONS})
    audit = {"raw_rows": len(bars), "segments": int(bars.segment_id.nunique()),
             "feature_valid_rows": int(features.feature_valid.sum()),
             "window_counts": {name: int(mask.sum()) for name, mask in masks.items()},
             "contract_identity_available": bool((bars.contract_id != "UNKNOWN_CONTINUOUS").all()),
             "minute_timestamp_assumption": "completed minute; unverified source convention",
             "zero_future_return_count": int((y == 0).sum()),
             "scaler_train_only": True, "stdp_uses_labels": False,
             "clipping_fraction": {name: float((np.abs(data.x[mask] / scale) > 1).mean())
                                   for name, mask in masks.items()}}
    dump_json(out / "data_audit.json", audit)
    print("Prepared windows: " + json.dumps(audit["window_counts"]), flush=True)
    predictions = {"train_mean": np.full(len(y), y[masks["train"]].mean())}
    ridge_selection = {}

    def readout(name, x):
        model, selection = select_ridge(x[masks["train"]], y[masks["train"]],
                                        x[masks["validation"]], y[masks["validation"]], cfg["ridge_alphas"])
        predictions[name] = model.predict(x)
        model.save(out / "models" / f"{name}_readout.npz")
        ridge_selection[name] = {"selected_alpha": model.alpha, "candidates": selection}

    raw_history = windows.reshape(len(windows), -1)
    readout("price_history", windows[:, :, 0])
    readout("raw_last", windows[:, -1])
    readout("raw_history", raw_history)
    network_cfg = NetworkConfig(**cfg["network"])
    histories, trained_networks = {}, []
    for seed in cfg["seeds"]:
        representations = {}
        for mode in ("random", "stdp"):
            name = f"{mode}_seed{seed}"
            path = out / "models" / f"{name}.pt"
            net = CompetitiveSTDP(len(FEATURE_NAMES), network_cfg, seed)
            if resume and path.exists() and (out / f"{name}_history.json").exists():
                state = torch.load(path, map_location="cpu", weights_only=True)
                net.load_state_dict(state["state_dict"])
                history = json.loads((out / f"{name}_history.json").read_text())
                print(f"Resumed {name}", flush=True)
            else:
                history = net.fit(windows[masks["train"]], learn_weights=(mode == "stdp"), log_prefix=name)
                if history[-1]["active_neurons"] < network_cfg.hidden // 2:
                    raise RuntimeError(f"Too many inactive neurons in {name}; inspect training diagnostics before evaluation")
                if mode == "stdp" and history[-1]["weight_relative_change"] < 1e-6:
                    raise RuntimeError("STDP weights did not change")
                if mode == "random" and not torch.equal(net.weight, net.initial_weight):
                    raise RuntimeError("Random-weight control weights changed")
                net.save(path, FEATURE_NAMES, {"train_end": cfg["train_end"], "mode": mode})
                dump_json(out / f"{name}_history.json", history)
            histories[name] = history
            before = net.state_dict()
            representation = net.transform(windows, cfg["encoding_seed"])
            if any(not torch.equal(before[k], net.state_dict()[k]) for k in before):
                raise RuntimeError("Evaluation mutated the trained network")
            representations[mode] = representation
            readout(name, representation)
            if mode == "stdp":
                trained_networks.append(net)
                readout(f"raw_plus_stdp_seed{seed}", np.concatenate((raw_history, representation), axis=1))
                shuffled = net.transform(windows, cfg["encoding_seed"], shuffle_history=True)
                representations["shuffled"] = shuffled
                readout(f"shuffled_history_seed{seed}", shuffled)
        np.savez_compressed(out / f"representations_seed{seed}.npz", **representations)
        dump_json(out / "training_history.json", histories)
        dump_json(out / "ridge_selection.json", ridge_selection)
        print(f"Completed representations and validation-selected readouts for seed {seed}", flush=True)
    for family in ("random", "stdp", "raw_plus_stdp", "shuffled_history"):
        predictions[f"{family}_ensemble"] = np.mean([predictions[f"{family}_seed{seed}"] for seed in cfg["seeds"]], axis=0)
    # All network/readout choices are complete before test metrics are inspected.
    metrics = {}
    for name, prediction in predictions.items():
        metrics[name] = {split: evaluate(y[mask], prediction[mask], meta.trading_date[mask])
                         for split, mask in masks.items() if split != "train"}
    prediction_frame = meta.copy()
    for name, prediction in predictions.items():
        prediction_frame[name] = prediction
    prediction_frame.to_csv(out / "predictions.csv", index=False)
    dump_json(out / "metrics.json", metrics)
    dump_json(out / "ridge_selection.json", ridge_selection)
    model_order = ["train_mean", "price_history", "raw_last", "raw_history", "random_ensemble",
                   "stdp_ensemble", "raw_plus_stdp_ensemble", "shuffled_history_ensemble"]
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest["input_unchanged"] = manifest["data_sha256"] == file_hash(cfg["data_path"])
    if not manifest["input_unchanged"]:
        raise RuntimeError("Input file changed during experiment")
    make_plots(out, histories, trained_networks, metrics, model_order)
    write_report(out, cfg, meta, metrics, histories, model_order, manifest)
    manifest["status"] = "complete"
    manifest["finished_local"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    dump_json(out / "manifest.json", manifest)
    print("Experiment complete: " + str(out / "report.md"), flush=True)
    for name in model_order:
        print(f"{name}: validation IC={fmt(metrics[name]['validation']['pearson_ic'])}, "
              f"test IC={fmt(metrics[name]['test']['pearson_ic'])}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ic_stdp_v1"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.resume)


if __name__ == "__main__":
    main()
