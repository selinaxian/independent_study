"""Merge daily IC members by instrument: python -m snn_data.standardize."""
import argparse
from collections import Counter
import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow

from .archives import CONTRACT, CONTINUOUS, archive_format, file_hash, iter_ic_members, member_symbol

TIMEZONE = ZoneInfo('Asia/Shanghai')
RAW_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'open_interest', 'volume']
VALUES = RAW_COLUMNS[1:]
KEY = ['instrument_id', 'datetime']
BAR_COLUMNS = ['instrument_id', 'contract_id', 'instrument_type', 'datetime', 'trading_date',
               'session', *VALUES, 'source_id', 'source_row', 'quality_flags']
QUARANTINE_COLUMNS = ['instrument_id', 'archive_date', 'source_id', 'source_row',
                      'reason', 'raw_fields_json', 'normalized_record_json']
DUPLICATE_COLUMNS = ['instrument_id', 'datetime', 'source_id', 'source_row', 'action']
REFERENCE_TIMES = tuple(
    f'{minute // 60:02d}:{minute % 60:02d}:00'
    for start, end in ((9 * 60 + 31, 11 * 60 + 30), (13 * 60 + 1, 15 * 60))
    for minute in range(start, end + 1)
)
REFERENCE_SET = set(REFERENCE_TIMES)


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def identity(*parts):
    return hashlib.sha256('\0'.join(parts).encode()).hexdigest()


def count_value(text):
    value = Decimal(text)
    if not value.is_finite() or value < 0 or value != value.to_integral_value() or value > 2**63 - 1:
        raise ValueError('invalid_count')
    return int(value)


def typed_bars(records):
    frame = pd.DataFrame(records, columns=BAR_COLUMNS)
    for column in ['instrument_id', 'contract_id', 'instrument_type', 'trading_date',
                   'session', 'source_id', 'quality_flags']:
        frame[column] = frame[column].astype('string')
    frame['datetime'] = pd.to_datetime(frame['datetime'], utc=True).dt.tz_convert('Asia/Shanghai')
    for column in ('open', 'high', 'low', 'close'):
        frame[column] = frame[column].astype('float64')
    for column in ('volume', 'open_interest', 'source_row'):
        frame[column] = frame[column].astype('int64')
    return frame


def parse_member(payload, symbol, source_id, archive_date):
    """Preserve source values and row locations; never fill malformed observations."""
    if not (CONTRACT.fullmatch(symbol) or symbol in CONTINUOUS):
        raise ValueError(f'Not an IC contract/continuous symbol: {symbol}')
    for encoding in ('utf-8-sig', 'gb18030'):
        try:
            text = payload.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError('Unrecognized CSV encoding')
    reader = csv.reader(io.StringIO(text), strict=True)
    header = next((row for row in reader if any(v.strip() for v in row)), [])
    if header and header != RAW_COLUMNS:
        raise ValueError(f'Unexpected columns for {symbol}: {header}')
    records, rejected, times = [], [], []
    raw_count = 0
    for row in reader:
        if not any(v.strip() for v in row):
            continue
        raw_count += 1
        source_row = reader.line_num
        try:
            if len(row) != len(RAW_COLUMNS):
                raise ValueError('wrong_column_count')
            instant = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=TIMEZONE)
            if instant.strftime('%Y%m%d') != archive_date:
                raise ValueError('archive_date_mismatch')
            prices = [float(value) for value in row[1:5]]
            if not all(math.isfinite(v) and v > 0 for v in prices):
                raise ValueError('invalid_price')
            op, high, low, close = prices
            if high < max(op, low, close) or low > min(op, high, close):
                raise ValueError('inconsistent_ohlc')
            oi, volume = count_value(row[5]), count_value(row[6])
            minute = instant.strftime('%H:%M:%S')
            flags = []
            if minute not in REFERENCE_SET:
                flags.append('outside_reference_session')
            if volume == 0:
                flags.append('zero_volume')
            records.append({
                'instrument_id': symbol, 'contract_id': symbol if CONTRACT.fullmatch(symbol) else None,
                'instrument_type': 'contract' if CONTRACT.fullmatch(symbol) else 'continuous',
                'datetime': instant, 'trading_date': instant.strftime('%Y-%m-%d'),
                'session': 'AM' if instant.hour < 12 else 'PM',
                'open': op, 'high': high, 'low': low, 'close': close,
                'open_interest': oi, 'volume': volume, 'source_id': source_id,
                'source_row': source_row, 'quality_flags': '|'.join(flags),
            })
            times.append(instant)
        except (ValueError, InvalidOperation, OverflowError) as exc:
            rejected.append({'instrument_id': symbol, 'archive_date': archive_date,
                             'source_id': source_id, 'source_row': source_row, 'reason': str(exc),
                             'raw_fields_json': json.dumps(row, ensure_ascii=False),
                             'normalized_record_json': ''})
    info = {'encoding': encoding, 'raw_rows': raw_count, 'parsed_rows': len(records),
            'quarantined_rows': len(rejected), 'source_time_ordered': times == sorted(times),
            'first_time': times[0].isoformat() if times else '',
            'last_time': times[-1].isoformat() if times else '',
            'header': ','.join(header)}
    return typed_bars(records), rejected, info


def resolve_duplicates(frame):
    """Exact repeats retain a deterministic source; conflicting keys retain neither."""
    frame = frame.sort_values(KEY + ['source_id', 'source_row'], kind='stable').reset_index(drop=True)
    repeated = frame.duplicated(KEY, keep=False)
    remove, duplicates, conflicts = [], [], []
    for _, group in frame[repeated].groupby(KEY, sort=False):
        exact = len(group[VALUES].drop_duplicates()) == 1
        for offset, (index, row) in enumerate(group.iterrows()):
            action = 'retained' if exact and offset == 0 else 'removed_exact' if exact else 'quarantined_conflict'
            duplicates.append({'instrument_id': row.instrument_id, 'datetime': row.datetime.isoformat(),
                               'source_id': row.source_id, 'source_row': int(row.source_row), 'action': action})
            if action != 'retained':
                remove.append(index)
            if not exact:
                conflicts.append({'instrument_id': row.instrument_id,
                                  'archive_date': row.datetime.strftime('%Y%m%d'),
                                  'source_id': row.source_id, 'source_row': int(row.source_row),
                                  'reason': 'conflicting_values_for_same_key', 'raw_fields_json': '',
                                  'normalized_record_json': json.dumps(row.to_dict(), ensure_ascii=False, default=str)})
    clean = frame.drop(index=remove).reset_index(drop=True)
    assert not clean.duplicated(KEY).any()
    return clean, duplicates, conflicts


def daily_quality(bars, members, archives, quarantine, duplicates):
    symbols = sorted({m['symbol'] for m in members})
    data_groups = {(symbol, day): g for (symbol, day), g in bars.groupby(['instrument_id', 'trading_date'], sort=False)}
    sources = {}
    for member in members:
        sources.setdefault((member['symbol'], member['archive_date']), []).append(member)
    rejected = Counter((r['instrument_id'], r['archive_date']) for r in quarantine)
    dedup = Counter((r['instrument_id'], r['datetime'][:10].replace('-', '')) for r in duplicates if r['action'] == 'removed_exact')
    bounds = {symbol: (g.trading_date.min(), g.trading_date.max()) for symbol, g in bars.groupby('instrument_id')}
    records = []
    for symbol in symbols:
        for date in sorted({a['archive_date'] for a in archives}):
            day = f'{date[:4]}-{date[4:6]}-{date[6:]}'
            members_here = sources.get((symbol, date), [])
            group = data_groups.get((symbol, day))
            times = set(group.datetime.dt.strftime('%H:%M:%S')) if group is not None else set()
            missing = sorted(REFERENCE_SET - times)
            extra = sorted(times - REFERENCE_SET)
            raw_rows = sum(m['raw_rows'] for m in members_here)
            rejected_rows, dedup_rows = rejected[(symbol, date)], dedup[(symbol, date)]
            written = len(group) if group is not None else 0
            if not members_here:
                status = 'not_in_archive'
            elif raw_rows == 0:
                status = 'empty_file'
            elif rejected_rows:
                status = 'quarantined_rows'
            elif missing or extra:
                status = 'incomplete_reference_grid'
            elif dedup_rows:
                status = 'deduplicated_exact'
            else:
                status = 'complete'
            assert raw_rows == written + rejected_rows + dedup_rows, (symbol, date)
            span = bounds.get(symbol)
            records.append({'instrument_id': symbol, 'trading_date': day,
                            'source_files': len(members_here), 'raw_rows': raw_rows,
                            'written_rows': written, 'quarantined_rows': rejected_rows,
                            'removed_exact_duplicates': dedup_rows, 'status': status,
                            'within_observed_span': bool(span and span[0] <= day <= span[1]),
                            'reference_minutes': 240, 'missing_minutes': len(missing),
                            'outside_reference_minutes': len(extra),
                            'missing_times': '|'.join(missing), 'outside_times': '|'.join(extra),
                            'zero_volume_rows': int((group.volume == 0).sum()) if group is not None else 0})
    return pd.DataFrame(records)


def copy_audit_evidence(audit, members, destination):
    index = audit / 'cffex_members.csv'
    if not index.exists():
        return {'status': 'unavailable', 'copied_files': []}
    old = pd.read_csv(index, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    expected = {(r.archive_date, r.member): r.payload_sha256 for r in old.itertuples()
                if member_symbol(r.member)}
    actual = {(r['archive_date'], r['source_member']): r['payload_sha256'] for r in members}
    if actual != expected:
        return {'status': 'source_members_differ_from_previous_audit', 'copied_files': []}
    copied = []
    for name in ('summary.json', 'ic_main_matches.csv', '20260618_field_differences.csv', 'missing_cffex_days.csv'):
        source = audit / name
        if source.exists():
            target = destination / f'audit_{name}'
            shutil.copyfile(source, target)
            copied.append({'file': target.name, 'sha256': file_hash(target)})
    return {'status': 'all_selected_payload_hashes_match', 'source_index_sha256': file_hash(index),
            'interpretation': 'Retrospective consistency only; not a real-time contract selection rule.',
            'copied_files': copied}


def save_csv(path, records, columns=None):
    pd.DataFrame(records, columns=columns).to_csv(path, index=False, encoding='utf-8-sig')


def write_readme(out, summary, coverage):
    rows = ['| 序列 | 类型 | 首日 | 末日 | 有数据天数 | 行数 |', '| --- | --- | --- | --- | ---: | ---: |']
    for r in coverage:
        rows.append(f"| {r['instrument_id']} | {r['instrument_type']} | {r['first_date']} | {r['last_date']} | {r['days']} | {r['rows']:,} |")
    text = f'''# IC标准分钟行情 v1

逐日数据已按序列合并并排序，原始行情数值保持不变。具体合约与滚动序列分开保存。

- 输入：{summary['archive_count']}个原始压缩包；ZIP/7z按文件头识别。
- 具体合约：{summary['contract_count']}个，合计{summary['contract_rows']:,}行。
- 滚动序列：{summary['continuous_count']}个，合计{summary['continuous_rows']:,}行。
- 隔离行：{summary['quarantined_rows']}；删除的完全重复行：{summary['removed_exact_duplicates']}。所有处理均保留来源。
- 供应商是否调整价格、分钟开始/结束标签和量单位仍未知；没有自行复权、填补或猜定合约。

{chr(10).join(rows)}

## 读取

```python
import pandas as pd

# 同一个具体合约的全部日期
bars = pd.read_parquet("{out.as_posix()}/contracts/IC2606.parquet")

# 全部具体合约；同一分钟存在多个合约是正常的
all_contracts = pd.read_parquet("{out.as_posix()}/contracts")
assert not all_contracts.duplicated(["contract_id", "datetime"]).any()

# 只取所需日期，再按合约分别计算特征
part = all_contracts[all_contracts.trading_date.between("2026-01-01", "2026-03-31")]

# 通过source_id定位原始文件
sources = pd.read_csv("{out.as_posix()}/metadata/source_members.csv")
row = bars.iloc[0]
origin = sources[sources.source_id == row.source_id].iloc[0]
print(origin.source_archive, origin.source_member, row.source_row)
```

`source_archive`相对路径以manifest中的source_root为根；source_row为原CSV行号，表头通常是第1行。
`previews/`内是每个序列首尾各10行CSV，便于直接打开，完整数据在Parquet中。
价格为float64，成交量/持仓量为int64，datetime带Asia/Shanghai时区。
只有具体合约文件的contract_id有值；ICL0/1/2/3/8/9的contract_id保持空值。

## 质量与信息边界

参考时段是现有数据观察到的09:31—11:30、13:01—15:00，每日240分钟，尚未由供应商确认时间标签定义。
`metadata/daily_quality.csv`逐序列逐包日期区分未提供文件、空文件、缺分钟等；只覆盖本批实际存在的包日期，不等价于已核查完整交易所日历。
`metadata/gaps_within_observed_span.csv`列出该序列首次至末次有行情之间的参考网格缺失，不把生命周期外日期直接认定为数据丢失。
`metadata/quarantine.csv`保存不合法行和同键冲突；`duplicates.csv`保存全部重复来源与处理决定。零成交量为提示，不自行替换行情。
全部价格、成交量和持仓量都来自对应文件。ICL8与IC2606的6月18日差异仍各自保留，未合并成一套价格。
复制的`audit_*`表仅在相关源文件哈希与之前审计完全相符时保留，表示事后匹配证据，不是官方主连规则或交易时点可用的主力选择信号。

后续特征、未来收益和神经状态必须按合约/连续交易时段分组；所有合约使用统一训练/验证/测试日期边界。
本次没有改动原final.csv、旧模型或训练程序；现有训练入口仍读取CSV，使用本库训练需要单独接入数据读取和分组逻辑。

## 来源与复现

`metadata/archives.csv`保存原包路径、真实格式、大小、mtime及SHA256；`source_members.csv`保存每个成员的内容哈希和来源索引。
`metadata/schema.json`记录字段和未知口径；`manifest.json`记录配置、代码/产物哈希及依赖版本；`requirements-lock.txt`保存本次运行环境。
源文件内容与来源核验结果记录在manifest和audit evidence中。原始包及final.csv未覆盖。

在项目根目录运行：

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python -m snn_data.standardize --source data/分钟线 --output data/standardized/ic_v2
```

输出目录必须不存在，以免覆盖既有版本。修改源数据或规则后生成新版本。
'''
    (out / 'README.md').write_text(text, encoding='utf-8')


def build(source, out, audit):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}; choose a new version directory')
    paths = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in ('.zip', '.7z'))
    if not paths:
        raise ValueError('No input archives')
    if any(not re.fullmatch(r'\d{8}', p.stem) for p in paths):
        raise ValueError('Archive filenames must provide YYYYMMDD source dates')
    out.mkdir(parents=True)
    for directory in ('contracts', 'continuous', 'previews', 'metadata'):
        (out / directory).mkdir()
    started = time.monotonic()
    protected = Path('data/final.csv')
    protected_hash = file_hash(protected) if protected.exists() else None
    snapshots = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    manifest = {'status': 'running', 'source_root': str(source.resolve()),
                'output_root': str(out.resolve()), 'started_local': datetime.now().astimezone().isoformat(),
                'source_code_sha256': {str(p): file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                'python': platform.python_version(), 'pandas': pd.__version__, 'pyarrow': pyarrow.__version__,
                'configuration': {'timezone': 'Asia/Shanghai', 'reference_sessions': ['09:31-11:30', '13:01-15:00'],
                                  'duplicate_policy': 'collapse exact; quarantine all conflicts',
                                  'price_adjustment_applied': False, 'fill_missing': False}}
    dump_json(out / 'metadata/manifest.json', manifest)
    try:
        frames, archives, members, quarantine = [], [], [], []
        for number, path in enumerate(paths, 1):
            relative = path.relative_to(source).as_posix()
            kind, sha = archive_format(path), file_hash(path)
            archive_id = identity(relative, sha)
            archives.append({'archive_id': archive_id, 'source_archive': relative, 'archive_date': path.stem,
                             'format': kind, 'bytes': snapshots[path][0], 'mtime_ns': snapshots[path][1], 'sha256': sha})
            for name, payload in iter_ic_members(path, kind):
                symbol = member_symbol(name)
                payload_hash = hashlib.sha256(payload).hexdigest()
                source_id = identity(archive_id, name, payload_hash)
                frame, rejected, info = parse_member(payload, symbol, source_id, path.stem)
                if len(frame):
                    frames.append(frame)
                quarantine.extend(rejected)
                members.append({'source_id': source_id, 'archive_id': archive_id, 'source_archive': relative,
                                'source_member': name, 'archive_date': path.stem, 'symbol': symbol,
                                'bytes': len(payload), 'payload_sha256': payload_hash, **info})
            if number % 25 == 0 or kind == '7z' or number == len(paths):
                print(f'Read {number}/{len(paths)}: {path.stem} ({kind})', flush=True)
        if not frames:
            raise ValueError('No parseable IC observations')
        all_bars = pd.concat(frames, ignore_index=True)
        bars, duplicates, conflicts = resolve_duplicates(all_bars)
        quarantine.extend(conflicts)
        coverage = []
        for symbol in sorted({m['symbol'] for m in members}):
            frame = bars[bars.instrument_id == symbol].reset_index(drop=True)
            kind = 'contract' if CONTRACT.fullmatch(symbol) else 'continuous'
            folder = 'contracts' if kind == 'contract' else 'continuous'
            path = out / folder / f'{symbol}.parquet'
            frame.to_parquet(path, index=False, compression='zstd')
            pd.testing.assert_frame_equal(frame, pd.read_parquet(path), check_exact=True)
            pd.concat([frame.head(10), frame.tail(10)]).drop_duplicates(KEY).to_csv(
                out / 'previews' / f'{symbol}.csv', index=False, encoding='utf-8-sig')
            coverage.append({'instrument_id': symbol, 'instrument_type': kind,
                             'delivery_month_from_code': f'20{symbol[2:4]}-{symbol[4:]}' if kind == 'contract' else '',
                             'first_date': str(frame.trading_date.min()) if len(frame) else '',
                             'last_date': str(frame.trading_date.max()) if len(frame) else '',
                             'days': int(frame.trading_date.nunique()), 'rows': len(frame),
                             'path': f'{folder}/{symbol}.parquet', 'sha256': file_hash(path)})
        quality = daily_quality(bars, members, archives, quarantine, duplicates)
        meta = out / 'metadata'
        save_csv(meta / 'archives.csv', archives)
        save_csv(meta / 'source_members.csv', members)
        save_csv(meta / 'coverage.csv', coverage)
        save_csv(meta / 'daily_quality.csv', quality)
        save_csv(meta / 'gaps_within_observed_span.csv', quality[quality.within_observed_span & (quality.missing_minutes > 0)])
        save_csv(meta / 'quarantine.csv', quarantine, QUARANTINE_COLUMNS)
        save_csv(meta / 'duplicates.csv', duplicates, DUPLICATE_COLUMNS)
        schema = {'primary_key': KEY, 'columns': {c: str(bars[c].dtype) for c in BAR_COLUMNS},
                  'contract_id_policy': 'filename code for named contracts; null for rolling series',
                  'source_id_lookup': 'metadata/source_members.csv; source_archive is relative to source_root',
                  'source_row_definition': 'physical CSV line number (header normally line 1)',
                  'timezone': 'Asia/Shanghai', 'bar_duration_seconds': 60,
                  'supplier': 'unknown', 'supplier_price_adjustment': 'unknown',
                  'supplier_timestamp_label': 'unknown', 'volume_units': 'unknown',
                  'trading_date_policy': 'local date for IC day sessions',
                  'gap_policy': 'no fill; observed archive dates only; no exchange calendar claimed',
                  'quality_flags': {'outside_reference_session': 'not on observed minute-end reference grid',
                                    'zero_volume': 'valid zero count, retained without fabrication'}}
        dump_json(meta / 'schema.json', schema)
        evidence = copy_audit_evidence(audit, members, meta)
        dump_json(meta / 'audit_evidence.json', evidence)
        contract_mask = bars.instrument_type == 'contract'
        summary = {'archive_count': len(archives), 'formats': dict(Counter(a['format'] for a in archives)),
                   'member_count': len(members), 'raw_rows': sum(m['raw_rows'] for m in members),
                   'contract_count': int(bars.loc[contract_mask, 'instrument_id'].nunique()),
                   'contract_rows': int(contract_mask.sum()),
                   'continuous_count': int(bars.loc[~contract_mask, 'instrument_id'].nunique()),
                   'continuous_rows': int((~contract_mask).sum()),
                   'quarantined_rows': len(quarantine),
                   'removed_exact_duplicates': sum(d['action'] == 'removed_exact' for d in duplicates),
                   'quality_status_counts': {str(k): int(v) for k, v in quality.status.value_counts().items()},
                   'all_parquet_roundtrips_exact': True,
                   'all_selected_payloads_match_prior_audit': evidence['status'] == 'all_selected_payload_hashes_match'}
        assert summary['raw_rows'] == len(bars) + summary['quarantined_rows'] + summary['removed_exact_duplicates']
        if protected_hash is not None:
            assert file_hash(protected) == protected_hash, 'Original final.csv changed'
        assert all((p.stat().st_size, p.stat().st_mtime_ns) == snapshots[p] for p in paths), 'Source archive changed'
        import subprocess
        import sys
        lock = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        (out / 'requirements-lock.txt').write_text(lock, encoding='utf-8')
        dump_json(meta / 'summary.json', summary)
        write_readme(out, summary, coverage)
        manifest.update({'status': 'complete', 'elapsed_seconds': time.monotonic() - started,
                         'finished_local': datetime.now().astimezone().isoformat(),
                         'original_final_csv_sha256': protected_hash, 'source_size_mtime_unchanged': True,
                         'artifact_sha256': {p.relative_to(out).as_posix(): file_hash(p)
                                             for p in sorted(out.rglob('*')) if p.is_file() and p != meta / 'manifest.json'}})
        dump_json(meta / 'manifest.json', manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f'Complete: {out / "README.md"}', flush=True)
        return summary
    except Exception as exc:
        manifest.update({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        dump_json(out / 'metadata/manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/分钟线'))
    parser.add_argument('--output', type=Path, default=Path('data/standardized/ic_v1'))
    parser.add_argument('--audit', type=Path, default=Path('outputs/minute_archive_audit'))
    args = parser.parse_args()
    build(args.source, args.output, args.audit)


if __name__ == '__main__':
    main()
