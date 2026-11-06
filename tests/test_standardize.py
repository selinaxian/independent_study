import csv
import io
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from snn_data.archives import archive_format, iter_ic_members, member_symbol
from snn_data.standardize import (RAW_COLUMNS, build, copy_audit_evidence, daily_quality,
                                  parse_member, resolve_duplicates)


def payload(rows):
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(RAW_COLUMNS)
    writer.writerows(rows)
    return text.getvalue().encode('utf-8-sig')


def observation(day='2026-01-05', minute='09:31:00', close='100.2001953125', volume='3'):
    return [f'{day} {minute}', '100.0', '101.0', '99.0', close, '500', volume]


def parsed(symbol='IC2606', rows=None, source='a', date='20260105'):
    return parse_member(payload(rows or [observation()]), symbol, source, date)[0]


def test_archive_selection_preserves_contract_identity_and_ignores_index(tmp_path):
    path = tmp_path / '20260105.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('20260105/中金所期货/IC2606_中证2606.csv', payload([observation()]))
        z.writestr('20260105/中金所期货/ICL8_中证主连.csv', payload([observation()]))
        z.writestr('20260105/中金所期货/IC500_中证500.csv', payload([observation()]))
        z.writestr('其他/IC2606_同名.csv', payload([observation()]))
    members = list(iter_ic_members(path))
    assert archive_format(path) == 'zip'
    assert {member_symbol(name) for name, _ in members} == {'IC2606', 'ICL8'}
    disguised = tmp_path / 'disguised.zip'
    disguised.write_bytes(b'7z\xbc\xaf\x27\x1c' + b'\0' * 16)
    assert archive_format(disguised) == '7z'


def test_source_row_timezone_and_float_integer_precision_survive_parquet(tmp_path):
    row = observation(volume='9007199254740993')
    body = payload([[], row])
    frame, rejected, info = parse_member(body, 'IC2606', 'source1', '20260105')
    assert not rejected and info['raw_rows'] == 1
    assert frame.loc[0, 'source_row'] == 3
    assert frame.loc[0, 'contract_id'] == 'IC2606'
    assert frame.loc[0, 'volume'] == 9007199254740993
    assert frame.loc[0, 'close'] == float(row[4])
    assert str(frame.datetime.dt.tz) == 'Asia/Shanghai'
    path = tmp_path / 'roundtrip.parquet'
    frame.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(path), check_exact=True)


def test_continuous_alias_has_no_fabricated_contract_and_empty_has_no_fake_bars():
    main = parsed('ICL8')
    assert main.contract_id.isna().all()
    assert main.instrument_type.tolist() == ['continuous']
    for body in (payload([]), b'\xef\xbb\xbf\n'):
        frame, bad, info = parse_member(body, 'IC2606', 'x', '20260105')
        assert frame.empty and not bad and info['raw_rows'] == 0


@pytest.mark.parametrize('column,value', [(0, 'not-a-time'), (0, '2026-01-06 09:31:00'),
                                         (2, '98.0'), (4, 'nan'), (4, '-1'),
                                         (5, '-1'), (6, '1.5'), (6, str(2**63))])
def test_invalid_rows_are_quarantined_with_original_values(column, value):
    row = observation()
    row[column] = value
    frame, bad, info = parse_member(payload([row]), 'IC2606', 'src', '20260105')
    assert frame.empty and len(bad) == info['quarantined_rows'] == 1
    assert json.loads(bad[0]['raw_fields_json']) == row
    assert bad[0]['source_id'] == 'src' and bad[0]['source_row'] == 2


def test_zero_volume_is_retained_and_unexpected_header_rejected():
    frame, bad, _ = parse_member(payload([observation(volume='0')]), 'IC2606', 'src', '20260105')
    assert len(frame) == 1 and not bad
    assert frame.quality_flags.tolist() == ['zero_volume']
    with pytest.raises(ValueError, match='Unexpected columns'):
        parse_member(b'date,close\n2026-01-05 09:31:00,100\n', 'IC2606', 'src', '20260105')


def test_same_timestamp_across_contracts_is_not_a_duplicate():
    frame = pd.concat([parsed('IC2606'), parsed('IC2609', source='b')], ignore_index=True)
    clean, duplicates, conflicts = resolve_duplicates(frame)
    assert len(clean) == 2 and not duplicates and not conflicts


def test_exact_duplicates_record_both_sources_and_conflicts_quarantine_both():
    a, b = parsed(source='a'), parsed(source='b')
    clean, duplicates, conflicts = resolve_duplicates(pd.concat([b, a], ignore_index=True))
    assert len(clean) == 1 and clean.source_id.tolist() == ['a'] and not conflicts
    assert {r['action'] for r in duplicates} == {'retained', 'removed_exact'}
    changed = parsed(rows=[observation(close='100.4')], source='c')
    other_contract = parsed('IC2609', source='d')
    clean, duplicates, conflicts = resolve_duplicates(pd.concat([a, changed, other_contract], ignore_index=True))
    assert clean.instrument_id.tolist() == ['IC2609']
    assert len(conflicts) == 2 and {r['source_id'] for r in conflicts} == {'a', 'c'}


def test_missing_archive_member_and_empty_member_remain_distinct():
    frames, members = [], []
    for date in ['20260105', '20260107', '20260108']:
        rows = [] if date == '20260108' else [observation(day=f'{date[:4]}-{date[4:6]}-{date[6:]}')]
        frame, _, info = parse_member(payload(rows), 'IC2606', date, date)
        frames.append(frame)
        members.append({'symbol': 'IC2606', 'archive_date': date, **info})
    dates = ['20260105', '20260106', '20260107', '20260108']
    quality = daily_quality(pd.concat(frames, ignore_index=True), members,
                            [{'archive_date': d} for d in dates], [], []).set_index('trading_date')
    assert quality.loc['2026-01-06', 'status'] == 'not_in_archive'
    assert quality.loc['2026-01-06', 'within_observed_span']
    assert quality.loc['2026-01-08', 'status'] == 'empty_file'
    assert not quality.loc['2026-01-08', 'within_observed_span']
    assert quality.written_rows.sum() == 2


def test_stale_audit_evidence_is_never_copied(tmp_path):
    source, target = tmp_path / 'audit', tmp_path / 'meta'
    source.mkdir(); target.mkdir()
    member = '中金所期货/IC2606_中证2606.csv'
    pd.DataFrame([{'archive_date': '20260105', 'member': member, 'payload_sha256': 'old'}]).to_csv(source / 'cffex_members.csv', index=False)
    (source / 'ic_main_matches.csv').write_text('stale')
    status = copy_audit_evidence(source, [{'archive_date': '20260105', 'source_member': member,
                                         'payload_sha256': 'new'}], target)
    assert status['status'] == 'source_members_differ_from_previous_audit'
    assert not list(target.iterdir())


def test_end_to_end_merges_days_per_contract_and_protects_output(tmp_path):
    source, target = tmp_path / 'archives', tmp_path / 'standardized'
    source.mkdir()
    for day in ['20260105', '20260106']:
        iso = f'{day[:4]}-{day[4:6]}-{day[6:]}'
        with zipfile.ZipFile(source / f'{day}.zip', 'w') as z:
            for code in ['IC2606', 'IC2609', 'ICL8']:
                z.writestr(f'{day}/中金所期货/{code}_示例.csv', payload([observation(day=iso)]))
    summary = build(source, target, tmp_path / 'no_audit')
    assert summary['contract_rows'] == 4 and summary['continuous_rows'] == 2
    all_contracts = pd.read_parquet(target / 'contracts')
    assert not all_contracts.duplicated(['contract_id', 'datetime']).any()
    assert len(pd.read_parquet(target / 'contracts/IC2606.parquet')) == 2
    assert pd.read_parquet(target / 'continuous/ICL8.parquet').contract_id.isna().all()
    manifest = json.loads((target / 'metadata/manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['source_size_mtime_unchanged']
    with pytest.raises(FileExistsError):
        build(source, target, tmp_path / 'no_audit')
