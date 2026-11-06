"""Freeze the complete v10 research deliverable; subsequent calls only verify."""
import json
import re
from pathlib import Path
from urllib.parse import unquote
import pandas as pd
from iclab.data import dump_json,sha256
from research_v10.state_run import ROOT,HERE

RESULTS=ROOT/'independent_study/results';OUT=RESULTS/'v10'


def main():
    path=OUT/'manifest.json'
    if path.exists():
        record=json.loads(path.read_text())
        for section in ['artifacts','sources']:
            for file,h in record[section].items():assert sha256(ROOT/file)==h,file
        print('Existing manifest verified:',len(record['artifacts']),'artifacts,',len(record['sources']),'sources');return
    required=['report.md','verification.json','metrics.csv','paired_intervals.csv','economic_scopes.csv']
    for name in required:assert (OUT/name).is_file(),name
    verification=json.loads((OUT/'verification.json').read_text());assert verification['status']=='complete'
    models=pd.read_csv(RESULTS/'v10_refit/development_metrics.csv').set_index('model')
    selection=json.loads((RESULTS/'v10_refit/economic_selection_lock.json').read_text())
    assert selection['primary']==min(selection['development_scores'],key=selection['development_scores'].get)
    for name,value in selection['development_scores'].items():assert abs(value-models.loc[name,'qlike'])<1e-14
    economic=pd.read_csv(RESULTS/'v10_economics/summary.csv');assert len(economic)==27 and (economic.net_pnl<0).all()
    documents=[OUT/'report.md',HERE/'README.md',HERE/'RESEARCH.md',ROOT/'independent_study/VERSIONS.md']
    link_count=0
    for document in documents:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',document.read_text()):
            if target.startswith(('http:','https:','#')):continue
            target=unquote(target.split('#')[0]);assert (document.parent/target).resolve().exists(),(document,target);link_count+=1
    artifacts={};sources={}
    for directory in sorted(RESULTS.glob('v10*')):
        if not directory.is_dir():continue
        for file in sorted(directory.rglob('*')):
            if file.is_file() and file!=path:artifacts[str(file.relative_to(ROOT))]=sha256(file)
    for file in sorted(HERE.rglob('*')):
        if file.is_file() and file.suffix in ['.py','.md','.json'] and '__pycache__' not in file.parts:
            sources[str(file.relative_to(ROOT))]=sha256(file)
    # Include exactly the upstream training/data dependencies recorded by each stage.
    for lock in sorted(RESULTS.glob('v10*/source_lock.json')):
        data=json.loads(lock.read_text())
        # Stage A, like B--F, stores a path -> hash mapping.
        for file,h in data.items():
            assert isinstance(h,str) and sha256(ROOT/file)==h,file
            sources[file]=h
    for file,h in json.loads((RESULTS/'v10_economics/verification.json').read_text())['sources'].items():
        assert sha256(ROOT/file)==h,file;sources[file]=h
    for file in ['independent_study/risk_v4/core.py','independent_study/finance_v8/core.py','independent_study/iclab/data.py',
                 'independent_study/event_stdp_v7/core.py','independent_study/configs/experiment.json',
                 'independent_study/results/v2/backtests/rolling_all__raw_history__base_trades.csv']:
        sources[file]=sha256(ROOT/file)
    record=dict(status='complete',scope='exploratory historical research; no established unique STDP increment or profitable strategy',
        artifacts=artifacts,sources=sources,local_links_checked=link_count,figures_visually_checked=4,
        primary=selection['primary'],all_27_economic_cases_net_negative=True)
    dump_json(path,record)
    summary={**verification,'manifest':str(path.relative_to(ROOT)),'manifest_sha256':sha256(path),
             'artifact_count':len(artifacts),'source_count':len(sources),'local_links_checked':link_count,'figures_visually_checked':4}
    dump_json(ROOT/'outputs/independent_v10_verification.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['economic']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
