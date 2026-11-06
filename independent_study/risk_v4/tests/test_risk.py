import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from iclab.backtest import simulate,trade_accounting
from risk_v4.core import fit_risk_rule,risk_lots,account,replay,matched_exposure,block_interval


def fixture():
    cfg=json.loads(Path('independent_study/configs/experiment.json').read_text())['trading']
    n=20;prices=100+np.arange(n)*.2
    bars=pd.DataFrame({'bar_id':np.arange(n),'datetime':pd.date_range('2026-04-01 10:00',periods=n,freq='min',tz='Asia/Shanghai'),
        'trading_date':['2026-04-01']*n,'segment_id':[0]*n,'contract_id':['IC2606']*n,
        'open':prices,'close':prices+.05,'volume':[100]*n})
    meta=pd.DataFrame({'sample_id':[0,1,2],'bar_id':[2,5,10],'entry_bar_id':[4,7,12]})
    return cfg,bars,meta


def test_rule_rejects_future_calibration_and_only_uses_supplied_past():
    values=np.arange(1,11.)
    rule=fit_risk_rule(values,['2026-03-31']*10,'2026-04-01')
    assert rule['median']==5.5 and rule['threshold']==8.2
    with pytest.raises(ValueError):fit_risk_rule(values,['2026-04-01']*10,'2026-04-01')
    with pytest.raises(ValueError):fit_risk_rule([np.nan],['2026-03-31'],'2026-04-01')


def test_integer_sizing_and_zero_forecasts():
    rule={'threshold':8.,'median':10.,'floor':1.}
    np.testing.assert_array_equal(risk_lots([0,5,10,11,20,21,100],rule,'sizing'),[2,2,2,1,1,0,0])
    np.testing.assert_array_equal(risk_lots([7,8,9],rule,'filter'),[1,1,0])
    with pytest.raises(ValueError):risk_lots([-1],rule,'sizing')


def test_lot_fees_are_per_order_and_slippage_scales():
    cfg,_,_=fixture();one=trade_accounting(100,102,1,cfg)
    assert account(100,102,1,1,cfg)==one
    two=account(100,102,1,2,cfg)
    assert two['entry_fee']==pytest.approx((100+.2)*200*(.000023+.00001)*2+1)
    assert two['net_pnl']==pytest.approx(2*one['net_pnl']+2*cfg['order_fee'])
    assert account(100,102,1,0,cfg)['net_pnl']==0
    with pytest.raises(ValueError):account(100,102,1,.5,cfg)


def test_replay_reproduces_original_and_keeps_skipped_slot():
    cfg,bars,meta=fixture();calendar=['2026-04-01','2026-04-02']
    s,t,e,d=simulate(bars,meta,[1,1,-1],0,cfg,calendar)
    assert t.sample_id.tolist()==[0,2]
    new,tr,eq,daily=replay(bars,t,[1,1],cfg,calendar)
    np.testing.assert_allclose(eq.equity,e.equity,atol=1e-7,rtol=0)
    np.testing.assert_allclose(daily.equity,d.equity,atol=1e-7,rtol=0)
    assert new['net_pnl']==pytest.approx(s['net_pnl'])
    _,skip,_,_=replay(bars,t,[0,1],cfg,calendar)
    assert skip.sample_id.tolist()==[0,2] and skip.lots.tolist()==[0,1]
    assert skip.iloc[1].entry_time==t.iloc[1].entry_time


def test_continuous_cash_and_margin():
    cfg,bars,meta=fixture();_,t,_,_=simulate(bars,meta,[1,1,-1],0,cfg,['2026-04-01'])
    small={**cfg,'capital':1000.}
    s,tr,eq,_=replay(bars,t,[2,2],small,['2026-04-01'])
    assert s['margin_rejections']==2 and s['net_pnl']==0 and (eq.equity==1000).all()
    _,tr,_,_=replay(bars,t,[2,1],cfg,['2026-04-01'])
    assert tr.iloc[-1].cash_after==pytest.approx(cfg['capital']+tr.net_pnl.sum())


def test_matched_exposure_expected_value_and_blocks():
    cfg,bars,meta=fixture();_,t,_,_=simulate(bars,meta,[1,1,-1],0,cfg,['2026-04-01'])
    _,tr,_,_=replay(bars,t,[2,0],cfg,['2026-04-01']);tr['fold']=0
    result,expected=matched_exposure(tr,cfg,repetitions=100)
    a=account(t.iloc[0].entry_open,t.iloc[0].exit_open,1,2,cfg)['net_pnl']
    b=account(t.iloc[1].entry_open,t.iloc[1].exit_open,-1,2,cfg)['net_pnl']
    assert expected.sum()==pytest.approx((a+b)/2)
    assert result['selection_excess']==pytest.approx(a-(a+b)/2)
    assert block_interval(np.zeros(23))==(0.,0.)
    assert block_interval(np.ones(23))==(23.,23.)
