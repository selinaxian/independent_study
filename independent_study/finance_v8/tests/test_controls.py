import numpy as np
import pandas as pd
from finance_v8.controls import safe_corr,covered_matched_exposure
from risk_v4.core import account


def test_control_holds_uncovered_assignments_fixed():
    cfg=dict(tick=.2,slippage_ticks=1,multiplier=200,open_fee_rate=0,
             close_today_fee_rate=0,broker_extra_rate=0,order_fee=1)
    tr=pd.DataFrame(dict(fold=[0]*4,covered=[False,False,True,True],
                        entry_open=[100]*4,exit_open=[101,102,103,104],side=[1]*4,lots=[1,1,1,0]))
    tr['net_pnl']=[account(r.entry_open,r.exit_open,r.side,r.lots,cfg)['net_pnl'] for r in tr.itertuples()]
    summary,expected=covered_matched_exposure(tr,cfg,100,7)
    np.testing.assert_allclose(expected[:2],tr.net_pnl.to_numpy()[:2])
    unit=[account(100,exit,1,1,cfg)['net_pnl'] for exit in [103,104]]
    np.testing.assert_allclose(expected[2:],np.array(unit)/2)
    assert np.isnan(safe_corr([0,0,0],[1,2,3]))
