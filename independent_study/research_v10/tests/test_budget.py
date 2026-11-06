import numpy as np
import pandas as pd
from research_v10.budget_core import budget_lots,matched_risk


def test_budget_units_monotonic_and_exact_matched_exposure():
    lots,risk=budget_lots(np.array([25.,100.,400.]),np.full(3,10000.),np.full(3,2500.))
    np.testing.assert_array_equal(lots,[2,1,0]);np.testing.assert_array_equal(risk,[1000,2000,4000])
    assert (lots*risk<=2500).all()
    tr=pd.DataFrame(dict(fold=[0]*3,contract_id=['IC']*3,session=['AM']*3,lots=lots))
    exp,exceed=matched_risk(tr,risk,np.full(3,2500.))
    np.testing.assert_allclose(exp,(5/3)*risk**2)
    np.testing.assert_allclose(exceed,[0,1/3,2/3])
