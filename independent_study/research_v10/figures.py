import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_v10.state_run import ROOT

R=ROOT/'independent_study/results';OUT=R/'v10/figures'


def main():
    OUT.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    metrics=pd.read_csv(R/'v10/metrics.csv');data=metrics[metrics.stage=='refit']
    models=['raw40','old40_stdp','raw50','readout50_stdp','readout50_random','readout50_shuffled','full50_stdp']
    labels=['Financial base, 40d','Base + STDP, 40d','Financial base, 50d','STDP + readout refit, 50d','Random + readout refit, 50d','Shuffled + readout refit, 50d','Full STDP refit, 50d']
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
    for ax,scope,title in zip(axes,['development','audit'],['Development: Apr 1 - Jun 8','Historical audit: Jun 9 - Sep 15']):
        g=data[data.scope==scope].set_index('model');vals=g.loc[models,'qlike'].to_numpy()
        ax.barh(np.arange(len(models)),vals,color=['#64748b','#94a3b8','#0f766e','#2563eb','#b8c4d2','#b8c4d2','#93b4e6'])
        ax.set_xlim(.25,.282);ax.set_yticks(np.arange(len(models)),labels);ax.set_title(title);ax.set_xlabel('Normalized QLIKE (lower is better; axis starts at 0.25)')
        for i,v in enumerate(vals):ax.text(v+.0002,i,f'{v:.5f}',va='center',fontsize=9)
        ax.grid(axis='x',alpha=.18)
    axes[0].invert_yaxis();fig.suptitle('Recent-data refit helps; a unique STDP increment is not established',fontsize=13);fig.tight_layout();fig.savefig(OUT/'prediction.png');plt.close(fig)
    c=pd.read_csv(R/'v10/calibration.csv');fig,ax=plt.subplots(figsize=(6.4,5))
    for m,label in [('raw40','Financial base, 40d'),('raw50','Financial base, 50d'),('readout50_stdp','STDP + readout refit, 50d')]:
        g=c[(c.scope=='audit')&(c.model==m)];ax.plot(np.sqrt(g.predicted_variance),np.sqrt(g.realized_variance),'o-',label=label)
    limits=[5,45];ax.plot(limits,limits,'--',color='gray',label='Calibrated');ax.set_xlabel('Root mean forecast variance, bp');ax.set_ylabel('Root mean realized variance, bp')
    ax.set_title('Audit calibration by forecast decile\nDescriptive bins; not trading thresholds');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'calibration.png');plt.close(fig)
    economics=pd.read_csv(R/'v10_economics/summary.csv').query('scenario == "base"').set_index('model')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    names=['raw50','readout50_stdp','full50_random','full50_shuffled'];g=economics.loc[names];point=g.variance_saving_fraction.to_numpy()*100
    low=-g.variance_difference_high.to_numpy()/g.matched_path_variance.to_numpy()*100;high=-g.variance_difference_low.to_numpy()/g.matched_path_variance.to_numpy()*100
    axes[0].errorbar(point,np.arange(4),xerr=np.stack([point-low,high-point]),fmt='o',capsize=4)
    axes[0].set_yticks(np.arange(4),['Financial base, 50d','STDP + readout refit','Full random refit','Full shuffled refit']);axes[0].invert_yaxis();axes[0].axvline(0,color='gray',ls='--')
    axes[0].set_xlabel('Path-variance reduction vs matched exposure (%)');axes[0].set_title('Same lot counts within fold / contract / session\n95% five-day blocks; all exploratory dates')
    for name,label in [('fixed_one','Fixed one lot'),('raw50','Financial base, 50d'),('readout50_stdp','STDP + readout refit')]:
        d=pd.read_csv(R/f'v10_economics/base__{name}_daily.csv');axes[1].plot(pd.to_datetime(d.trading_date),d.pnl.cumsum()/10000,label=label)
    axes[1].axhline(0,color='gray',lw=.8);axes[1].set_ylabel('Cumulative net PnL, CNY 10,000');axes[1].set_title('Same CNY 2m capital; one-tick slippage\nExisting direction opportunities; all still lose money');axes[1].legend(fontsize=8);axes[1].tick_params(axis='x',rotation=30)
    fig.tight_layout();fig.savefig(OUT/'economics.png');plt.close(fig)
    s=pd.read_csv(R/'v10/illustrative_state_features.csv');units=[13,3,5];labels=(s[s.unit==units[0]].feature+'@'+s[s.unit==units[0]].delay.astype(str)).tolist()
    values=np.stack([s[s.unit==u].preferred_bin.to_numpy() for u in units]);fig,ax=plt.subplots(figsize=(14,4))
    im=ax.imshow(values,vmin=1,vmax=5,cmap='RdYlBu_r',aspect='auto');ax.set_xticks(np.arange(36),labels,rotation=60,ha='right',fontsize=8);ax.set_yticks(range(3),[f'State {u}' for u in units])
    ax.set_title('One fold / one seed: states chosen by past occupancy, not future success\nPrototype expected amplitude bin (1 = low, 5 = high); @ denotes actual minute delay')
    fig.colorbar(im,ax=ax,label='Expected training-quantile bin',shrink=.7);fig.tight_layout();fig.savefig(OUT/'states.png');plt.close(fig)
    print('Wrote four figures to',OUT)


if __name__=='__main__':main()
