"""Figures for the second round.
fig2_selection.png  rows = Science, News; columns = (a) never-endorser controls at the real endorsement week, (b) the same
                    users and design 13 weeks earlier (placebo date), (c) controls who endorse the same feed >=13 weeks later.
fig3_timing.png     likes on the feed's posts per endorser, by hour around the moment of endorsing the feed.
Blue solid + circles = endorsers; grey dashed + squares = comparison (dash and marker shape are the secondary encoding)."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BL,GR,INK,GRID,MUT='#2a78d6','#898781','#0b0b0b','#e1e0d9','#52514e'
plt.rcParams.update({'font.size':9,'axes.edgecolor':GRID,'axes.labelcolor':MUT,'xtick.color':MUT,'ytick.color':MUT,'font.family':'DejaVu Sans'})
W=pd.concat([pd.read_csv(f'results/rq3r_weekly_{t}.csv') for t in ['Science','News']]); W=W[(W.K==10)&(W.outcome=='like_any')]
PAN=[('full','same',13,'real','(a) Controls never endorse the feed\nreal endorsement week','Weeks relative to endorsing the feed'),
     ('full','same',13,'placebo','(b) Same users, same design\n13 weeks before they endorse','Weeks relative to the placebo week'),
     ('nyt','all',0,'real','(c) Controls endorse the same feed\nat least 13 weeks later','Weeks relative to endorsing the feed')]
fig,ax=plt.subplots(2,3,figsize=(10.5,5.6),sharex=True,sharey='row',facecolor='#fcfcfb')
for i,f in enumerate(['Science','News']):
    for j,(d,s,S,date,lab,xl) in enumerate(PAN):
        a=ax[i,j]; g=W[(W.feed==f)&(W.design==d)&(W['sample']==s)&(W.S==S)&(W.date==date)].sort_values('k'); a.set_facecolor('#fcfcfb')
        a.axvline(0,color=GRID,lw=6,zorder=0); a.grid(axis='y',color=GRID,lw=.6); a.set_axisbelow(True)
        a.plot(g.k,100*g.control,color=GR,lw=2,ls=(0,(4,2)),marker='s',ms=3.5,label='Matched controls')
        a.plot(g.k,100*g.treated,color=BL,lw=2,marker='o',ms=4,label='Endorsers' if date=='real' else 'Future endorsers')
        for sp in ['top','right']: a.spines[sp].set_visible(False)
        if i==0: a.set_title(lab,fontsize=9,color=INK)
        if j==0: a.set_ylabel(f'{f}\n% liking ≥1 feed post',color=INK)
        if i==1: a.set_xlabel(xl)
        a.text(0.02,0.95,f'n = {int(g.n.iloc[0]):,}',transform=a.transAxes,va='top',fontsize=8,color=MUT); a.set_xticks(range(-8,9,2))
        if i==0: a.legend(frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(0.0,0.9))
fig.suptitle('The gap to never-endorsers is already there 13 weeks before the endorsement (b); against later endorsers it is a short episode (c)',fontsize=10,color=INK,x=0.01,ha='left')
fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig('figures/fig2_selection.png',dpi=170)
H=pd.read_csv('results/a1_timing_hourly.csv'); S=pd.read_csv('results/a1_timing_summary.csv',index_col=0)
fig,ax=plt.subplots(1,3,figsize=(10.5,3.3),sharex=True,facecolor='#fcfcfb')
for j,f in enumerate(['Science','News','#UkrainianView']):
    a=ax[j]; g=H[H.feed==f].sort_values('hour'); a.set_facecolor('#fcfcfb'); a.grid(axis='y',color=GRID,lw=.6); a.set_axisbelow(True)
    a.plot(g.hour+0.5,g.other_likes_per_user,color=GR,lw=1.6,ls=(0,(4,2)),label="Posts of other feeds")
    a.plot(g.hour+0.5,g.target_likes_per_user,color=BL,lw=1.8,label="Posts of the endorsed feed")
    for sp in ['top','right']: a.spines[sp].set_visible(False)
    a.set_title(f"{f}  (n = {int(g.n.iloc[0]):,})",fontsize=9,color=INK); a.set_xlabel('Hours relative to endorsing the feed'); a.set_xticks(range(-72,73,24))
    r=S[f]; a.annotate(f"±1 h: {r['rate_ratio_1h']:.0f}× the same user's\nusual rate at that hour",xy=(0.5,g.target_likes_per_user.max()),xytext=(8,g.target_likes_per_user.max()*0.86),fontsize=8,color=MUT)
    if j==0: a.set_ylabel('Likes per endorser per hour',color=INK)
fig.suptitle('Likes on the feed’s posts spike in the hour around the endorsement and are back to baseline within two hours',fontsize=10,color=INK,x=0.01,ha='left')
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,frameon=False,fontsize=8,loc='upper right',bbox_to_anchor=(0.995,0.905),ncol=2)
fig.tight_layout(rect=(0,0,1,0.88)); fig.savefig('figures/fig3_timing.png',dpi=170); print('saved')
