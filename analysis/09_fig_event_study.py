"""Figure 1: weekly probability of liking >=1 post surfaced by the endorsed feed, endorsers vs matched controls.
Rows = feeds, columns = control definitions. Blue solid = endorsers, grey dashed = controls (shape + dash as secondary encoding)."""
import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
E=pd.read_csv('results/rq3_event_study.csv'); E=E[E.outcome=='like_any']
FEEDS=['Science','News','#UkrainianView']; SPECS=[('early','Controls matched on history\nup to week −5'),('full','Controls matched on history\nup to week −1'),('active','Controls endorsed a different\nfeed in the same week')]
BL,GR,INK,GRID='#2a78d6','#898781','#0b0b0b','#e1e0d9'
plt.rcParams.update({'font.size':9,'axes.edgecolor':GRID,'axes.labelcolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','font.family':'DejaVu Sans'})
fig,ax=plt.subplots(3,3,figsize=(10.5,7.6),sharex=True,sharey='row',facecolor='#fcfcfb')
for i,f in enumerate(FEEDS):
    for j,(s,lab) in enumerate(SPECS):
        a=ax[i,j]; d=E[(E.feed==f)&(E.spec==s)].sort_values('k'); a.set_facecolor('#fcfcfb')
        a.axvline(0,color=GRID,lw=6,zorder=0); a.grid(axis='y',color=GRID,lw=.6); a.set_axisbelow(True)
        a.plot(d.k,100*d.control,color=GR,lw=2,ls=(0,(4,2)),marker='s',ms=3.5,label='Matched controls')
        a.plot(d.k,100*d.treated,color=BL,lw=2,marker='o',ms=4,label='Endorsers')
        for sp in ['top','right']: a.spines[sp].set_visible(False)
        if i==0: a.set_title(lab,fontsize=9,color=INK)
        if j==0: a.set_ylabel(f'{f}\n% liking ≥1 feed post',color=INK)
        if i==2: a.set_xlabel('Weeks relative to endorsing the feed')
        a.text(0.02,0.95,f'n = {int(d.n.iloc[0]):,}',transform=a.transAxes,va='top',fontsize=8,color='#52514e')
        a.set_xticks(range(-8,9,2))
ax[0,2].legend(frameon=False,fontsize=8,loc='upper right')
fig.suptitle('Engagement with a feed rises before users endorse it, peaks in the endorsement week (shaded), then declines; matched controls regress faster',fontsize=10,color=INK,x=0.01,ha='left')
fig.tight_layout(rect=(0,0,1,0.96)); fig.savefig('figures/fig1_event_study.png',dpi=170); print('saved')
