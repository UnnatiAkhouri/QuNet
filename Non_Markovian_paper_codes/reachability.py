import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product

"""
This code uses the PSWAP transition matrix code to compute the reachability of a Pauli operator
under repeated action of PSWAP in brickwork
"""
# Transition matrix (θ=π/4) CHANGE THE VALUES HERE FOR DIFFERENT PSWAP ANGLES
c,s,c2,s2,s2t = 0.707,0.707,0,1,0.5
T = np.zeros((16,16)); T[[0,5,10,15],[0,5,10,15]]=1
T[1,1]=c;T[1,11]=-s;T[2,2]=c;T[2,7]=s;T[3,3]=c2;T[3,6]=-s2t;T[3,9]=s2t;T[3,12]=s2
T[4,4]=c;T[4,14]=-s;T[6,3]=s2t;T[6,6]=c2;T[6,9]=-s2t;T[6,12]=s2;T[7,2]=-s;T[7,7]=c
T[8,8]=c;T[8,13]=s;T[9,3]=-s2t;T[9,6]=s2t;T[9,9]=c2;T[9,12]=s2;T[11,1]=s;T[11,11]=c
T[12,3]=s2;T[12,6]=s2t;T[12,9]=-s2t;T[12,12]=c2;T[13,8]=-s;T[13,13]=c;T[14,4]=s;T[14,14]=c
P = {p:i for i,p in enumerate(['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ'])}
pairs = lambda s,e: [s[:2],s[2:4],s[4:]] if e else [s[5]+s[0],s[1:3],s[3:5]]

def evolve(s,e):
    p=pairs(s,e); r={}
    for i0 in range(16):
        if abs(T[P[p[0]],i0])<1e-10:continue
        for i1 in range(16):
            if abs(T[P[p[1]],i1])<1e-10:continue
            for i2 in range(16):
                if abs(T[P[p[2]],i2])<1e-10:continue
                ps=[list(P.keys())[i]for i in[i0,i1,i2]]; o=''.join(ps)if e else ps[0][1]+ps[1]+ps[2]+ps[0][0]
                r[o]=r.get(o,0)+T[P[p[0]],i0]*T[P[p[1]],i1]*T[P[p[2]],i2]
    return{k:v for k,v in r.items()if abs(v)>1e-10}

def find_reachable(start):
    reach,curr,e={start},{start},1
    for _ in range(20):
        nxt={}
        for s in curr:
            for n,w in evolve(s,e).items():reach.add(n);nxt[n]=nxt.get(n,0)+w
        curr={k:v for k,v in nxt.items()if abs(v)>1e-10};e=1-e
        if not curr:break
    return reach

# Single string analysis
start = 'ZIIIII'  # You can plug in which ever Pauli word's reachability you want to check: 'IIIIII', 'XXXXXX', 'XYXYXY', etc.
G,reach,curr,e=nx.DiGraph(),{start},{start},1
for _ in range(15):
    nxt={}
    for s in curr:
        for n,w in evolve(s,e).items():reach.add(n);nxt[n]=nxt.get(n,0)+w;G.add_edge(s,n,weight=w)
    curr={k:v for k,v in nxt.items()if abs(v)>1e-10};e=1-e
    if not curr:break

print(f"Single string '{start}': {len(reach)}/{4**6} reachable ({100*len(reach)/4096:.1f}%)")

# I/Z coverage analysis
#iz_strings = [''.join(p) for p in product('IZ', repeat=6)]
#all_iz_reach = set()
#for s in iz_strings:
#    all_iz_reach.update(find_reachable(s))

#xy_even = {w for w in [''.join(p) for p in product('IXYZ',repeat=6)] if (w.count('X')+w.count('Y'))%2==0}
#print(f"\nAll 64 I/Z strings combined: {len(all_iz_reach)}/2048 of XY-even ({100*len(all_iz_reach)/2048:.1f}%)")
#print(f"XY-even space: 2048 | I/Z union: {len(all_iz_reach)} | Missing: {len(xy_even)-len(all_iz_reach)}")

# Viz
pos=nx.spring_layout(G,k=1.5,iterations=50)
nx.draw(G,pos,node_color=['red'if n==start else'lightblue'for n in G],node_size=200,
        with_labels=True,font_size=5,arrows=True,alpha=0.7,edge_color='gray',width=0.5)
plt.title(f"{len(reach)} reachable from {start} | I/Z union: 940/2048")
plt.savefig('pauli_graph_ZIIIII.png',dpi=200,bbox_inches='tight')
print("✓ Graph saved")