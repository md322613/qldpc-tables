"""Coset lifted product codes (Rajpurohit-Lalitha-Garani, arXiv:2610.12782, Def. III.1): GAP construction + code automorphisms.
Automorphisms are blockwise left/right coset actions that preserve both row spaces; orbits give exact case splits."""
import json, subprocess, numpy as np, galois
GF2=galois.GF(2)
def gw(word): return '*'.join([f'pc[{g}]^{e}' for g,e in word]) or 'One(G)'
def gap_build(r):
    words=sorted({json.dumps(w) for row in r['A']+r['B'] for ent in row for w in ent})
    H=','.join(gw(w) for w in r['H'])
    L=[f'G:=SmallGroup({r["G"][0]},{r["G"][1]});; pc:=Pcgs(G);; H:=Subgroup(G,[{H}]);;',
       'reps:=List(RightCosets(G,H),c->Representative(c)^-1);;',
       'idx:=function(y) local i; for i in [1..Length(reps)] do if y^-1*reps[i] in H then return i; fi; od; return fail; end;;',
       'N:=Normalizer(G,H);; m:=Length(reps);; gens:=Concatenation(List(pc),GeneratorsOfGroup(N),GeneratorsOfGroup(Centre(G)));;',
       'Print("{\\"m\\":",m,",\\"words\\":{");']
    for t,wj in enumerate(words):
        L.append(f'g:={gw(json.loads(wj))};; Print("\\"{t}\\":{{\\"L\\":",List(reps,x->idx(g*x)),",\\"R\\":",List(reps,x->idx(x*g)),"}},");')
    L.append('Print("\\"_gens\\":[");')
    L.append('for i in [1..Length(gens)] do g:=gens[i]; Print("{\\"L\\":",List(reps,x->idx(g*x)),",\\"R\\":",List(reps,x->idx(x*g)),",\\"inN\\":",g in N,"}"); if i<Length(gens) then Print(","); fi; od;')
    L.append('Print("]}}\\n"); QUIT;')
    res=subprocess.run(['gap','-q','-o','4g'],input='\n'.join(L),capture_output=True,text=True,timeout=1200)
    D=json.loads(res.stdout.replace(' ','').replace('\n',''))
    perm={words[int(t)]:v for t,v in D['words'].items() if t!='_gens'}
    return D['m'],perm,D['words']['_gens']
def pm(p,m):
    M=np.zeros((m,m),dtype=np.uint8)
    for i,j in enumerate(p): M[j-1,i]^=1
    return M
def assemble(r,m,perm):
    A,B=r['A'],r['B']; nA0,nA1,nB0,nB1=len(A),len(A[0]),len(B),len(B[0])
    el=lambda e,k:(sum(pm(perm[json.dumps(w)][k],m).astype(int) for w in e)%2).astype(np.uint8)
    LA=[[el(A[i][j],'L') for j in range(nA1)] for i in range(nA0)]; RB=[[el(B[k][l],'R') for l in range(nB1)] for k in range(nB0)]
    q1=lambda j,k:(j*nB0+k)*m; q2=lambda i,l:nA1*nB0*m+(i*nB1+l)*m; n=m*(nA1*nB0+nA0*nB1)
    Hx=np.zeros((nA0*nB0*m,n),dtype=np.uint8); Hz=np.zeros((nA1*nB1*m,n),dtype=np.uint8)
    for i in range(nA0):
        for k in range(nB0):
            r0=(i*nB0+k)*m
            for j in range(nA1): Hx[r0:r0+m,q1(j,k):q1(j,k)+m]^=LA[i][j]
            for l in range(nB1): Hx[r0:r0+m,q2(i,l):q2(i,l)+m]^=RB[k][l]
    for j in range(nA1):
        for l in range(nB1):
            r0=(j*nB1+l)*m
            for k in range(nB0): Hz[r0:r0+m,q1(j,k):q1(j,k)+m]^=RB[k][l].T
            for i in range(nA0): Hz[r0:r0+m,q2(i,l):q2(i,l)+m]^=LA[i][j].T
    nblocks=nA1*nB0+nA0*nB1
    return Hx,Hz,nblocks
def automorphism_orbits(Hx,Hz,m,nblocks,gens):
    """Test blockwise L(g) and R(g) permutations; keep those preserving both row spaces. Return orbits (sorted, as lists)."""
    n=Hx.shape[1]; rx=np.linalg.matrix_rank(GF2(Hx)); rz=np.linalg.matrix_rank(GF2(Hz))
    keep=[]
    for g in gens:
        for kind in ('L','R'):
            p=[g[kind][i]-1 for i in range(m)]
            sigma=np.array([b*m+p[i] for b in range(nblocks) for i in range(m)])   # qubit q -> sigma[q]
            inv=np.argsort(sigma)
            ok=np.linalg.matrix_rank(GF2(np.vstack([Hx,Hx[:,inv]])))==rx and np.linalg.matrix_rank(GF2(np.vstack([Hz,Hz[:,inv]])))==rz
            if ok: keep.append(sigma)
    # union-find orbits
    par=list(range(n))
    def f(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    for s in keep:
        for q in range(n): 
            a,b=f(q),f(int(s[q]))
            if a!=b: par[a]=b
    orb={}
    for q in range(n): orb.setdefault(f(q),[]).append(q)
    return sorted((sorted(v) for v in orb.values()),key=lambda v:v[0]),len(keep)
