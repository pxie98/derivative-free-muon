"""Independent numerical checks for the measured-data experiment."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

from run_walnut import ROOT, DATA, METHODS, GRID, COORD_GRID, TUNE_SEEDS, TEST_SEEDS, Q, BUDGET
from report_walnut import name


def main():
    raw=loadmat(DATA)
    full_a=raw['A'].tocsr()
    full_b=raw['m'].reshape(-1,order='F')
    assert full_a.shape==(9840,6724) and raw['m'].shape==(82,120)
    assert set(TUNE_SEEDS).isdisjoint(TEST_SEEDS)
    chosen=json.loads((ROOT/'results'/'selected_steps.json').read_text())
    records=[]
    rng=np.random.default_rng(20261024)
    for v in [20,120]:
        rows=np.concatenate([np.arange(82)+82*j for j in range(0,120,120//v)])
        a=full_a[rows]
        b=full_b[rows]
        norm_b2=float(b@b)
        def f(w):
            x=w.reshape(-1,order='F')
            residual=a@x-b
            return (np.dot(residual,residual)+10*np.dot(x,x))/(2*norm_b2)
        # Confirm the vectorization and central slopes against an independent adjoint.
        max_slope_error=0.
        for _ in range(12):
            w=rng.normal(scale=.01,size=(82,82))
            u,z=rng.normal(size=82),rng.normal(size=82)
            h=np.outer(u/np.linalg.norm(u),z/np.linalg.norm(z))
            grad=(a.T@(a@w.ravel(order='F')-b)+10*w.ravel(order='F'))/norm_b2
            expected=float(grad@h.ravel(order='F'))
            slope=(f(w+1e-3*h)-f(w-1e-3*h))/(2e-3)
            max_slope_error=max(max_slope_error,abs(slope-expected))
        assert max_slope_error<1e-10
        for m in METHODS:
            grid=COORD_GRID if m=='coordinate' else GRID
            means={}
            for e in grid:
                vals=[]
                for seed in ([707] if m=='coordinate' else TUNE_SEEDS):
                    stem=name('tune',v,m,seed,e)
                    r=json.loads((ROOT/'results'/(stem+'.json')).read_text())
                    saved=np.load(ROOT/'results'/(stem+'.npz'))
                    np.testing.assert_allclose(f(saved['w']),r['objective'],rtol=1e-12,atol=1e-14)
                    vals.append(r['objective'])
                means[e]=float(np.mean(vals))
            assert min(means,key=means.get)==chosen[str(v)][m]
            for seed in ([901] if m=='coordinate' else TEST_SEEDS):
                stem=name('test',v,m,seed,chosen[str(v)][m])
                r=json.loads((ROOT/'results'/(stem+'.json')).read_text())
                saved=np.load(ROOT/'results'/(stem+'.npz'))
                h=saved['history']; w=saved['w']
                cost=2*(6724 if m=='coordinate' else Q)+1
                assert h[0,0]==1 and h[0,2]==.5
                assert np.all(np.diff(h[:,0])==cost)
                assert r['calls']==1+r['iterations']*cost==h[-1,0]
                assert r['calls']<=BUDGET<r['calls']+cost
                np.testing.assert_allclose(f(w),r['objective'],rtol=1e-12,atol=1e-14)
                np.testing.assert_allclose(h[-1,2],r['objective'],rtol=1e-12)
                np.testing.assert_allclose(np.linalg.norm(a@w.ravel(order='F')-b)/np.linalg.norm(b),r['data_residual'],rtol=1e-12)
                if v==20:
                    mask=np.ones(len(full_b),bool);mask[rows]=False
                    heldout=np.linalg.norm((full_a@w.ravel(order='F')-full_b)[mask])/np.linalg.norm(full_b[mask])
                    np.testing.assert_allclose(heldout,r['heldout_residual'],rtol=1e-12)
                timing=ROOT/'results'/(name('timing',v,m,901,chosen[str(v)][m])+'.npz')
                if seed==901:
                    assert timing.exists(), str(timing)
                    serial=np.load(timing)
                    np.testing.assert_allclose(w,serial['w'],atol=1e-12,rtol=1e-10)
                    np.testing.assert_allclose(h[:,2],serial['history'][:,2],atol=1e-12,rtol=1e-10)
        reference=json.loads((ROOT/'results'/f'reference_{v}.json').read_text())
        wstar=np.load(ROOT/'results'/f'reference_{v}.npz')['w']
        x=wstar.ravel(order='F')
        residual=np.linalg.norm(a.T@(a@x-b)+10*x)/np.linalg.norm(a.T@b)
        assert residual<2e-10
        np.testing.assert_allclose(f(wstar),reference['objective'],rtol=1e-12)
        records.append(dict(views=v,max_slope_error=max_slope_error,reference_residual=residual,status='PASS'))
    result={'status':'PASS','checks':['dataset dimensions','disjoint tuning/test seeds','complete selected runs',
              'step selection','forward-call accounting','recomputed final objectives','fitted and held-out residuals',
              'central difference vs adjoint','CG stationarity','serial replay'], 'cases':records}
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
