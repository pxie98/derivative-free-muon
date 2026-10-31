"""Function-value matrix optimization on measured walnut CT projections.

The optimizer receives only a scalar objective callable. Each call applies
the published sparse forward operator once; no quadratic shortcuts are used.
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 for comparable CPU timings.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.sparse.linalg import LinearOperator, cg

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'Data82.mat'
MD5 = '5698942708300bc34b87931c6d91f6b6'
METHODS = ['rankone', 'orthogonal', 'sphere', 'zo_sgd', 'coordinate']
GRID = [.001, .003, .01, .03, .1]
COORD_GRID = GRID + [.3, 1.]
TUNE_SEEDS = [707, 808]
TEST_SEEDS = [901, 902, 903, 904, 905]
BUDGET = 20000
Q = 32
RADIUS = 1e-3
BETA = .9


def problem(views):
    assert hashlib.md5(DATA.read_bytes()).hexdigest() == MD5
    data = loadmat(DATA)
    full_a = data['A'].tocsr()
    full_b = data['m'].ravel(order='F')
    n = int(np.sqrt(full_a.shape[1]))
    cols = np.arange(0, 120, 120 // views)
    rows = (n * cols[:, None] + np.arange(n)[None, :]).ravel()
    a, b = full_a[rows], full_b[rows]
    scale = float(b @ b)
    def objective(w):
        x = w.ravel(order='F')
        r = a @ x - b
        return float((r @ r + 10 * (x @ x)) / (2 * scale))
    return n, a, b, full_a, full_b, objective


def polar(m):
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    keep = s > np.finfo(float).eps * max(m.shape) * s[0]
    return u[:, keep] @ vt[keep]


def optimize(objective, n, method, seed, eta0, budget=BUDGET):
    rng = np.random.default_rng(seed)
    w = np.zeros((n, n))
    momentum = w.copy()
    calls = 0
    def query(z):
        nonlocal calls
        calls += 1
        return objective(z)
    start = time.perf_counter()
    f = query(w)
    history = [[calls, 0, f, time.perf_counter()-start]]
    q = n*n if method == 'coordinate' else Q
    t = 0
    polar_seconds = 0.
    while calls + 2*q + 1 <= budget:
        g = np.zeros_like(w)
        if method == 'orthogonal':
            left = np.linalg.qr(rng.normal(size=(n,n)))[0]
            right = np.linalg.qr(rng.normal(size=(n,n)))[0]
            atoms = rng.choice(n*n, q, replace=False)
        for k in range(q):
            if method == 'coordinate':
                h = np.zeros_like(w)
                h.flat[k] = 1.
            elif method == 'sphere':
                h = rng.normal(size=w.shape)
                h /= np.linalg.norm(h)
            elif method == 'orthogonal':
                i, j = divmod(int(atoms[k]), n)
                h = np.outer(left[:, i], right[:, j])
            else:
                u, v = rng.normal(size=n), rng.normal(size=n)
                h = np.outer(u/np.linalg.norm(u), v/np.linalg.norm(v))
            slope = (query(w + RADIUS*h) - query(w - RADIUS*h))/(2*RADIUS)
            g += slope*h
        g *= n*n/q
        momentum = BETA*momentum + (1-BETA)*g
        clock = time.perf_counter()
        if method == 'zo_sgd':
            norm = np.linalg.norm(momentum)
            direction = np.sqrt(n)*momentum/norm if norm else momentum.copy()
        else:
            direction = polar(momentum)
        polar_seconds += time.perf_counter()-clock
        eta = eta0/np.sqrt(1+t/100)
        w -= eta*direction
        f = query(w)
        t += 1
        history.append([calls, t, f, time.perf_counter()-start])
    return w, np.asarray(history), polar_seconds


def run_job(job):
    views, method, seed, eta, phase, budget = job
    path = ROOT / 'results' / f'{phase}_{views}_{method}_{seed}_{eta:g}'.replace('.', 'p')
    if path.with_suffix('.json').exists():
        return json.loads(path.with_suffix('.json').read_text())
    n, a, b, full_a, full_b, objective = problem(views)
    w, history, polar_seconds = optimize(objective, n, method, seed, eta, budget)
    x = w.ravel(order='F')
    result = dict(views=views, method=method, seed=seed, eta=eta, phase=phase,
                  calls=int(history[-1,0]), iterations=int(history[-1,1]),
                  objective=float(history[-1,2]), seconds=float(history[-1,3]),
                  polar_seconds=polar_seconds,
                  data_residual=float(np.linalg.norm(a@x-b)/np.linalg.norm(b)),
                  all_view_residual=float(np.linalg.norm(full_a@x-full_b)/np.linalg.norm(full_b)))
    if views < 120:
        mask = np.ones(len(full_b), bool)
        mask[(n*np.arange(0,120,120//views)[:,None]+np.arange(n)).ravel()] = False
        result['heldout_residual'] = float(np.linalg.norm((full_a@x-full_b)[mask])/np.linalg.norm(full_b[mask]))
    np.savez_compressed(path.with_suffix('.npz'), w=w, history=history)
    path.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def reference(views):
    n, a, b, full_a, full_b, objective = problem(views)
    calls = {'forward':0, 'adjoint':0}
    trajectory = []
    start = time.perf_counter()
    def hessian(x):
        calls['forward'] += 1
        calls['adjoint'] += 1
        return a.T @ (a@x) + 10*x
    rhs = a.T@b
    calls['adjoint'] += 1
    def callback(x):
        trajectory.append(x.copy())
    operator = LinearOperator((n*n,n*n), matvec=hessian, dtype=float)
    x, info = cg(operator, rhs, rtol=1e-10, atol=0, maxiter=5000, callback=callback)
    seconds = time.perf_counter()-start
    residual = np.linalg.norm(a.T@(a@x-b)+10*x)/np.linalg.norm(rhs)
    assert info == 0 and residual < 2e-10
    # All post-solve evaluations are diagnostics, excluded from solver timing.
    fstar = objective(x.reshape(n,n,order='F'))
    norm_b2 = float(b@b)
    # Strong convexity gives a certified upper bound on objective suboptimality.
    bound = np.linalg.norm(a.T@(a@x-b)+10*x)**2/(20*norm_b2)
    result = dict(views=views, objective=fstar, seconds=seconds,
                  iterations=len(trajectory), **calls,
                  relative_normal_residual=float(residual), objective_error_bound=float(bound),
                  data_residual=float(np.linalg.norm(a@x-b)/np.linalg.norm(b)),
                  all_view_residual=float(np.linalg.norm(full_a@x-full_b)/np.linalg.norm(full_b)))
    if views < 120:
        mask=np.ones(len(full_b),bool)
        mask[(n*np.arange(0,120,120//views)[:,None]+np.arange(n)).ravel()]=False
        result['heldout_residual']=float(np.linalg.norm((full_a@x-full_b)[mask])/np.linalg.norm(full_b[mask]))
    path = ROOT/'results'/f'reference_{views}'
    np.savez_compressed(path.with_suffix('.npz'), w=x.reshape(n,n,order='F'),
                        objective_history=[objective(z.reshape(n,n,order='F')) for z in trajectory])
    path.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--phase',choices=['all','tune','test','reference','timing'],default='all')
    args=parser.parse_args()
    (ROOT/'results').mkdir(exist_ok=True)
    if args.phase in ['all','reference']:
        for views in [20,120]: print('reference',reference(views),flush=True)
    if args.phase in ['all','tune']:
        jobs=[(v,m,s,e,'tune',BUDGET) for v in [20,120] for m in METHODS
              for e in (COORD_GRID if m=='coordinate' else GRID) for s in (TUNE_SEEDS if m!='coordinate' else [707])]
        with ProcessPoolExecutor(args.workers) as pool:
            for r in pool.map(run_job,jobs):
                print('tune',r['views'],r['method'],r['seed'],r['eta'],r['objective'],flush=True)
        chosen={}
        for v in [20,120]:
            chosen[str(v)]={}
            for m in METHODS:
                means={}
                for e in (COORD_GRID if m=='coordinate' else GRID):
                    values=[json.loads((ROOT/'results'/(f'tune_{v}_{m}_{s}_{e:g}'.replace('.', 'p')+'.json')).read_text())['objective']
                            for s in (TUNE_SEEDS if m!='coordinate' else [707])]
                    means[e]=float(np.mean(values))
                chosen[str(v)][m]=min(means,key=means.get)
        (ROOT/'results'/'selected_steps.json').write_text(json.dumps(chosen,indent=2)+'\n')
        print('Selected',chosen,flush=True)
    if args.phase in ['all','test']:
        chosen=json.loads((ROOT/'results'/'selected_steps.json').read_text())
        jobs=[(v,m,s,chosen[str(v)][m],'test',BUDGET) for v in [20,120] for m in METHODS
              for s in (TEST_SEEDS if m!='coordinate' else [901])]
        with ProcessPoolExecutor(args.workers) as pool:
            for r in pool.map(run_job,jobs):
                print('test',r['views'],r['method'],r['seed'],r['objective'],flush=True)
    if args.phase == 'timing':
        chosen=json.loads((ROOT/'results'/'selected_steps.json').read_text())
        for v in [20,120]:
            for m in METHODS:
                result=run_job((v,m,901,chosen[str(v)][m],'timing',BUDGET))
                print('timing',v,m,result['seconds'],flush=True)
    cpu = platform.processor()
    if platform.system() == 'Darwin':
        try:
            cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                          text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    environment=dict(cpu=cpu,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                     platform=platform.platform(),machine=platform.machine(), workers=args.workers,
                     blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),budget=BUDGET,
                     q=Q,radius=RADIUS,beta=BETA,grid=GRID,coordinate_grid=COORD_GRID,tune_seeds=TUNE_SEEDS,test_seeds=TEST_SEEDS,
                     data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest())
    (ROOT/('timing_environment.json' if args.phase=='timing' else 'environment.json')).write_text(json.dumps(environment,indent=2)+'\n')


if __name__=='__main__': main()
