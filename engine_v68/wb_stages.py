"""Breadth-first boosting: publish a complete cross-fit before deepening it."""
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits
from .chain_features import BudgetExpired, check_clock


def run(names, work, bank, split, profile, deadline, workers):
    cache = {}; completed = {}; history = {}; costs = []
    max_step = 60 if profile in ('catboost', 'hgb') else 45 if profile == 'deep' else 35
    steps = list(range(10, max_step+1, 10))
    if steps[-1] != max_step: steps.append(max_step)

    def one(name, step):
        check_clock(deadline)
        j = work[name]; y = j['y']; K = j['K']; rows = np.flatnonzero(j['train']); rest = np.flatnonzero(~j['train'])
        if name not in cache:
            parts = []
            for f in range(2):
                check_clock(deadline)
                tr = rows[split[rows] != f]; va = rows[split[rows] == f]; q = np.r_[va,rest]
                X,c = bank.matrix(name,tr)
                parts.append(dict(X=X,c=c,tr=tr,q=q,va=va,mask=split[rows]==f,model=None,step=0))
            cache[name] = parts
        o = np.zeros((len(rows),K),np.float32); full = np.zeros((len(y),K),np.float32)
        for part in cache[name]:
            check_clock(deadline)
            X,c,tr,q=part['X'],part['c'],part['tr'],part['q']
            model = part['model']; classes = np.unique(y[tr]); p = np.zeros((len(q),K),np.float32)
            if len(classes) < 2:
                if len(classes):p[:, int(classes[0])] = 1.
                else:p[:] = 1./K
            elif profile == 'catboost':
                import pandas as pd
                from catboost import CatBoostClassifier,Pool
                if 'fit_pool' not in part:
                    data = pd.DataFrame(X)
                    for col in range(c):data[col]=data[col].astype(np.int32)
                    part['fit_pool']=Pool(data.iloc[tr],y[tr],cat_features=list(range(c)),thread_count=1)
                    part['query_pool']=Pool(data.iloc[q],cat_features=list(range(c)),thread_count=1)
                update=CatBoostClassifier(iterations=step-part['step'],depth=4,learning_rate=.1,l2_leaf_reg=2.,thread_count=1,verbose=False,allow_writing_files=False,one_hot_max_size=65535,random_seed=20261020,task_type='CPU')
                class Clock:
                    def after_iteration(self,info):return time.monotonic()<deadline
                update.fit(part['fit_pool'],init_model=model,callbacks=[Clock()])
                check_clock(deadline);model=update
                p[:,model.classes_.astype(int)]=model.predict_proba(part['query_pool'],thread_count=1)
            elif profile == 'hgb':
                from .chain_features import _fit_hgb
                if model is None:
                    model=_fit_hgb(X[tr],y[tr],c,step);model.set_params(warm_start=True)
                else:
                    model.set_params(max_iter=step);model.fit(X[tr],y[tr])
                check_clock(deadline);p[:,model.classes_.astype(int)]=model.predict_proba(X[q])
            else:
                from lightgbm import LGBMClassifier
                update=LGBMClassifier(n_estimators=step-part['step'],num_leaves=31 if profile=='deep' else 15,learning_rate=.06 if profile=='deep' else .1,min_child_samples=20,reg_lambda=2.,n_jobs=1,verbosity=-1,metric='None',deterministic=True,force_col_wise=True,random_state=20261020)
                update.fit(X[tr],y[tr],categorical_feature=list(range(c)),init_model=None if model is None else model.booster_,callbacks=[lambda env: check_clock(deadline)])
                check_clock(deadline);model=update
                p[:,model.classes_.astype(int)]=model.predict_proba(X[q])
            part['model']=model;part['step']=step
            o[part['mask']]=p[:len(part['va'])];full[rest]+=p[len(part['va']):]/2
        return o,full

    with ThreadPoolExecutor(max_workers=workers,initializer=lambda:threadpool_limits(limits=1,user_api='openmp')) as pool:
        for step in steps:
            def guarded(name):
                began=time.monotonic()
                try: value=one(name,step)
                except BudgetExpired:return name,None
                costs.append(time.monotonic()-began)
                return name,value
            for name,value in pool.map(guarded,names):
                if value is not None:
                    completed[name]=value
                    history.setdefault(name,[]).append(step)
            if time.monotonic()>=deadline:break
    return completed, history, costs
