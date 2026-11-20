import final_config as _final
"""Offline dual foundation-model probabilities with complete fold admission."""
import hashlib
import gc
import os
import shutil
import time
from pathlib import Path
import numpy as np
from chain_features import BudgetExpired
os.environ['TABPFN_MODEL_CACHE_SIZE']='2'
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('TABPFN_DISABLE_TELEMETRY','1')
CHECKPOINTS={
 'tabpfn':(('Prior-Labs/tabpfn_3_5','06bf2ba35c80a92a3b9abb436b99cf49e7a0365e','tabpfn-v3.5-20260909.safetensors','ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3'),),
 'tabpfn_multiclass':(('Prior-Labs/tabpfn_3_5','06bf2ba35c80a92a3b9abb436b99cf49e7a0365e','tabpfn-v3.5-20260909_multiclass.safetensors','4f1e155b09ae1c9860c75cd0a16f811b7431ddac4cb8022ee5e4498070f44149'),),
 'kumo':(('nvidia/Kumo-Tabular','c5893d418e162b9f841ae179dc767afefff73831','large/classifier.pt','cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613'),)}

import sys
_SHARED = getattr(sys, '_sbench_gpu_shared', None)
if _SHARED is None:
    _SHARED = {'paths': {}, 'kumo': None, 'tabpfn': {}}
    sys._sbench_gpu_shared = _SHARED
_PATHS = _SHARED['paths']; _KUMO = None; TRACE = {}; _BROKEN = set()

def available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:return False

def weights_paths(arm='tabpfn'):
    if arm in _PATHS and _PATHS[arm]:
        return _PATHS[arm]
    from huggingface_hub.constants import HF_HUB_CACHE
    exact=[]; reserve=[]; seen=set()
    roots=[]
    for value in (HF_HUB_CACHE, os.environ.get('HF_HOME'), '/root/.cache/huggingface/hub',
                  '/cache', '/hf-cache', Path.home()/'.cache/huggingface/hub', Path(__file__).parent):
        if not value:
            continue
        base=Path(value)
        for candidate in (base, base/'hub'):
            key=str(candidate)
            if key not in seen:
                seen.add(key); roots.append(candidate)
    for repo,revision,filename,digest in CHECKPOINTS[arm]:
        candidates=[]
        for base in roots:
            folder=base/('models--'+repo.replace('/','--'))/'snapshots'
            candidates.extend((folder/revision/filename, base/filename))
            if folder.is_dir():
                candidates.extend(sorted(folder.glob('*/'+filename)))
            # A host cache can be mounted without the canonical HF directory.
            if base.is_dir():
                try:
                    candidates.extend(base.rglob(filename))
                except OSError:
                    pass
        local_seen=set()
        for path in candidates:
            path=Path(path)
            key=str(path)
            if key in local_seen or key in seen:
                continue
            local_seen.add(key)
            try:
                if not path.is_file():
                    continue
                size=path.stat().st_size
            except OSError:
                continue
            if size <= 10*1024*1024:
                continue
            h=hashlib.sha256()
            try:
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(1<<20),b''):
                        h.update(block)
            except OSError:
                continue
            if h.hexdigest()==digest:
                exact.append(str(path))
            else:
                reserve.append(str(path))
    result=[]
    for path in exact:
        if path not in result:
            result.append(path)
    if not result:
        TRACE.setdefault(arm,{'calls':0,'skipped':0,'contexts':[]})['skipped'] += 1
    if result:
        _PATHS[arm] = result
    return result
def has_weights():
    """Admit foundations only when an offline checkpoint passes its digest."""
    try:
        return any(weights_paths(arm) for arm in CHECKPOINTS)
    except ImportError:
        return False

def weights_path():
    paths=weights_paths('tabpfn')
    return paths[0] if paths else None

def _is_safetensors(checkpoint):
    if Path(checkpoint.path).suffix=='.safetensors':return True
    with open(checkpoint.path,'rb') as stream:
        size=int.from_bytes(stream.read(8),'little')
        return 2<=size<=100_000_000 and stream.read(1)==b'{'

def context_rows(X,train,query,categorical,limit=3072,strata=(),seed=20261103):
    """Stratum quotas plus query-marginal importance sampling, without labels.

    A half-population component retains rare training strata. The query part
    uses every query row and categorical frequency ratios, never code medians.
    """
    train=np.asarray(train,dtype=np.int64)
    if len(train)<=limit:return train
    rng=np.random.default_rng(seed)
    cols=list(strata[:2]) or list(range(min(2,categorical)))
    joined=np.concatenate((train,np.asarray(query,dtype=np.int64)))
    _,group=np.unique(X[joined][:,cols],axis=0,return_inverse=True)
    tg=group[:len(train)];qg=group[len(train):];ng=int(group.max())+1
    counts=np.bincount(tg,minlength=ng)
    qc=np.bincount(qg,minlength=ng).astype(float);qc[counts==0]=0
    mass=.5*counts/counts.sum()+.5*(qc/qc.sum() if qc.sum() else counts/counts.sum())
    target=limit*mass;quota=np.minimum(counts,np.floor(target).astype(int))
    while quota.sum()<limit:
        eligible=np.flatnonzero(quota<counts)
        rank=eligible[np.argsort(-(target-quota)[eligible],kind='stable')]
        quota[rank[:min(len(rank),limit-int(quota.sum()))]]+=1
    logw=np.zeros(len(train))
    for col in range(categorical):
        _,inv=np.unique(X[joined,col],return_inverse=True)
        tc=np.bincount(inv[:len(train)]).astype(float)
        qcount=np.bincount(inv[len(train):],minlength=len(tc)).astype(float)
        # minlength of the training vector also includes query-only categories.
        tc=np.pad(tc,(0,max(0,len(qcount)-len(tc))))
        ratio=(qcount+1)/(len(query)+len(tc))/( (tc+1)/(len(train)+len(tc)))
        logw+=np.log(np.clip(ratio[inv[:len(train)]],.2,5.))/max(1,categorical)
    weight=.5+.5*np.exp(logw)
    chosen=[]
    for g in np.flatnonzero(quota):
        idx=np.flatnonzero(tg==g);w=weight[idx];w=w/w.sum()
        chosen.extend(train[rng.choice(idx,int(quota[g]),replace=False,p=w)])
    return np.asarray(chosen,dtype=np.int64)


def _tabpfn(X,y,train,query,K,categorical,path,deadline,estimators=1):
    from tabpfn import TabPFNClassifier
    from tabpfn.checkpoint import Checkpoint
    if not getattr(Checkpoint,'_sbench_header_detection',False):
        Checkpoint.is_safetensors=property(_is_safetensors);Checkpoint._sbench_header_detection=True
    y_tr = y[train]
    classes_tr, counts_tr = np.unique(y_tr, return_counts=True)
    tail_classes = None
    native_ready = ('multiclass' in Path(path).name) and len(classes_tr) <= 160
    fmode = 'fit_preprocessors' if len(query) <= 6144 else 'fit_with_cache'
    def classifier(checkpoint):
        return TabPFNClassifier(model_path=checkpoint,device='cuda',n_estimators=estimators,
            random_state=20261103,categorical_features_indices=list(range(categorical)),
            ignore_pretraining_limits=True,fit_mode=fmode,kv_cache_precision='auto')
    model = classifier(path)
    if native_ready:
        try:
            model.fit(X[train],y_tr)
        except ValueError:
            del model
            gc.collect()
            ordinary = weights_paths('tabpfn')
            if not ordinary:return None
            model = classifier(ordinary[0])
            native_ready = False
    if not native_ready:
        if len(classes_tr) > 10:
            order = np.argsort(-counts_tr, kind='stable')
            top9 = classes_tr[order[:9]]
            tail_classes = classes_tr[order[9:]]
            tail_counts = counts_tr[order[9:]].astype(np.float64) + 0.5
            tail_weights = (tail_counts / tail_counts.sum()).astype(np.float32)
            remap = {int(c): i for i, c in enumerate(top9)}
            y_fit = np.array([remap.get(int(v), 9) for v in y_tr], dtype=np.int64)
        else:
            y_fit = y_tr
        model.fit(X[train],y_fit)
    out=np.zeros((len(query),K),np.float32)
    step = 6144
    for start in range(0,len(query),step):
        if time.monotonic()>=deadline:return None
        p_chunk = model.predict_proba(X[query[start:start+step]])
        if tail_classes is None:
            out[start:start+step,model.classes_.astype(int)] = p_chunk
        else:
            for col_idx, cls_id in enumerate(model.classes_.astype(int)):
                if cls_id < 9:
                    out[start:start+step, int(top9[cls_id])] = p_chunk[:, col_idx]
                else:
                    out[start:start+step][:, tail_classes] = p_chunk[:, col_idx:col_idx+1] * tail_weights[None, :]
    return out

def _kumo_eager():
    """Use the library's CUDA PyTorch implementations without Triton JIT."""
    import importlib
    for name in ('rmsnorm_cast', 'segment_multi_reduce'):
        module = importlib.import_module('sdm._kernels.' + name)
        setattr(module, '_triton_' + name, None)


def _kumo(X,y,train,query,K,categorical,path,deadline,estimators=1):
    global _KUMO
    import torch
    from sdm import TableTensor,Task,Stype
    from sdm.models import KumoTabular
    if not (shutil.which('cc') or shutil.which('gcc') or os.environ.get('CC')):
        _kumo_eager()
    if _SHARED['kumo'] is not None and _SHARED['kumo'][0] == path:
        _KUMO = _SHARED['kumo']
    if _KUMO is None or _KUMO[0]!=path:
        model=KumoTabular(task='classification',size='large',pretrained=False,device='cpu')
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        # NVIDIA publishes the inner classifier state; some re-saves include
        # the outer models.classification prefix. Both load strictly.
        if any(k.startswith('models.') for k in state):model.load_state_dict(state,strict=True,assign=True)
        else:model.models[Task.classification].load_state_dict(state,strict=True,assign=True)
        precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model=model.to(device='cuda',dtype=precision).eval()
        del state
        gc.collect()
        if sys.platform == 'linux':
            try:
                import ctypes
                ctypes.CDLL(None).malloc_trim(0)
            except Exception:
                pass
        _KUMO=(path,model);_SHARED['kumo']=_KUMO
    model=_KUMO[1]
    def table(rows):
        import pandas as pd
        data=pd.DataFrame(np.asarray(X[rows],dtype=np.float32),columns=[str(c) for c in range(X.shape[1])])
        types={str(c):('categorical' if c<categorical else 'numerical') for c in range(X.shape[1])}
        return TableTensor.from_pandas(data,stypes=types,device='cuda')
    target=TableTensor.from_tensor(torch.as_tensor(y[train,None],dtype=torch.int64,device='cuda'))
    out=np.zeros((len(query),K),np.float32)
    step = 6144
    for attempt in range(2):
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                model.fit(table(train),target,num_estimators=estimators,generator=torch.Generator(device='cuda').manual_seed(20261103))
                for start in range(0,len(query),step):
                    if time.monotonic()>=deadline:return None
                    p=model.predict(table(query[start:start+step]))
                    cols=[int(c) for c in p.columns[Stype.numerical]]
                    out[start:start+step,cols]=p.numerical.float().cpu().numpy()
            return out
        except RuntimeError:
            if attempt:
                raise
            _kumo_eager()
            out.fill(0)
        finally:
            model.clear()

def proba(X,y,train,query,K,categorical,deadline,arm='tabpfn',context=3072,estimators=1,strata=(),context_X=None,context_categorical=None):
    if not available() or time.monotonic()>=deadline:return None
    trace=TRACE.setdefault(arm,{'calls':0,'skipped':0,'contexts':[]})
    sample_X = X if context_X is None else context_X
    sample_categories = categorical if context_categorical is None else context_categorical
    train=context_rows(sample_X,train,query,sample_categories,context,strata)
    present=np.unique(y[train]);out=np.zeros((len(query),K),np.float32)
    if len(present)==0:return None
    if len(present)==1:out[:,present[0]]=1.;return out
    weight_arm = 'tabpfn_multiclass' if arm == 'tabpfn' and K > 10 else arm
    if weight_arm not in _PATHS:_PATHS[weight_arm]=weights_paths(weight_arm)
    fn=_tabpfn if arm=='tabpfn' else _kumo
    for path in (_PATHS[weight_arm] or (weights_paths('tabpfn') if arm == 'tabpfn' else ())):
        try:
            out=fn(X,y,train,query,K,categorical,path,deadline,estimators)
            if out is None:return None
            if out.shape!=(len(query),K) or not np.isfinite(out).all() or np.any(out.sum(1)<=0):
                trace['skipped'] += 1;continue
            trace['calls']+=1;trace['contexts'].append(len(train));trace['checkpoint']=Path(path).name
            trace.setdefault('checkpoint_calls', {})[Path(path).name] = trace.setdefault('checkpoint_calls', {}).get(Path(path).name, 0) + 1
            return out/out.sum(1,keepdims=True)
        except Exception as exc:
            trace['skipped'] += 1
    return None


def candidate(X,y,rows,rest,K,categorical,fold,deadline,timing,strata=(),feature_factory=None,capture=False,arm='tabpfn',n_folds=None):
    remaining=max(1,timing.get('remaining',1));cost=max(.05,timing['recent_fit'])
    per_item=(deadline-time.monotonic())/remaining
    generous = _final.enabled('FOUNDATION_5FOLD') and (timing.get('phase') == 'final' or per_item > 8.5 * cost)
    survey_key = (len(X), X.shape[1], timing.get('phase'))
    if timing.get('survey_key') != survey_key:
        timing.pop('survey_n_folds', None)
        timing['survey_key'] = survey_key
    if generous:
        n_folds = 5
        timing['survey_n_folds'] = 5
        timing['last_n_folds'] = 5
        context = 6144
        estimators = ((4 if per_item > 2.5 else (3 if per_item > 1.8 else 2))
                      if arm == 'tabpfn' else (2 if per_item > 1.4 else 1))
    else:
        if X.shape[1]>=17 and len(X)>=8000:
            n_folds = int(timing.setdefault('survey_n_folds', 2))
            timing['last_n_folds'] = int(n_folds)
            context=2560 if per_item>3.5*cost else 2048
            estimators=2 if per_item>4.5*cost else 1
        elif len(X)<8000 and X.shape[1]>=9:
            n_folds = int(timing.setdefault('survey_n_folds', 2))
            timing['last_n_folds'] = int(n_folds)
            context=3584 if per_item>3.5*cost else 2560
            estimators=2 if per_item>4.5*cost else 1
        else:
            n_folds = int(timing.setdefault('survey_n_folds', 5 if per_item > 7.5 * cost else 2))
            timing['last_n_folds'] = int(n_folds)
            context=6144 if per_item>(n_folds+2.5)*cost else 3584
            estimators=3 if per_item>2.5*(n_folds+1)*cost else 1
    long_final = timing.get('long_final', False)
    if time.monotonic()+(.08 if long_final else max(.6,(n_folds+.3)*cost))>=deadline:return None
    fold_arr=np.asarray(fold,dtype=np.int64)
    fold_id=np.where(fold_arr[rows]>=0, fold_arr[rows] % n_folds, -1)
    if np.any(fold_id<0) or len(np.unique(fold_id))!=n_folds:return None
    o=np.zeros((len(rows),K),np.float32);full=np.zeros((len(X),K),np.float32);partitions={}
    def call(train,query,part):
        if time.monotonic()+(.03 if long_final else max(.08,1.15*timing['recent_fit']))>=deadline:return None
        began=time.monotonic()
        try:
            features=X if feature_factory is None else feature_factory(train,part)
            model_categories = categorical
            if isinstance(features, tuple):
                features, model_categories = features
            model_categories = int(model_categories)
            if not 0 <= model_categories <= features.shape[1]:
                raise ValueError('categorical boundary outside feature matrix')
            p=proba(features,y,train,query,K,model_categories,deadline,arm=arm,
                    context=context,estimators=estimators,strata=strata,
                    context_X=X,context_categorical=categorical)
            if p is None:return None
            return p
        finally:
            if timing['calls']>=1:timing['recent_fit']=.7*timing['recent_fit']+.3*(time.monotonic()-began)
            timing['calls']+=1
    for f in range(n_folds):
        mask=fold_id==f;train=rows[~mask];valid=rows[mask]
        if len(train)<4 or not len(valid):return None
        query = np.concatenate((valid, rest)) if len(rest) else valid
        pred=call(train,query,f)
        if pred is None:return None
        o[mask]=pred[:len(valid)]
        if len(rest):
            full[rest] += pred[len(valid):] / n_folds
    if len(rest) and per_item > (n_folds + 1.4) * cost and time.monotonic() + max(0.1, 1.3 * timing['recent_fit']) < deadline:
        pred_all = call(rows, rest, 'all')
        if pred_all is not None:
            full[rest] = 0.5 * full[rest] + 0.5 * pred_all
    if capture:
        full_oof=full.copy();full_oof[rows]=o
        partitions['all']=full_oof
        timing['partitions']=partitions
    timing.setdefault('completed',[]).append({'arm':arm,'context':context,'estimators':estimators,'folds':n_folds,'rows':len(rows)})
    return o,full

