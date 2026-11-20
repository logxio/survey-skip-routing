"""Observed-category interactions fitted on each training partition."""
from itertools import combinations
import numpy as np

def interaction_columns(width, names=None):
    # Eight inexpensive legacy crosses plus named survey interactions.
    pairs=[(a,b) for a in range(min(2,width)) for b in range(a+1,width)][:8]
    names=list(names or ())
    groups=[('wave','governorate_col'),
            ('wave','s2_4a_origin_governorate_lbl'),
            ('s1_2_country_asylum_lbl','governorate_col'),
            ('s1_2_country_asylum_lbl','s2_4a_origin_governorate_lbl'),
            ('governorate_col','s2_4a_origin_governorate_lbl'),
            ('s1_2_country_asylum_lbl','governorate_col','s2_4a_origin_governorate_lbl'),
            ('user.educationLevel','user.age'),('user.gender','user.educationLevel'),
            ('user.country','user.educationLevel'),('user.country','user.age'),
            ('user.country','user.educationLevel','user.age'),
            ('Country','Q13'),('Q02','Q13'),('Q03','Q13'),
            ('Country','Q03','Q13'),('Country','Q02','Q13')]
    for group in groups:
        if all(n in names for n in group):
            cols=tuple(sorted(names.index(n) for n in group))
            if cols not in pairs:pairs.append(cols)
    return pairs[:14 if 'governorate_col' in names else 15]

def compact_codes(X, train, columns, min_support=None):
    if min_support is None:min_support=max(10,int(.005*len(train)))
    # Schema integer codes; mixed radix uses no target and never fits query rows.
    keys=np.zeros(len(X),np.int64)
    for c in columns:keys=keys*(int(X[:,c].max())+1)+X[:,c]
    values,counts=np.unique(keys[train],return_counts=True)
    values=values[counts>=min_support]
    pos=np.searchsorted(values,keys);hit=pos<len(values)
    if len(values):hit &= values[np.minimum(pos,len(values)-1)]==keys
    return np.where(hit,pos+1,0).astype(np.int32),len(values)+1

def categorical_matrix(X,train,names=None):
    pieces=[X.astype(np.float32)]
    for columns in interaction_columns(X.shape[1],names):
        codes,_=compact_codes(X,train,columns)
        pieces.append(codes[:,None])
    return np.column_stack(pieces)

def sparse_design(X,levels,train,names=None,max_columns=600,retain_crosses=False):
    from scipy import sparse
    columns=[];width=0
    for f,size in enumerate(levels):columns.append(X[:,f]+width);width+=int(size)
    singles=width
    for group in interaction_columns(X.shape[1],names):
        code,size=compact_codes(X,train,group)
        columns.append(code+width);width+=size
    c=np.column_stack(columns)
    M=sparse.csr_matrix((np.ones(c.size,np.float32),(np.repeat(np.arange(len(X)),c.shape[1]),c.ravel())),shape=(len(X),width))
    if retain_crosses:return M
    support=np.asarray(M[train].sum(0)).ravel()
    ix=np.arange(singles,width)
    ix=ix[np.argsort(-support[ix],kind='stable')[:max(0,max_columns-singles)]]
    return M[:,np.r_[np.arange(singles),np.sort(ix)]].tocsr()
