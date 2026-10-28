"""Camera-only restriction of an already fitted transfer rule; identity is never read."""
import numpy as np

def restricted_weights(g, cameras, kappa, gate_predict, standardize, context, softmax, original_weights=None):
    j,f=g['J'],g['Fi'];n,m=len(j),len(f)
    if not n or not m:return np.zeros((n,m)),np.zeros(n,dtype=bool)
    cameras=np.asarray(cameras)
    mask=cameras[j,None]!=cameras[f][None,:]
    available=mask.any(axis=1)
    if mask.all() and original_weights is not None:
        return original_weights.copy(),available
    cf=g['CF'].copy();cf[~mask]=-9.0
    x=g['X'].copy();x[:,0]=cf.max(axis=1);x[:,5]=x[:,0]-x[:,3]
    association=softmax(kappa*cf,mask)
    probability=gate_predict(np.c_[standardize(x)[:,[0,5]],np.tile(context,(n,1))])[:,1]
    weights=probability[:,None]*association
    weights[~available]=0
    return weights,available

def coverage(g, pids, cameras):
    j,f=g['J'],g['Fi'];pids=np.asarray(pids);cameras=np.asarray(cameras)
    same=pids[j,None]==pids[f][None,:]
    any_source=same.any(axis=1)
    xsource=(same&(cameras[j,None]!=cameras[f][None,:])).any(axis=1)
    return dict(ir_only_items=len(j),same_identity_complete_any_camera=int(any_source.sum()),
                same_identity_complete_other_camera=int(xsource.sum()),same_identity_sources_only_same_camera=int((any_source&~xsource).sum()))
