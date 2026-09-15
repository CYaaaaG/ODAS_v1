import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
def open_world_metrics(y,pred,seen=(0,1,2)):
    y,pred=np.asarray(y),np.asarray(pred); known=np.isin(y,seen); novel=~known; out={'known_accuracy':float(accuracy_score(y[known],pred[known])) if known.any() else 0.0}
    if novel.any():
        yt,yp=y[novel],pred[novel]; a,b=np.unique(yt),np.unique(yp); cost=np.zeros((len(a),len(b)),int)
        for i,c in enumerate(a):
            for j,d in enumerate(b): cost[i,j]=-np.sum((yt==c)&(yp==d))
        r,col=linear_sum_assignment(cost); mp={b[j]:a[i] for i,j in zip(r,col)}; out['novel_clustering_accuracy']=float(accuracy_score(yt,[mp.get(v,-1) for v in yp]))
    else: out['novel_clustering_accuracy']=0.0
    out['overall_accuracy']=float(accuracy_score(y,pred)); p1,r,f,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0); out.update(macro_precision=float(p1),macro_recall=float(r),macro_f1=float(f),confusion_matrix=confusion_matrix(y,pred).tolist()); return out
