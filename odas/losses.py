import torch
import torch.nn.functional as F
def soft_ce(target,logits): return -(target.detach()*F.log_softmax(logits,1)).sum(1).mean()
def ucm_loss(logits,z,threshold=.5):
    if len(z)<2:return logits.sum()*0
    s=z@z.T; s.fill_diagonal_(-1); j=s.argmax(1); keep=s[torch.arange(len(z),device=z.device),j]>=threshold
    return soft_ce(logits[j[keep]].softmax(1),logits[keep]) if keep.any() else logits.sum()*0
def graph_loss(q,z,threshold=.7,temperature=.07):
    if len(q)<2:return z.sum()*0
    t=q@q.T; t.fill_diagonal_(1); t=torch.where(t>=threshold,t,torch.zeros_like(t)); t=t/t.sum(1,keepdim=True).clamp_min(1e-8); p=(z@z.T/temperature).softmax(1)
    return -(t.detach()*p.clamp_min(1e-8).log()).sum(1).mean()
def cacm_loss(q,z1,z2,threshold=.9,temperature=.07):
    conf,cls=q.max(1); t=((cls[:,None]==cls[None,:])&(conf[:,None]>=threshold)&(conf[None,:]>=threshold)).float(); t[torch.arange(len(q)),torch.arange(len(q))]=1; t=t/t.sum(1,keepdim=True).clamp_min(1e-8); p=(z1@z2.T/temperature).softmax(1)
    return -(t.detach()*p.clamp_min(1e-8).log()).sum(1).mean()
