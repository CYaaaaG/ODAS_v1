import torch
class MemoryBank:
    def __init__(self,size,num_classes,dim,temperature=.07): self.size=size; self.temperature=temperature; self.z=torch.empty(0,dim); self.p=torch.empty(0,num_classes)
    @torch.no_grad()
    def update(self,z,p): self.z=torch.cat([self.z.to(z.device),z.detach()])[-self.size:]; self.p=torch.cat([self.p.to(z.device),p.detach()])[-self.size:]
    def smooth(self,z,p,weight=.9):
        if not len(self.z): return p
        a=(z@self.z.to(z.device).T/self.temperature).softmax(1); return weight*p+(1-weight)*(a@self.p.to(z.device))
