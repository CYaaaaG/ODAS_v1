import torch
def transform(x,cfg,strong=False):
    a=cfg.get('augmentation',{}); y=x.clone(); noise=a.get('strong_noise' if strong else 'weak_noise',0.0)
    if a.get('horizontal_flip',True) and torch.rand(())<(0.5 if strong else 0.2): y=y.flip(-1)
    m=int(a.get('max_shift',2))
    if m and torch.rand(())<.5: y=torch.roll(y,int(torch.randint(-m,m+1,(1,))),dims=-1)
    return (y+float(noise)*torch.randn_like(y)).clamp(0,1)
