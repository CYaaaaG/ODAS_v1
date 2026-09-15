import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from odas.data import load_config, CaoDataset, make_open_world_indices
from odas.augment import transform
from odas.model import ODASNet
from odas.memory import MemoryBank
from odas.losses import soft_ce, ucm_loss, graph_loss, cacm_loss
from odas.evaluation import open_world_metrics

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def evaluate(model, root, cfg, device):
    ds=CaoDataset(root,'test',image_size=cfg['image_size']); dl=DataLoader(ds,cfg['batch_size'],shuffle=False,num_workers=cfg.get('num_workers',0)); ys=[]; ps=[]
    model.eval()
    for x,y,_ in dl: ps.append(model(x.to(device))[0].argmax(1).cpu().numpy()); ys.append(y.numpy())
    model.train(); return open_world_metrics(np.concatenate(ys),np.concatenate(ps),cfg['seen_classes'])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/cao_odas.json'); ap.add_argument('--mode',choices=['odas','supervised'],default='odas'); args=ap.parse_args()
    cfg=load_config(args.config); seed_all(cfg['seed']); device=torch.device('cuda' if cfg.get('device')=='cuda' and torch.cuda.is_available() else 'cpu'); root=Path(cfg['data_root'])
    li,ui=make_open_world_indices(root,cfg.get('labeled_fraction',.5),tuple(cfg['seen_classes']),cfg['seed'])
    ld=DataLoader(CaoDataset(root,'train',li,cfg['image_size']),cfg['batch_size'],shuffle=True,drop_last=True,num_workers=cfg.get('num_workers',0)); ud=DataLoader(CaoDataset(root,'train',ui,cfg['image_size']),cfg['batch_size']*cfg.get('unlabeled_multiplier',2),shuffle=True,drop_last=True,num_workers=cfg.get('num_workers',0))
    model=ODASNet(cfg['num_classes']).to(device); opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg.get('weight_decay',0)); bank=MemoryBank(cfg.get('memory_size',1024),cfg['num_classes'],128,cfg.get('memory_temperature',.07)); out=Path(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True); history=[]
    json.dump({'labeled_indices':li,'unlabeled_indices':ui,'seen_classes':cfg['seen_classes'],'seed':cfg['seed'],'mode':args.mode},open(out/'split.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    ud_cycle=None
    for epoch in range(cfg['epochs']):
        t0=time.time(); totals={'loss':0.,'supervised':0.,'ucm':0.,'graph':0.,'cacm':0.}; n=0; ud_cycle=iter(ud)
        for xl,yl,_ in tqdm(ld,desc=f'epoch {epoch+1}/{cfg["epochs"]}'):
            try: xu,_,_=next(ud_cycle)
            except StopIteration: ud_cycle=iter(ud); xu,_,_=next(ud_cycle)
            xl,yl,xu=xl.to(device),yl.to(device),xu.to(device); xlw=transform(xl,cfg); xuw=transform(xu,cfg); xus1=transform(xu,cfg,True); xus2=transform(xu,cfg,True)
            logits_l,_,_=model(xlw); logits_u,zu,_=model(xuw); logits_s1,zs1,_=model(xus1); _,zs2,_=model(xus2)
            loss_s=torch.nn.functional.cross_entropy(logits_l,yl)
            if args.mode=='supervised': loss=loss_s; parts=[loss_s,loss_s*0,loss_s*0,loss_s*0]
            else:
                p=torch.softmax(logits_u.detach(),1); q=bank.smooth(zu.detach(),p,cfg.get('pseudo_label_weight',.9)); bank.update(zu,p)
                conf=q.max(1).values; selected=conf>=cfg.get('graph_threshold',.7); loss_u=soft_ce(q[selected],logits_s1[selected]) if selected.any() else loss_s*0
                lu=ucm_loss(logits_u,zu,cfg.get('ucm_similarity_threshold',.5)); lg=graph_loss(q,torch.nn.functional.normalize(zs1,1),cfg.get('graph_threshold',.7),cfg.get('temperature',.07)); lc=cacm_loss(q,zs1,zs2,cfg.get('class_aware_threshold',.9),cfg.get('temperature',.07)); loss=cfg['lambda_s']*loss_s+cfg['lambda_uc']*lu+cfg['lambda_u']*(loss_u+lg)+cfg['lambda_c']*lc; parts=[loss_s,lu,lg,lc]
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); b=len(xl); n+=b
            for k,v in zip(['supervised','ucm','graph','cacm'],parts): totals[k]+=float(v.detach())*b
            totals['loss']+=float(loss.detach())*b
        metrics=evaluate(model,root,cfg,device); row={'epoch':epoch+1,'seconds':time.time()-t0,**{k:v/n for k,v in totals.items()},**metrics}; history.append(row); print(json.dumps(row,ensure_ascii=False)); torch.save({'model':model.state_dict(),'config':cfg,'epoch':epoch+1},out/'last.pt'); json.dump(history,open(out/'history.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print('saved:',out)

if __name__=='__main__': main()
