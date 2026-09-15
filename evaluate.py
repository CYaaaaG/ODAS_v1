import argparse,json,torch
from torch.utils.data import DataLoader
from odas.data import load_config,CaoDataset
from odas.model import ODASNet
from odas.evaluation import open_world_metrics
import numpy as np
ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--config',default='configs/cao_odas.json'); a=ap.parse_args(); c=load_config(a.config); d=torch.device('cuda' if c.get('device')=='cuda' and torch.cuda.is_available() else 'cpu'); m=ODASNet(c['num_classes']).to(d); m.load_state_dict(torch.load(a.checkpoint,map_location=d)['model']); m.eval(); ds=CaoDataset(c['data_root'],'test',image_size=c['image_size']); dl=DataLoader(ds,c['batch_size']); y=[];p=[]
with torch.no_grad():
    for x,t,_ in dl: p.append(m(x.to(d))[0].argmax(1).cpu().numpy()); y.append(t.numpy())
print(json.dumps(open_world_metrics(np.concatenate(y),np.concatenate(p),c['seen_classes']),ensure_ascii=False,indent=2))
