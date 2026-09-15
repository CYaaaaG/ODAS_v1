import torch
from torch import nn
from torchvision.models import resnet18
class ODASNet(nn.Module):
    def __init__(self,num_classes=6,embedding_dim=128):
        super().__init__(); net=resnet18(weights=None); net.conv1=nn.Conv2d(1,64,7,2,3,bias=False); self.encoder=nn.Sequential(*list(net.children())[:-1]); d=net.fc.in_features
        self.classifier=nn.Linear(d,num_classes); self.projector=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,embedding_dim))
    def forward(self,x):
        f=self.encoder(x).flatten(1); return self.classifier(f),nn.functional.normalize(self.projector(f),dim=1),f
