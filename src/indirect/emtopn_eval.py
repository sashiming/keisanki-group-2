import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
class nw (nn.Module):
    def __init__(self):
        super(nw,self).__init__()
        mid = 500
        self.fc1 = nn.Linear(8,mid)
        self.fc2 = nn.Linear(mid,mid)
        self.fc3 = nn.Linear(mid,mid)
        self.fc4 = nn.Linear(mid,1)
        self.sig = nn.Sigmoid()
    def forward(self,x):
        x = self.sig(self.fc1(x))
        x = self.sig(self.fc2(x))
        x = self.sig(self.fc3(x))
        return self.fc4(x)

class emtopnmodel:
    def __init__(self):
        with open("./models/emtopn.pkl","rb") as f:
            self.model = pickle.load(f)
    def eval(self,a,b,c,d,e,f,g,h):
        return self.model(torch.tensor([float(a),float(b),float(c),float(d),float(e),float(f),float(g),float(h)])).item()

