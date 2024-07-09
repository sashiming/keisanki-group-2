import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas
import pickle
import random
class nw (nn.Module):
    def __init__(self):
        super(nw,self).__init__()
        mid = 500
        #中間層のニューロン数は500にしても250にしても125にしても精度はあまり変わらない(500が僅差で一位) 1000だと過学習が起きちゃう
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
    
#load data
with open('./data/train_data.pkl',"rb") as f:
    traindata:pandas.DataFrame = pickle.load(f)

with open('./data/validation_data.pkl',"rb") as g:
    valdata:pandas.DataFrame = pickle.load(g)
x = []
y = []
xt = []
yt = []
for i in range(traindata.shape[0]):
    x.append(np.array([float(traindata[301][i]),
                         float(traindata[302][i]),
                         float(traindata[303][i]),
                         float(traindata[304][i]),
                         float(traindata[305][i]),
                         float(traindata[306][i]),
                         float(traindata[307][i]),
                         float(traindata[308][i])]))
    y.append(np.array([float(traindata[309][i])]))
for i in range(valdata.shape[0]):
    xt.append(np.array([float(valdata[301][i]),
                         float(valdata[302][i]),
                         float(valdata[303][i]),
                         float(valdata[304][i]),
                         float(valdata[305][i]),
                         float(valdata[306][i]),
                         float(valdata[307][i]),
                         float(valdata[308][i])]))
    yt.append(np.array([float(valdata[309][i])]))
#再現性確保のためシード固定
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
model = nw()
gosa = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = 0.0001)
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for iter in range(traindata.shape[0]//10):
        i = random.randrange(traindata.shape[0])
        optimizer.zero_grad()
        input = torch.from_numpy(x[i].astype(np.float32)).clone()
        outputs = model(input)
        loss = gosa(outputs, torch.from_numpy(np.array([y[i]]).astype(np.float32)))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i in range(valdata.shape[0]):
            outputs = model(torch.from_numpy(xt[i].astype(np.float32)).clone())
            loss = gosa(outputs, torch.from_numpy(np.array([yt[i]]).astype(np.float32)))
            val_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss / (traindata.shape[0]//10)}, Validation Loss: {val_loss / valdata.shape[0]}')
with open("./models/emtopn.pkl","wb") as f:
    pickle.dump(model,f)
    