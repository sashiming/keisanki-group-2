from run_model import run_model
from pathlib import Path
from sklearn import metrics
import pickle
import pandas as pd
import torch.nn as nn

class nw(nn.Module):
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

root_dir = Path(__file__).parents[1]
model_dir = root_dir.joinpath('models')
model_paths = []
model_paths.append(('direct_model.pkl', True))
model_paths.append(('emo_neuralnet.pkl', False))
model_paths.append(('emo_ridge_best.pkl', False))
model_paths.append(('emo_xgboost_best.pkl', False))

validation_data_path = root_dir.joinpath('data', 'validation_data.pkl')
with validation_data_path.open(mode='rb') as f:
    validation_data = pickle.load(f)
validation_input = validation_data.iloc[:, 1:301]
validation_correct = validation_data.iloc[:, 309:310]

best_score = float('inf')
best_model_info = None
for (path, is_direct) in model_paths:
    predict = pd.DataFrame(run_model(model_dir.joinpath(path), validation_input, is_direct))
    score = metrics.mean_squared_error(validation_correct, predict)
    print(f'model name: {path}, score: {score}')
    for i in range(10):
        print(f'sentence: {validation_data.iloc[i, 0]}, prediction: {predict.iloc[i, 0]}')
    if score < best_score:
        best_score = score
        best_model_info = (path, is_direct)

print(f'best_model: {best_model_info[0]}')
best_model_info_path = root_dir.joinpath('data', 'best_model_info.txt')
with best_model_info_path.open(mode='w') as f:
    f.write(f'{best_model_info[0]}\n')
    f.write(f'{best_model_info[1]}\n')