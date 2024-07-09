from run_model import run_model
from pathlib import Path
from sklearn import metrics
import format_string 
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
with root_dir.joinpath('data', 'best_model_info.txt').open(mode='r') as f:
    content = f.readlines()
    model_path = Path(content[0].rstrip())
    is_direct = (content[1].rstrip() == 'True')
print(f'selected model: {model_path}, is_direct: {is_direct}')

test_data_path = root_dir.joinpath('data', 'test_data.pkl')
with test_data_path.open(mode='rb') as f:
    test_data = pickle.load(f)
test_input = test_data.iloc[:, 1:301]
predict = pd.DataFrame(run_model(model_dir.joinpath(model_path), test_input, is_direct))

set_sentences = set()
score_sentences = []
for i in range(len(test_data.index)):
    s = test_data.iloc[i, 0]
    p = predict.iloc[i, 0]
    if s not in set_sentences:
        set_sentences.add(s)
        score_sentences.append((p, s))
score_sentences.sort()
max_len = max(format_string.calc_length(s) for s in set_sentences)

with root_dir.joinpath('data', 'result.txt').open(mode='w') as file:
    for (p, s) in score_sentences:
        file.write(f"{format_string.align_text(s, max_len)} | {p:10.4f}\n")
