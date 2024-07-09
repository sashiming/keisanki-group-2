from pathlib import Path
import pickle
import torch
import torch.nn as nn
import pandas as pd

def run_model(model_path, input, is_direct):
    with model_path.open(mode='rb') as f:
        model = pickle.load(f)
    predict = model.predict(input)
    if not is_direct:
        emtopn_path = Path(__file__).parents[1].joinpath('models', 'emtopn.pkl')
        with emtopn_path.open(mode='rb') as f:
            emtopn = pickle.load(f)
        emtopn.eval()
        predict = emtopn(torch.tensor(predict.astype('f4')))
        predict = pd.DataFrame(predict.detach().numpy())
    return predict