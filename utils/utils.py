import os
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch import nn

import re
import matplotlib.pyplot as plt
from datetime import datetime
import json


def to_np(tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()

class OptimizerCreator():

    def __init__(self):
        self.builders = {
            'Adam': lambda parameters, kwargs : optim.Adam(parameters, **kwargs)
        }

    def create(self, optimizer, parameters, kwargs):
        return self.builders[optimizer](parameters, kwargs)

class CriterionCreator():

    def __init__(self):
        self.builders = {
            'MSE': nn.MSELoss
        }

    def create(self, criterion):
        return self.builders[criterion]()

def load_scores(path=r'./output/score',
                day=None, month=None, year=None,
                key=None, display=False):
    
    files = [
        f for f in os.listdir(path) 
            if re.match(r'\d\d_\d\d_\d\d_.*', f)]
    
    if year: 
        files = [f for f in files if f[0:2]==year]
    if month: 
        files = [f for f in files if f[3:5]==month]
    if day: 
        files = [f for f in files if f[6:8]==day]
    if key: 
        files = [f for f in files if f[15:]==key]
    
    if display:
        print(files)
        
    res = {}
    for f in files:
        res[f[15:]] = pd.read_csv(os.path.join(path, f))
    return res

def plot_scores(dic_scores, window_size=20, target_score=None):
    
    fig, axe = plt.subplots(1,1,figsize=(14,7))

    for key, result in dic_scores.items():
        score = np.array(result.score)
        score_averaged = []
        for i in range(len(score)):
            score_averaged.append(
                np.mean(
                    score[max(0, i-window_size//2): min(len(score)-1, i+window_size//2)]))
        
        axe.plot(score_averaged, label=key)
        if target_score:
            axe.hlines(13, 0, len(score_averaged), 'k', linestyle=':')
        axe.set_ylabel('Score')
        axe.set_xlabel('Episode #')

    fig.legend(bbox_to_anchor=(1, .85), loc='upper left')

def create_time_suffix():
    now = datetime.now()
    return f'{str(now.year)[-2:]}_{now.month:02}_{now.day:02}_{now.hour:02}h{now.minute:02}'

def save_scores(scores, key, path):

    df_score = pd.DataFrame(np.vstack((range(1,len(scores)+1),scores)).T, columns=['episode','score'])
    df_score.episode = df_score.episode.astype(int)

    print(' ... saving score ...')
    df_score.to_csv(os.path.join(path, f'{create_time_suffix()}_{key}'), index=False)
    
def save_AC_models(agent, key, path):

    file_name_base = os.path.join(path, f'{create_time_suffix()}_{key}')
    
    if hasattr(agent, "actor_network"):
        print('... saving actor ...')
        torch.save(agent.actor_network.state_dict(), file_name_base + '_actor.pth')
        
    if hasattr(agent, "critic_network"):
        print('... saving critic ...')
        torch.save(agent.critic_network.state_dict(), file_name_base + '_critic.pth')

def save_configuration(agent, key, path):
    
    config_file_name = os.path.join(path, f'{create_time_suffix()}_{key}.json')
    
    with open(config_file_name, 'w') as config_file:
        json.dump(agent.config.dict, config_file)
