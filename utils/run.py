import os

import numpy as np
import pandas as pd
import torch

from collections import deque
from datetime import datetime
import json


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

class Runner:

    def __init__(self) -> None:

        file_location = os.path.dirname(__file__)
        self.path_score = os.path.join(file_location, r'./../../output/score')
        self.path_model = os.path.join(file_location, r'./../../output/model')
        self.path_config = os.path.join(file_location, r'./../../output/configuration')

    def run(self, agent, env, brain_name, nb_episodes, key,
            average_on=10, save_score=True, save_config=True, save_weights=False, save_interaction=False):
        
        scores = deque()
        scores_window = deque(maxlen=average_on)
        
        for episode in range(nb_episodes):
            
            env_info = env.reset(train_mode=True)[brain_name] 
            state = env_info.vector_observations[0]           
            score = 0  
            
            while True:
                
                action = agent.act(state, noise=True)
                
                env_info = env.step(action)[brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                
                agent.step(state, action, reward, next_state, done)
                
                state = next_state
                score += reward
                if done:
                    scores.append(score)
                    scores_window.append(score)
                    score_averaged = np.mean(list(scores_window))
                    print(f"\rEpisode {episode} Score: {score_averaged}", end='\r')
                    break
        print(f"\rLast score: {round(score_averaged,5)}")          
        if save_score:
            save_scores(scores, key, self.path_score)
        if save_config:
            save_configuration(agent, key, self.path_config)
        if save_weights:
            save_AC_models(agent, key, self.path_model)
        if save_interaction:
            raise Exception('not implemented yet')