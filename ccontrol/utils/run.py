import os

import numpy as np
from collections import deque

from ccontrol.utils import save_scores, save_AC_models, save_configuration

class Runner:

    def __init__(self) -> None:

        file_location = os.path.dirname(__file__)
        self.path_score = os.path.join(file_location, r'./../../output/score')
        self.path_model = os.path.join(file_location, r'./../../output/model')
        self.path_config = os.path.join(file_location, r'./../../output/configuration')

    def run(self, agent, env, brain_name, nb_episodes, key,
            average_on=10, target_score=None, target_over=100,
            save_score=True, save_config=True, save_weights=False, save_interaction=False):
            
            scores = deque()
            scores_target = deque(maxlen=target_over)
            scores_window = deque(maxlen=average_on)
            is_solved = ''
            
            for episode in range(1, nb_episodes+1):
                
                env_info = env.reset(train_mode=True)[brain_name] 
                states = env_info.vector_observations          
                score = 0  
                
                while True:
                    
                    actions = agent.act(states, noise=True)
                    
                    env_info = env.step(actions)[brain_name]
                    next_states = env_info.vector_observations
                    rewards = env_info.rewards
                    dones = env_info.local_done
                    
                    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                        agent.step(state, action, reward, next_state, done)
                    
                    states = next_states
                    score += np.mean(rewards)
                    
                    if np.any(dones):
                        scores.append(score)
                        scores_target.append(score)
                        scores_window.append(score)
                        score_averaged = np.mean(list(scores_window))
                        print(f"\rEpisode {episode} Score: {score_averaged}{is_solved}",
                            end='\r')
                        if target_score:
                            if (len(is_solved) == 0) & (np.mean(list(scores_target)) > target_score):
                                is_solved = f' -> Solved in {episode} episodes'
                        break
                        
            print(f"\nLast score: {round(score_averaged,5)} {is_solved}")  
                    
            if save_score:
                save_scores(scores, key, self.path_score)
    
            if save_config:
                save_configuration(agent, key, self.path_config)
    
            if save_weights:
                save_AC_models(agent, key, self.path_model)
    
            if save_interaction:
                raise Exception('not implemented yet')

    def run_single_agent(self, agent, env, brain_name, nb_episodes, key,
                        average_on=10, target_score=None, target_over=100,
                        save_score=True, save_config=True, save_weights=False, save_interaction=False):
        
        scores = deque()
        scores_target = deque(maxlen=target_over)
        scores_window = deque(maxlen=average_on)
        is_solved = ''
        
        for episode in range(1, nb_episodes+1):
            
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
                    scores_target.append(score)
                    scores_window.append(score)
                    score_averaged = np.mean(list(scores_window))
                    print(f"\rEpisode {episode} Score: {score_averaged}{is_solved}",
                        end='\r')
                    if target_score:
                        if (len(is_solved) == 0) & (np.mean(list(scores_target)) > target_score):
                            is_solved = f' -> Solved in {episode} episodes'
                    break

        print(f"\rLast score: {round(score_averaged,5)} {is_solved}")  
                
        if save_score:
            save_scores(scores, key, self.path_score)

        if save_config:
            save_configuration(agent, key, self.path_config)

        if save_weights:
            save_AC_models(agent, key, self.path_model)

        if save_interaction:
            raise Exception('not implemented yet')