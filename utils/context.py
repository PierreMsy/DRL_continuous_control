from unityagents import UnityEnvironment
import gym

class Context:

    def __init__(self, env) -> None:

        if isinstance(env, UnityEnvironment):

            self.state_size = env.vector_observation_space_size
            self.action_size = env.vector_action_space_size
            self.action_min = None
            self.action_max = None
            raise Exception('Implement it plz')

        elif isinstance(env, gym.wrappers.time_limit.TimeLimit):

            self.state_size = env.observation_space.shape
            if len(self.state_size) == 1:
                self.state_size = self.state_size[0]
            self.action_size = env.action_space.shape
            if len(self.action_size) == 1:
                self.action_size = self.action_size[0]
            self.action_min = env.action_space.low
            self.action_max = env.action_space.high

        else:
            raise Exception('Environment provided must be either open gym'+
            'or Unity environnement')