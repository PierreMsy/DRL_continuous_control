

class Context:

    def __init__(self, env) -> None:
        self.state_size = env.vector_observation_space_size
        self.action_size = env.vector_action_space_size

