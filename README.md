# Continuous control

This package offer implementation of policy based model-free methods to solve Reinforcement Learnig environment whose action space is continuous. Those environments can be challenging to solve because the agent has to choose which action to do as well at its intensity which for instance DQN is incapable of. The environnement used to do the benchmark is ML-Unity Reacher.

## The Environnement

This Unity environment consist of a **double-jointed arm** that can move in a three dimensional space and a spherical target location. The goal of the agents is to **move its hand to the goal location**, and keep it there as it get a +0.1 **reward** for each step it’s hand is in goal location.

The simulation can contain up to 20 agents that share the same behavior parameters. At each time step, each agent must return **action** vector $\in [0,1]^4$ corresponding   to torque that will be applied to the two joints.

They observe 33-dimensional **state space** corresponding to position, rotation, velocity, and angular velocities of the two arms.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

## Getting Started

https://github.com/Unity-Technologies/ml-agents

You will need to install PyTorch, and to download Unity Reacher environment. You can read more about Unity ML-Agents [here](https://github.com/Unity-Technologies/ml-agents).

1. Install the dependencies using the requirements file.
    - cd to the directory where requirements.txt is located.
    - activate your virtualenv.
    - run: `pip install -r requirements.txt` in your shell.


2. Install this **ccontrol package** by running `pip install .` in your shell where the `setup.py` of the package is.

3. Download the environment from one of the links below.  You need only select the environment that matches your operating system and the desired number of agents:

One agent environment:
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Twenty agents environment:
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

4. Place the file in your working repository and unzip (or decompress) the file.


## Instructions

Follow the instructions in `Continuous_control.ipynb` to train an agent an watch it interact with the environment.  
You will need to intanciate the `agent` with its `configuration` and make it learn by interacting with the environment throught the use a `runner`.

**1. instantiate an agent :**

```python
cfg = DDPG_configuration(
    seed=2,
    update_every=5,
    gamma=.97,
    tau=5e-4,
    actor_config={
        'architecture': 'BN'
    },
    critic_config={
        'architecture': 'BN',
        'learning_rate': 2e-4
    }
)
ctx = Context(brain, {'action_min': -1,  'action_max': 1})
agent = DDPG_agent(ctx, cfg)
```

**2. Make it learn by using a runner**

```python
key = 'my_key' # used to store the score / configuration / networks
nb_episodes = 500
runner.run(agent, env, brain_name, nb_episodes, key, target_score=30,
           save_score=True, save_config=True, save_weights=True)
```

**3. Watch it interact with the environment**

```python
env_info = env.reset(train_mode=True)[brain_name] 
states = env_info.vector_observations          
score = 0  

while True:
    
    actions = agent.act(states, noise=False)
    
    env_info = env.step(actions)[brain_name]
    next_states = env_info.vector_observations
    rewards = env_info.rewards
    dones = env_info.local_done
    
    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        agent.step(state, action, reward, next_state, done)
    
    states = next_states
    score += np.mean(rewards)
    if np.any(dones):
        break
```

**It is also possible to load a pre-trained agent using the following method:**

```python
# the agent need to be instantiated with the right configuration first.
agent = load_agent(key_20A, agent)

```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)