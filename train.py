import gymnasium as gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
import itertools

import agent_class as agent

# This is for Contextual Lunar Lander environment
# env = gym.make('LunarLander-v3')

# Define ranges for random environment parameters
gravity_range = (-11.99, -10.66)  # Gravity range
wind_power_range = (0.01, 19.99)  # Wind power range
turbulence_power_range = (0.01, 1.99)  # Turbulence power range

def create_random_env():
    # Randomly sample environment parameters
    gravity = np.random.uniform(*gravity_range)
    wind_power = np.random.uniform(*wind_power_range)
    turbulence_power = np.random.uniform(*turbulence_power_range)

    env = gym.make(
        "LunarLander-v3",
        gravity=gravity,
        enable_wind=True,  # Enable wind
        wind_power=wind_power,
        turbulence_power=turbulence_power,
    )
    return env

# Create a random environment
env = create_random_env()

# We need to know the dimensionality of the state space, as well as how many
# actions are possible
N_actions = env.action_space.n
observation, info = env.reset()
N_state = len(observation)+3

print('dimension of state space =',N_state)
print('number of actions =',N_actions)

# We create an instance of the agent class. 
# At initialization, we need to provide 
# - the dimensionality of the state space, as well as 
# - the number of possible actions

parameters = {'N_state':N_state, 'N_actions':N_actions}

my_agent = agent.dqn(parameters=parameters)
track_embedder = agent.TrackEmbeddingWrapper(f"./model_file/FactualEncoder.pt")

training_results = my_agent.train(environment=env,
                                verbose=True, model_filename=f"./trained_agents/ContextualNet.pth")