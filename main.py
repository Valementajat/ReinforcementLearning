
import numpy as np



import gymnasium as gym
# Register the custom environment with Gym
from gymnasium.envs.registration import register

# Q-learning imports
from agents.QAgent import QAgent
from evaluation.combinedEvaluation import evaluate_combined
from training.QTrain import train_q_learning
from evaluation.QEvaluation import evaluate_q_learning

# VFA imports
from agents.VFAAgent import VFAAgent, featurize
from training.VFATrain import train_vfa
from evaluation.VFAEvaluation import evaluate_vfa

from training.CombinedTrain import train_combined
from CombinedMetrics import print_comparison

from mapCreator import create_map 
from helper import save_episode, show_frame

# Initialize the environment
SEED = 42
register(
    id="ToxicSwamp-v0",
    entry_point="env:ToxicSwampEnv",  
)

# Environment parameters for gym.make
ENV_DET = dict(
    map_fn=create_map,          # function from mapCreator to create the map on the fly
    toxic_pct=0.25,
    healthy_pct=0.1,
    slippery=0.0,
    fog_radius=4,
    max_energy=150,
    energy_bins=10,
    
)
""" print("Creating environment with parameters:")
for key, value in ENV_DET.items():
    print(f"  {key}: {value}") """

env = gym.make("ToxicSwamp-v0", **ENV_DET, render_mode="rgb_array")  # Create the  environment 

print(f"Observation space: {env.observation_space}")
print(f"Action space:      {env.action_space}")
print(f"Render modes:      {env.metadata['render_modes']}")

obs, info = env.reset(seed=SEED)
print(f"Initial observation (fog window + energy bin): {obs}")
print(f"Agent position: {env.unwrapped._agent_pos}")
print(f"Energy: {env.unwrapped.energy}")


# Hyperparameters
num_episodes = 100000
max_training_steps = 1000


# Initialize the Q-learning agent

Qagent = QAgent(
    actions=list(range(env.action_space.n)),
    alpha=0.1,
    gamma=0.8,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.998,
    seed=SEED
    
)

# Initialize the VFA agent
VFAagent = VFAAgent(
    actions=list(range(env.action_space.n)),
    featurize=featurize,
    w=np.zeros(18),
    alpha=0.0002,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.998,
    seed=SEED

)

""" # ── Training loop Q learning ─────────────────────────────────────────────────
train_q_learning(env, Qagent, num_episodes=num_episodes, max_training_steps=max_training_steps)

# ── Evaluation  Q learning ─────────────────────────────────────────────────
evaluate_q_learning(env, Qagent)

 """

""" # ── Training loop VFA ──────────────────────────────────────────────────
train_vfa(env, VFAagent, num_episodes=num_episodes, max_training_steps=max_training_steps)

# ── Evaluation  VFA ─────────────────────────────────────────────────
evaluate_vfa(env, VFAagent)
 """

# ── Training loop for both ──────────────────────────────────────────────────

# To save time, this part was made with Claude AI based on the both training files
train_metrics = train_combined(
    env, Qagent, VFAagent,
    num_episodes=num_episodes,
    max_training_steps=max_training_steps,
)

eval_metrics = evaluate_combined(env, Qagent, VFAagent)

print_comparison(train_metrics, eval_metrics)