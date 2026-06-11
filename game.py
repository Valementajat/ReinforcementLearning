
from mapCreator import create_map 
import gymnasium as gym

# Register the custom environment with Gym
from gymnasium.envs.registration import register

register(
    id="ToxicSwamp-v0",
    entry_point="env:ToxicSwampEnv",  
)


# Environment parameters for tabular q learning



# Environment parameters for gym.make
ENV_DET = dict(
    map_fn=create_map,          # function from mapCreator to create the map on the fly
    toxic_pct=0.25,
    healthy_pct=0.10,
    slippery=0.3,
    fog_radius=2,
    max_energy=50,
    energy_bins=10,
)

"  render_mode=rgb_array"
print("Creating environment with parameters:")
for key, value in ENV_DET.items():
    print(f"  {key}: {value}")

env = gym.make("ToxicSwamp-v0", **ENV_DET)  # Create the first environment 

print(f"Observation space: {env.observation_space}")
print(f"Action space:      {env.action_space}")
print(f"Render modes:      {env.metadata['render_modes']}")

obs, info = env.reset(seed=42)
print(f"Initial observation: {obs}  (row={obs // env.unwrapped.grid_width}, col={obs % env.unwrapped.grid_width})")


