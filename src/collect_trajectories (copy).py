
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import metaworld

env = gym.make("Meta-World/MT1", env_name="reach-v3")

NUM_EPISODES = 200
MAX_STEPS = 200

dataset = []

for episode in tqdm(range(NUM_EPISODES), desc="Collecting trajectories"):

    obs, info = env.reset()

    episode_data = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "dones": []
    }

    for step in range(MAX_STEPS):

        action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        episode_data["observations"].append(obs)
        episode_data["actions"].append(action)
        episode_data["rewards"].append(reward)
        episode_data["next_observations"].append(next_obs)
        episode_data["dones"].append(done)

        obs = next_obs

        if done:
            break

    for key in episode_data:
        episode_data[key] = np.array(episode_data[key])

    dataset.append(episode_data)

env.close()

print(f"Collected {len(dataset)} trajectories")
np.savez_compressed("../datasets/reach_random_dataset.npz", dataset=dataset)