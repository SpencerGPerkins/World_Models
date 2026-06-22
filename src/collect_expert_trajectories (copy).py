import gymnasium as gym
import numpy as np
from tqdm import tqdm

from metaworld.policies import SawyerReachV3Policy

env = gym.make("Meta-World/MT1", env_name="reach-v3")

policy = SawyerReachV3Policy()
NUM_EPISODES = 200
MAX_STEPS = 200
dataset = []

for episode in tqdm(range(NUM_EPISODES), desc="Collecting trajectories"):

    obs, info = env.reset()

    traj = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "dones": []
    }

    for step in range(200):

        action = policy.get_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        traj["observations"].append(obs)
        traj["actions"].append(action)
        traj["rewards"].append(reward)
        traj["next_observations"].append(next_obs)
        traj["dones"].append(done)

        obs = next_obs

        if done:
            break

    for key in traj:
        traj[key] = np.array(traj[key])

    dataset.append(traj)

np.savez_compressed("../datasets/reach_expert_dataset.npz", dataset=dataset)
