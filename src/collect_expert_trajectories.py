import numpy as np
from tqdm import tqdm
import gymnasium as gym
import os
import imageio
import metaworld
from metaworld.policies import SawyerReachV3Policy

# Set up directories
base_dir = "../datasets/reach_expert"
traj_dir = os.path.join(base_dir, "trajectories")
video_dir = os.path.join(base_dir, "videos")

os.makedirs(traj_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)


# Environment
env = gym.make(
    "Meta-World/MT1",
    env_name="reach-v3",
    render_mode="rgb_array"
)

# Expert policy
policy = SawyerReachV3Policy()

NUM_EPISODES = 200
MAX_STEPS = 200

for episode in tqdm(range(NUM_EPISODES), desc="Expert rollouts"):

    obs, info = env.reset()

    traj = {
        "images": [],
        "robot_states": [],
        "actions": [],
        "rewards": [],
        "next_images": [],
        "dones": []
    }

    frames = []

    frame = env.render()

    # Iterate
    for step in range(MAX_STEPS):
        action = policy.get_action(obs)
        r_t = obs[:4]

        traj["images"].append(frame)
        traj["robot_states"].append(r_t)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_frame = env.render()

        traj["actions"].append(action)
        traj["rewards"].append(reward)
        traj["next_images"].append(next_frame)
        traj["dones"].append(done)

        frames.append(frame)

        obs = next_obs
        frame = next_frame

        if done:
            break
    
    # Save episode
    traj_path = os.path.join(traj_dir, f"ep_{episode:04d}.npz")
    np.savez_compressed(traj_path, **{k: np.array(v) for k, v in traj.items()})

    video_path = os.path.join(video_dir, f"ep_{episode:04d}.mp4")
    if len(frames) > 0:
        imageio.mimsave(video_path, np.stack(frames), fps=30)

env.close()