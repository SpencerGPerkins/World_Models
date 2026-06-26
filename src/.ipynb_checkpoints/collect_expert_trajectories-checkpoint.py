import numpy as np
from tqdm import tqdm
import gymnasium as gym
import os
import imageio
import metaworld
from metaworld.policies import SawyerReachV3Policy

base_dir = "../datasets/reach_expert"
traj_dir = os.path.join(base_dir, "trajectories")
video_dir = os.path.join(base_dir, "videos")

os.makedirs(traj_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

env = gym.make(
    "Meta-World/MT1",
    env_name="reach-v3",
    render_mode="rgb_array"
)

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

    img_t = env.render()

    for step in range(MAX_STEPS):

        action = policy.get_action(obs)

        robot_t = obs[:4]

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        img_tp1 = env.render()

        traj["images"].append(img_t)
        traj["robot_states"].append(robot_t)
        traj["actions"].append(action)
        traj["rewards"].append(reward)
        traj["next_images"].append(img_tp1)
        traj["dones"].append(done)

        frames.append(img_t)

        obs = next_obs
        img_t = img_tp1

        if done:
            break

    np.savez_compressed(
        os.path.join(traj_dir, f"ep_{episode:04d}.npz"),
        **{k: np.array(v) for k, v in traj.items()}
    )

    if len(frames) > 0:
        imageio.mimsave(
            os.path.join(video_dir, f"ep_{episode:04d}.mp4"),
            np.stack(frames),
            fps=30
        )

env.close()