import numpy as np
from tqdm import tqdm
import gymnasium as gym
import os
import imageio
import metaworld

base_dir = "../datasets/reach_random"
traj_dir = os.path.join(base_dir, "trajectories")
video_dir = os.path.join(base_dir, "videos")

os.makedirs(traj_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

env = gym.make(
    "Meta-World/MT1",
    env_name="reach-v3",
    render_mode="rgb_array"
)

NUM_EPISODES = 200
MAX_STEPS = 200

for episode in tqdm(range(NUM_EPISODES), desc="Random rollouts"):

    obs, info = env.reset()

    traj = {
        "images": [],
        "robot_states": [],
        "actions": [],
        "rewards": [],
        "next_images": [],
        "next_robot_states": [],
        "dones": []
    }

    frames = []

    frame = env.render()

    for step in range(MAX_STEPS):

        action = env.action_space.sample()

        robot_state = obs[:4]

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_robot_state = next_obs[:4]

        next_frame = env.render()

        # Store transition
        traj["images"].append(frame)
        traj["robot_states"].append(robot_state)
        traj["actions"].append(action)
        traj["rewards"].append(reward)
        traj["next_images"].append(next_frame)
        traj["next_robot_states"].append(next_robot_state)
        traj["dones"].append(done)

        frames.append(frame)

        obs = next_obs
        frame = next_frame

        if done:
            break

    # SAVE
    traj_path = os.path.join(traj_dir, f"ep_{episode:04d}.npz")
    np.savez_compressed(
        traj_path,
        images=np.array(traj["images"]),
        robot_states=np.array(traj["robot_states"]),
        actions=np.array(traj["actions"]),
        rewards=np.array(traj["rewards"]),
        next_images=np.array(traj["next_images"]),
        next_robot_states=np.array(traj["next_robot_states"]),
        dones=np.array(traj["dones"]),
    )
    video_path = os.path.join(video_dir, f"ep_{episode:04d}.mp4")

    if len(frames) > 0:
        imageio.mimsave(video_path, np.stack(frames), fps=30)

env.close()