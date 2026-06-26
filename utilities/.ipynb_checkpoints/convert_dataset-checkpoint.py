import numpy as np
import glob
import os
from tqdm import tqdm

# Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))

src_dir = os.path.join(BASE_DIR, "reach_random", "trajectories")
out_file = os.path.join(BASE_DIR, "dino_style_reach.npz")

files = sorted(glob.glob(os.path.join(src_dir, "ep_*.npz")))

if len(files) == 0:
    raise ValueError(f"No episode files found in {src_dir}")

obs, actions, next_obs = [], [], []

# Build Transition
for f in tqdm(files, desc="Processing episodes"):
    data = np.load(f, allow_pickle=True)

    images = data["images"]
    a = data["actions"]

    # safety check (prevents silent crashes)
    if len(images) < 2:
        continue

    for t in range(len(images) - 1):
        obs.append(images[t])
        actions.append(a[t])
        next_obs.append(images[t + 1])

# To arrays
obs = np.array(obs)
actions = np.array(actions)
next_obs = np.array(next_obs)

# Save Dataset
np.savez_compressed(
    out_file,
    obs=obs,
    actions=actions,
    next_obs=next_obs
)

# Sanity check
print("Saved dataset to:", out_file)
print("obs shape:", obs.shape)
print("actions shape:", actions.shape)
print("next_obs shape:", next_obs.shape)