import gymnasium as gym
import metaworld

env = gym.make("Meta-World/MT1", env_name="reach-v3")

obs, info = env.reset()

print(obs.shape)


env = gym.make("Meta-World/MT1", env_name="reach-v3")

obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print(reward)

    if terminated or truncated:
        obs, info = env.reset()