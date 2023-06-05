import gymnasium as gym

from stable_baselines3 import DQN, A2C, PPO

model = DQN("MlpPolicy", "CartPole-v1").learn(10_000, progress_bar=True)
# model = A2C("MlpPolicy", "CartPole-v1").learn(10_000, progress_bar=True)
# model = PPO("MlpPolicy", "CartPole-v1").learn(10_000, progress_bar=True)

env = gym.make("CartPole-v1", render_mode="human")
env.metadata["render_fps"] = 30

episodes = 5
total_rewards = []

for ep in range(episodes):
    # Performance tracking
    total_reward = 0

    done = False
    obs, _ = env.reset()
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if truncated:
            done = True
        # Performance tracking
        total_reward += reward

    total_rewards.append(total_reward)
    print("Episode: ", ep)
    print("Total reward: ", total_reward)

print("Average reward: ", sum(total_rewards) / len(total_rewards))
