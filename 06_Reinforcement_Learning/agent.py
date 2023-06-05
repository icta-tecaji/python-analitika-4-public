from stable_baselines3 import PPO
from snakeenv import SnekEnv

env = SnekEnv()
policy = "MlpPolicy"
model_name = "snek_V7"
env.render_training = True
train = False

try:
    model = PPO.load(f"./models/{model_name}", env)
except FileNotFoundError as e:
    model = PPO(policy, env)

model.learn(3_000, progress_bar=True) if train else None
model.save(f"models/{model_name}") if train else None

done = False
obs, _ = env.reset()
while not done:
    env.render()

    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if truncated:
        done = True
