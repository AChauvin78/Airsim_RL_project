import gym
import time
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


# Import the custom environment
from  airgym.envs.drone_env import AirSimDroneEnv

# Create vectorized environments
num_envs = 1
env = DummyVecEnv([lambda: Monitor(AirSimDroneEnv(ip_address="127.0.0.1", step_length=0.25)) for _ in range(num_envs)])

# Save the model
model = DQN.load("saved_models/run_5000_time_steps/dqn_airsim_drone")

# To test the trained model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # Print information about the current step
    print(f"Action: {action}, Reward: {reward}, Done: {done}, Obs: {obs}")

    if done:
        obs = env.reset()
