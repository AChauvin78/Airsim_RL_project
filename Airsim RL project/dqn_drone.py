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

# Set up TensorBoard logging
tensorboard_log_dir = "./tb_logs/"
logger = configure(tensorboard_log_dir, ["stdout", "log", "csv"])

# Initialize the DQN model with your specified parameters
model = DQN(
    "MlpPolicy",  # Change to "MlpPolicy" if your observations are not image-based
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=128,
    train_freq=4,
    target_update_interval=500,
    learning_starts=1000,
    buffer_size=100000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cuda",  # Use "cpu" if you don't have a compatible GPU
    tensorboard_log=tensorboard_log_dir,
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=1000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train the model
model.learn(total_timesteps=5000,
    tb_log_name="dqn_airsim_drone_run_" + str(time.time()),
    **kwargs
)

# Save the model
model.save("dqn_airsim_drone")

# To test the trained model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # Print information about the current step
    print(f"Action: {action}, Reward: {reward}, Done: {done}, Obs: {obs}")

    if done:
        obs = env.reset()
