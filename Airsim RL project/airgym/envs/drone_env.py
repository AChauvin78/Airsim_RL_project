import setup_path
import airsim
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces

class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address="127.0.0.1", step_length=0.25):
        super(AirSimDroneEnv, self).__init__()
        self.step_length = step_length
        self.max_episode_steps = 50
        self.current_step = 0


        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)  # 7 discrete actions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self._setup_flight()

    def __del__(self):
        if hasattr(self, 'drone') and self.drone is not None:
            self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.moveToPositionAsync(0, 0, -5, 5).join()
        self.drone.moveByVelocityAsync(2.5, 0, 0, 0.5).join()

    def reset(self, seed=None, options=None):
        # Reset the time step counter
        self.current_step = 0

        # Seed the environment's random number generator
        super().reset(seed=seed)
        self._setup_flight()
        return self._get_obs(), {}  # Return observation and additional info

    def _get_obs(self):
        self.drone_state = self.drone.getMultirotorState()
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        # Print the drone's current coordinates
        #print(f"Drone Coordinates: X: {self.state['position'].x_val}, "
        #  f"Y: {self.state['position'].y_val}, Z: {self.state['position'].z_val}")

        sensor_data = self.drone.getDistanceSensorData()
        distance_obs = sensor_data.distance
        if distance_obs <= 10:
            self.near_obstacle = True
        else:
            self.near_obstacle = False

        max_pos_range = 30
        max_velocity = 20
        max_sensor_range = 50
        max_altitude = 1000

        obs = np.array([
            self.state["position"].x_val / max_pos_range,
            self.state["position"].y_val / max_pos_range,
            self.state["position"].z_val / max_pos_range,
            self.state["velocity"].x_val / max_velocity,
            self.state["velocity"].y_val / max_velocity,
            self.state["velocity"].z_val / max_velocity,
            distance_obs / max_sensor_range,
            self.drone.getBarometerData().altitude/max_altitude
        ], dtype=np.float32)

        return obs

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        # Increment the time step counter
        self.current_step += 1

        # Check if the time limit has been reached
        if self.current_step >= self.max_episode_steps:
            done = True  # End the episode

        return obs, reward, done, False, {}

    def calculate_yaw(self, velocity): # Calculate Target Yaw from Velocity Vector so that the distance sensor point towards the velocity vector
        return math.atan2(velocity.y_val, velocity.x_val)

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        
        # Calculate the desired yaw to align with the velocity vector
        yaw = self.calculate_yaw(quad_vel)
        
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            duration=5,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(yaw))
        ).join()

    def _compute_reward(self):
        thresh_dist = 20
        safe_distance = 10
        beta = 1

        pts = [np.array([0, 0, -5]),np.array([250, 0, -5])]

        sensor_data = self.drone.getDistanceSensorData()
        distance = sensor_data.distance

        quad_pt = np.array(list((self.state["position"].x_val, self.state["position"].y_val,self.state["position"].z_val,)))

        if self.state["collision"]:
            reward = -1000
        else:
            dist = 10000000
            for i in range(0, len(pts) - 1):
                dist = min(dist, np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1]))) / np.linalg.norm(pts[i] - pts[i + 1]))

            if dist > thresh_dist:
                reward = -10
            else:
                reward_dist = math.exp(-beta * dist) - 0.5
                reward_speed = (np.linalg.norm([self.state["velocity"].x_val, self.state["velocity"].y_val, self.state["velocity"].z_val,])- 0.5)
                reward = reward_dist + reward_speed
        
        if distance < safe_distance:
            reward -= (safe_distance - distance)  # Penalize being close to obstacles

        return reward, reward <= -100

    def interpret_action(self, action):   
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)
        return quad_offset
