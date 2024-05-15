import gymnasium as gym
from gymnasium import spaces
import pandas
import numpy as np

class Unicycle2DWithObstacle(gym.Env):
    def __init__(self):
        super(Unicycle2DWithObstacle, self).__init__()
        self.observation_space = spaces.Box(low=np.array([-10.0,-10.0,-np.inf]), high=np.array([10.0,10.0,np.inf]), dtype=np.float32)  # pos_x, pos_y, theta
        self.action_space = spaces.Box(low=np.array([-1,-np.pi/4]), high=np.array([1,np.pi/4]), dtype=np.float32)  # v, angle_v

        self.current_state = np.array([0.0, 0.0, 0.0])
        self.goal = np.array([5.0, 4.0])
        self.obstacles = np.array([[1.0, 1.0, 0.4],
                                   [2.5, 2.0, 0.4],
                                   [4.0, 2.0, 0.4]])
        self.dt = 0.2
        self.info = {}
        self.reward_weights = {
            'goal_reached': 10.0,
            'obstacle_collision': -10.0
        }

    def check_collision(self):
        for obstacle in self.obstacles:
            if np.linalg.norm(self.current_state[:2] - obstacle[:2]) < obstacle[2]:
                return True  
        return False 

    def check_goal(self):
        if np.linalg.norm(self.current_state[:2] - self.goal) < 0.1:
            return True
        return False
    
    def reset(self):
        self.current_state = np.array([0.0, 0.0, 0.0])
        return self.current_state, {}

    def step(self, action):
        velocity, angular_velocity = action
        self.current_state += np.array([0.9 * velocity * np.cos(self.current_state[2]),
                                        0.9 * velocity * np.sin(self.current_state[2]),
                                        angular_velocity]) * self.dt  
        if self.check_collision():
            reward = self.reward_weights['obstacle_collision']
            done = True
            self.info = {'obstacle_collision'}
        if self.check_goal():
            reward = self.reward_weights['goal_reached'] 
            done = True
            self.info = {'goal_reached'}
        else:
            reward = -np.linalg.norm(self.current_state[:2] - self.goal)
            done = False
            self.info = {'driving...'}

        return self.current_state, reward, done, False, self.info