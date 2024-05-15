import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Vehicle parameters
MASS_NORM = 1000.0  # Nominal vehicle mass (kg)
ENGINE_POWER = 100.0  # Maximum power output (kW)
FRONTAL_AREA = 2.0  # Frontal area of the vehicle (m^2)
WHEELBASE = 2  # Wheelbase (m)
CAR_WIDTH = 2  # Vehicle width (m)
CAR_LENGTH = 3  # Vehicle length (m)
MASS_UNIT = 70.0  # Mass per object (kg)

# Environment parameters
DRAG_COEFFICIENT = 0.28  # Drag coefficient
AIR_DENSITY = 1.293  # Air density (g/L = kg/m^3)
ROAD_FRICTION_COEFFICIENT_NORM = 0.8  # Normal road friction coefficient
ROAD_FRICTION_COEFFICIENT_ICY = 0.2  # Ice road friction coefficient
ROLLING_RESISTANCE_COEFFICIENT = 0.03  # Rolling resistance coefficient


class Car2DWithObstacle(gym.Env):
    def __init__(self):
        super(Car2DWithObstacle, self).__init__()

        self.observation_space = spaces.Box(low=np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-1,-1]), high=np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,1,1]), dtype=np.float32) 
        self.action_space = spaces.Box(low=np.array([-7,-np.deg2rad(45)]), high=np.array([7,np.deg2rad(45)]), dtype=np.float32)  
        self.front_load_unit_num = 0
        self.rear_load_unit_num = 0
        self.mass = MASS_NORM + MASS_UNIT * (self.front_load_unit_num + self.rear_load_unit_num)
        self.yaw_inertia = self.calculate_yaw_inertia()

        # Initialization
        self.current_state = np.array([0., 0., 0., 0., 0., 0., 1., 0.]) # px, py, v, vx, vy, theta, cos, sin
        self.goal_state = np.array([5.0, 5.0, 2.0])
        self.obstacles = np.array([[1.0, 1.0, 0.4, 0.9, 0.0],  # x, y, radius, v, theta
                                   [2.5, 3.0, 0.4, 1.25, -3*np.pi/4],
                                   [4.0, 2.0, 0.4, 1.2, np.pi/2],
                                   [0.0, 3.0, 0.4, 1.15, 0.0],
                                   [-1., 5.0, 0.4, 1.1, -np.pi/4],
                                   [2.0, 6.0, 0.4, 1.0, np.pi]])  # moving obstacels
        # self.obstacles = np.array([[1.0, 1.0, 0.4, 0.0, 0.0],
        #                            [2.5, 2.0, 0.4, 0.0, 0.0],
        #                            [4.0, 2.0, 0.4, 0.0, 0.0]])  # fixed obstacles
        self.state_scaler = np.array([0.1, 0.1, 0.5, 0.5, 1.0, 1.0])
        self.reward_computing = np.array([1., 1., 0., 0., 0., 0.])
        self.dt = 0.02  # Simulation time step
        self.info = {}
        self.angle_diff = np.nan  # heading_angle ~ obstacle
        self.vector_to_obstacle = np.nan  
        self.reward_weights = {
            'goal_reached': 0.,
            'collision': 0.
        }
        self.reset()

    def reset(self):
        while True:
            px = random.uniform(-2, 1)
            py = random.uniform(-2, 1)
            theta = random.uniform(-np.pi, np.pi)
            if not self.check_collision(np.array([px,py])):
                break

        self.current_state = np.array([px, py, 0, 0, 0, theta, np.cos(theta), np.sin(theta)]) 
        self.obstacles = np.array([[1.0, 1.0, 0.4, 0.9, 0.0],
                                   [2.5, 3.0, 0.4, 1.25, -3*np.pi/4],
                                   [4.0, 2.0, 0.4, 1.2, np.pi/2],
                                   [0.0, 3.0, 0.4, 1.15, 0.0],
                                   [-1., 5.0, 0.4, 1.1, -np.pi/4],
                                   [2.0, 6.0, 0.4, 1.0, np.pi]])  # moving obstacels
        # self.obstacles = np.array([[1.0, 1.0, 0.4, 0.0, 0.0],
        #                            [2.5, 2.0, 0.4, 0.0, 0.0],
        #                            [4.0, 2.0, 0.4, 0.0, 0.0]])  # fixed obstacles
        return self.current_state, {'reset'}

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def check_collision(self, obs):
        for obstacle in self.obstacles:
            if np.linalg.norm(obs[:2] - obstacle[:2]) <= obstacle[2]:
                return True
        return False
    
    def check_collision_near(self):
        for obstacle in self.obstacles:
            if np.linalg.norm(self.current_state[:2] - obstacle[:2]) < obstacle[2] + 0.2:
                self.vector_to_obstacle = obstacle[:2] - self.current_state[:2] 
                angle_to_obstacle = np.arctan(self.vector_to_obstacle[1]/self.vector_to_obstacle[0])
                heading_angle = self.current_state[-3]
                angle_diff = angle_to_obstacle - heading_angle
                self.angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
                if -np.pi/4 < angle_diff < np.pi/4:
                    return True
        self.angle_diff = np.nan
        self.vector_to_obstacle = np.nan
        return False

    def check_goal(self):
        if np.linalg.norm(self.current_state[:2] - self.goal_state[:2]) < 0.2:
            return True
        return False

    # Calculate lateral rotational inertia (cite: https://wenku.baidu.com/view/b12cd2e5f8c75fbfc77db239.html?_wkts_=1695020954985)
    def calculate_yaw_inertia(self):
        front_mass = MASS_NORM / 2.0 + MASS_UNIT * self.front_load_unit_num
        rear_mass = MASS_NORM / 2.0 + MASS_UNIT * self.rear_load_unit_num
        return (front_mass + 0.48 * rear_mass) * (WHEELBASE / 2)**2 + (1-0.48) * rear_mass * (WHEELBASE / 2)**2

    def step(self, action, last_action):
        # Dynamic model reference: https://www.researchgate.net/profile/Philip-Polack/publication/318810853_The_kinematic_bicycle_model_A_consistent_model_for_planning_feasible_trajectories_for_autonomous_vehicles/links/5addcbc2a6fdcc29358b9c01/The-kinematic-bicycle-model-A-consistent-model-for-planning-feasible-trajectories-for-autonomous-vehicles.pdf
        acceleration_signal , front_wheel_angle = action
        px, py, velocity, vx, vy, heading_angle, cos_theta, sin_theta = self.current_state
        # beta = np.arctan(0.5 * np.tan(front_wheel_angle))
        # angle_velocity = velocity * np.sin(beta)
        angle_velocity = -0.5 * velocity * np.tan(front_wheel_angle)

        
        road_friction_coefficient = ROAD_FRICTION_COEFFICIENT_NORM  # Normal non-icy road
        air_resistance = 0.5 * DRAG_COEFFICIENT * AIR_DENSITY * FRONTAL_AREA * velocity**2
        friction = self.mass * ROAD_FRICTION_COEFFICIENT_NORM * 9.8
        adjustment_term =  air_resistance/self.mass + ROLLING_RESISTANCE_COEFFICIENT
        if velocity != 0:
            adjustment_term *= np.sign(velocity)
        if friction >= self.mass * acceleration_signal:
            acceleration = acceleration_signal - adjustment_term
        else:
            acceleration = friction/self.mass - adjustment_term
        
        # self.current_state += np.array([velocity*np.cos(heading_angle+beta),
        #                                 velocity*np.sin(heading_angle+beta),
        #                                 acceleration,
        #                                 angle_velocity*np.exp(road_friction_coefficient/ROAD_FRICTION_COEFFFICIENT_NORM-1)/self.yaw_inertia*self.mass]) * self.dt  
        
        # px, py, v, vx, vy, theta, cos, sin
        self.current_state += np.array([velocity*np.cos(heading_angle),
                                        velocity*np.sin(heading_angle),
                                        acceleration,
                                        0,
                                        0,
                                        angle_velocity*np.exp(road_friction_coefficient/ROAD_FRICTION_COEFFICIENT_NORM-1)/self.yaw_inertia*self.mass, 
                                        0,
                                        0]) * self.dt  
        self.current_state[3:5] = np.array([np.cos(self.current_state[-3])*self.current_state[2],np.sin(self.current_state[-3])*self.current_state[2]])
        self.current_state[-2:] = np.array([np.cos(self.current_state[-3]),np.sin(self.current_state[-3])])

        for obstacle in self.obstacles:
            obstacle_x, obstacle_y, obstacle_radius, obstacle_velocity, obstacle_theta = obstacle
            obstacle += self.dt * np.array([obstacle_velocity*np.cos(obstacle_theta),
                            obstacle_velocity*np.sin(obstacle_theta),
                            0,
                            0,
                            0])
            if not (-3 < obstacle[0] < 7 and -3 < obstacle[1] < 7):
                obstacle[3] -= 10*self.dt
        
        dis_2_goal_scaled = - np.linalg.norm(self.goal_state[:3] - self.current_state[:3]) 
        # print(goal_reward)

        if self.check_goal():
            reward = self.reward_weights['goal_reached'] + dis_2_goal_scaled
            done = True
            self.info = {'goal_reached'}
        elif self.check_collision(self.current_state):
            reward = self.reward_weights['collision'] * 0.5 + dis_2_goal_scaled
            done = True
            self.info = {'obstacle_collision'}
        else:  
            reward = dis_2_goal_scaled
            done = False
            self.info = {'driving...'}
        reward += - np.linalg.norm(action-last_action) * 0.5  
        # print('goal_reward: ', goal_reward, ', total_reward: ', reward)
        return self.current_state, reward, done, False, self.info