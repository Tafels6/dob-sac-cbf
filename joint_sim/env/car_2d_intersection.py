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

'''
Car2DIntersection - Intersection scenario

     10        |   '   |
               |   '   |
      3 -------/   '   \-------
        t
      0 -  -  -    +    -  -  -

     -3 -------\   '   /-------
           7   |   '   |
    -10        |   ' s |
       -10    -3   0   3      10
    
    s: start    [1.5, -10]
    t: target   [-10, 1.5]
    Single lane width: 3m
    Intersection arc radius: 1.5m
    Drive on the right side

'''
# Lane parameters
LANE_WIDTH = 3
LANE_TURNING_RADIUS = 1.5

class Car2DIntersection(gym.Env):
    def __init__(self):
        super(Car2DIntersection, self).__init__()

        self.observation_space = spaces.Box(low=np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-1,-1]), high=np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,1,1]), dtype=np.float32)  # pos_x, pos_y, vx, vy, v, theta, ∈R
        self.action_space = spaces.Box(low=np.array([-7,-np.deg2rad(45)]), high=np.array([7,np.deg2rad(45)]), dtype=np.float32)  # acceleration , front_wheel_angle
        self.front_load_unit_num = 0
        self.rear_load_unit_num = 0
        self.mass = MASS_NORM + MASS_UNIT * (self.front_load_unit_num + self.rear_load_unit_num)
        self.yaw_inertia = self.calculate_yaw_inertia()

        self.current_state = np.array([1.5, -10., 0., 0., 0., np.pi/2, 0., 1.])  # px, py, v, vx, vy, theta, cos, sin
        self.goal_state = np.array([-10., 1.5, 5, np.pi])  # px, py, v, theta
        self.reward_weights = np.array([0.5,0.5,0.5,1.0,1.0])

        # Initialize other vehicles with oc_x, oc_py, oc_v, oc_theta, turn_option (-1: left turn, 0: straight, 1: right turn)
        self.other_cars_init = np.array([
            # West to East
            [-10., -1.5, 4., 0, 0], 
            [-40., -1.5, 4., 0, 0],
            # [-40., -1.5, 5., 0, -1],
            # North to South
            [-1.5, 20., 4., -np.pi/2, 0],
            [-1.5, 60., 4., -np.pi/2, 0], 
            # [-1.5, 60., 5., -np.pi/2, -1],
            # East to West
            [40., 1.5, 4., np.pi, 0],
            [50., 1.5, 4., np.pi, 0],
            # [50., 1.5, 5., np.pi, -1],
        ])  
        self.other_cars = np.copy(self.other_cars_init)

        self.dt = 0.02  # Simulation time step
        self.info = {}

        self.rewards = {
            'goal_reached': 1.,
            'collision': -1.,
            'in_lane': 1.,
        }

    def reset(self):
        self.current_state = np.array([1.5, -10., 0., 0., 0., np.pi/2, 0., 1.])
        self.other_cars = np.copy(self.other_cars_init)
        return self.current_state, {'reset'}

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def check_collision(self):
        for other_car in self.other_cars:
            if np.linalg.norm(self.current_state[:2] - other_car[:2]) <= 0.4:
                return True, {'collision_car'}
        px, py = self.current_state[:2]
        if abs(px) >= LANE_WIDTH+LANE_TURNING_RADIUS:
            if abs(py) >= LANE_WIDTH+0.1:
                return True, {'collision_lane'}
        if abs(py) >= LANE_WIDTH+LANE_TURNING_RADIUS:
            if abs(px) >= LANE_WIDTH+0.1:
                return True, {'collision_lane'}
        if abs(px) < LANE_WIDTH+LANE_TURNING_RADIUS and abs(py) < LANE_WIDTH+LANE_TURNING_RADIUS:
            if np.linalg.norm(np.array([abs(px),abs(py)]) - np.array([LANE_WIDTH+LANE_TURNING_RADIUS,LANE_WIDTH+LANE_TURNING_RADIUS])) <= LANE_TURNING_RADIUS:
                return True, {'collision_lane_arc'}
        return False, {}
    
    def check_in_lane(self):
        px, py = self.current_state[:2]
        if abs(px) >= LANE_WIDTH+LANE_TURNING_RADIUS:
            if abs(py) < CAR_WIDTH/2 or abs(py) > LANE_WIDTH - CAR_WIDTH/2:  
                return False
        if abs(py) >= LANE_WIDTH+LANE_TURNING_RADIUS:
            if abs(px) < CAR_WIDTH/2 or abs(px) > LANE_WIDTH - CAR_WIDTH/2:
                return False
        return True

    def check_goal(self):
        return np.linalg.norm(self.current_state[:2] - self.goal_state[:2]) < 0.5
    
    def reward2goal(self):
        x_agent, y_agent, v_agent = self.current_state[:3]
        cos_agent, sin_agent = self.current_state[-2:]
        x_goal, y_goal, v_goal, theta_goal = self.goal_state

        reward = - np.linalg.norm(self.reward_weights*(
            np.array([x_agent, y_agent, v_agent, cos_agent, sin_agent]) -    \
            np.array([x_goal, y_goal, v_goal, np.cos(theta_goal), np.sin(theta_goal)]))
        )
        return reward

    # Calculate lateral rotational inertia (cite: https://wenku.baidu.com/view/b12cd2e5f8c75fbfc77db239.html?_wkts_=1695020954985)
    def calculate_yaw_inertia(self):
        front_mass = MASS_NORM / 2.0 + MASS_UNIT * self.front_load_unit_num
        rear_mass = MASS_NORM / 2.0 + MASS_UNIT * self.rear_load_unit_num
        return (front_mass + 0.48 * rear_mass) * (WHEELBASE / 2)**2 + (1-0.48) * rear_mass * (WHEELBASE / 2)**2

    def step(self, action, last_action, real_world, high_order_model):
        # Dynamic model reference: https://www.researchgate.net/profile/Philip-Polack/publication/318810853_The_kinematic_bicycle_model_A_consistent_model_for_planning_feasible_trajectories_for_autonomous_vehicles/links/5addcbc2a6fdcc29358b9c01/The-kinematic-bicycle-model-A-consistent-model-for-planning-feasible-trajectories-for-autonomous-vehicles.pdf
        acceleration_signal, front_wheel_angle = action
        px, py, velocity, vx, vy, heading_angle, cos_theta, sin_theta = self.current_state
        
        turn_tuning_for_real_world = 1.0
        
        if real_world:
            # Power calculations for real-world dynamics
            road_friction_coefficient = ROAD_FRICTION_COEFFICIENT_NORM  # Normal non-icy road
            air_resistance = 0.5 * DRAG_COEFFICIENT * AIR_DENSITY * FRONTAL_AREA * velocity**2
            friction = self.mass * ROAD_FRICTION_COEFFICIENT_NORM * 9.8
            turn_tuning_for_real_world = np.exp(road_friction_coefficient / ROAD_FRICTION_COEFFICIENT_NORM - 1) / self.yaw_inertia * self.mass
            # Check if the ground provides enough friction
            adjustment_term = air_resistance / self.mass + ROLLING_RESISTANCE_COEFFICIENT
            if velocity != 0:
                adjustment_term *= np.sign(velocity)
            if friction >= self.mass * acceleration_signal:
                acceleration = acceleration_signal - adjustment_term
            else:
                acceleration = friction / self.mass - adjustment_term
        else:
            acceleration = acceleration_signal
        
        # Dynamic model selection
        if high_order_model:
            # Un-simplified model
            slip_angle = np.arctan(0.5 * np.tan(front_wheel_angle))
            self.current_state += np.array([velocity * np.cos(heading_angle + slip_angle),
                                            velocity * np.sin(heading_angle + slip_angle),
                                            acceleration,
                                            0,
                                            0,
                                            velocity * np.sin(slip_angle) * turn_tuning_for_real_world,  # Add effect of friction and inertia for turning
                                            0,
                                            0]) * self.dt
        else:
            # Simplified model using first-order CBF
            slip_angle = np.arctan(0.5 * np.tan(front_wheel_angle))

            self.current_state += np.array([velocity * np.cos(heading_angle) - velocity * slip_angle * np.sin(heading_angle),
                                            velocity * np.sin(heading_angle) + velocity * slip_angle * np.cos(heading_angle),
                                            acceleration,
                                            0,
                                            0,
                                            slip_angle * velocity * turn_tuning_for_real_world,  # Add effect of friction and inertia for turning
                                            0,
                                            0]) * self.dt
        
        self.current_state[3:5] = np.array([np.cos(self.current_state[5]) * self.current_state[2], np.sin(self.current_state[5]) * self.current_state[2]])
        self.current_state[-2:] = np.array([np.cos(self.current_state[5]), np.sin(self.current_state[5])])

        for other_car in self.other_cars:  # x, y, v, theta, turn_option
            x_oc, y_oc, v_oc, theta_oc, option_oc = other_car
            if (abs(x_oc) <= LANE_WIDTH + LANE_TURNING_RADIUS) and (abs(y_oc) <= LANE_WIDTH + LANE_TURNING_RADIUS):
                if option_oc >= 0:
                    theta_oc -= self.dt * option_oc * v_oc / (LANE_TURNING_RADIUS + 0.5 * LANE_WIDTH)
                else:
                    theta_oc -= self.dt * option_oc * v_oc / (LANE_TURNING_RADIUS + 1.5 * LANE_WIDTH)
            
            other_car += self.dt * np.array([v_oc * np.cos(theta_oc),
                                            v_oc * np.sin(theta_oc),
                                            0,
                                            0,
                                            0])
            other_car[3] = theta_oc
        
        # Rewards calculation
        reward = self.reward2goal()
        reward += 1.5 if self.check_in_lane() else -1.5  # Traffic rules adherence
        reward += - np.linalg.norm(action - last_action) * 0.5  # Action smoothness penalty
        collision, collision_info = self.check_collision()

        if self.check_goal():
            done = True
            self.info = {'goal_reached'}
        elif collision:
            reward -= 1
            done = True
            self.info = collision_info
        else:  
            done = False
            self.info = {'driving...'}

        return self.current_state, reward, done, False, self.info