from cvxopt import matrix, solvers
import numpy as np

def cbf(env, obs, u_rl, d_hat,sense_range):
    action_num = env.action_space.shape[0]
    P = matrix(np.diag(np.ones(action_num)), tc='d')
    q = matrix(np.zeros((action_num,1)))
    G = np.array([]).reshape(0, 2)  
    h = np.array([])  
    px, py, v, vx, vy, theta, cos_theta, sin_theta = obs
    d_hat_tmp = d_hat[[0,1,2,5]]
    for obstacle in env.obstacles:
        if np.linalg.norm(obstacle[:2] - obs[:2]) >= sense_range:
            continue
        obstacle_x, obstacle_y, obstacle_radius, obstacle_velocity, obstacle_theta = obstacle
        acc, front_wheel_angle = u_rl
        # beta = np.arctan(0.5*np.tan(front_wheel_angle))
        # angle_velocity = v * np.sin(beta)
        angle_velocity = -v * 0.5 * np.tan(front_wheel_angle)
        u_rl_tmp = np.array([acc,angle_velocity])
        lie_f_hx_dx = np.array([[v * np.cos(theta),
                                v * np.sin(theta),
                                (px - obstacle_x) * np.cos(theta) + (py - obstacle_y) * np.sin(theta),
                                -(px - obstacle_x) * (v * np.sin(theta)) + (py - obstacle_y) * v * np.cos(theta)]])

        G_tmp = - np.array([[(px - obstacle_x) * np.cos(theta) + (py - obstacle_y) * np.sin(theta),
                    -(px - obstacle_x) * v * np.sin(theta) + (py - obstacle_y) * v * np.cos(theta)]])
        h_tmp = 1* v**2+ \
            1*(0.5 * ((px - obstacle_x)**2 + (py - obstacle_y)**2 - obstacle_radius**2) + ((px - obstacle_x) * (v * np.cos(theta) - obstacle_velocity * np.cos(obstacle_theta)) + (py - obstacle_y) * (v * np.sin(theta) - obstacle_velocity * np.sin(obstacle_theta)))) \
            + 1*((px - obstacle_x) * (v * np.cos(theta) - obstacle_velocity * np.cos(obstacle_theta)) + (py - obstacle_y) * (v * np.sin(theta) - obstacle_velocity * np.sin(obstacle_theta)))**1 \
            - np.dot(G_tmp,u_rl_tmp)  \
            + 1*np.dot(lie_f_hx_dx, d_hat_tmp)
        G = np.vstack((G, G_tmp))
        h = np.append(h, h_tmp)

    u_bar = matrix([0.0,0.0])
    u_modified = False
    if np.all(h>=0):
        u_modified = False
    else:
        u_modified = True
        G = matrix(G,tc='d')
        h = matrix(h,tc='d')
        try:
            solvers.options['show_progress'] = False
            sol = solvers.qp(P, q, G, h)
            u_bar = sol['x']
        except:
            print('obs: ', obs)
            print('u_rl_tmp: ',u_rl_tmp)
            print('P: ', P, 'q: ', q, 'G: ', G, 'h: ', h)
            for i,obstacle in enumerate(env.obstacles):
                if np.linalg.norm(obstacle[:2] - obs[:2]) < sense_range:
                    print(i,'_obstacle',obstacle)
    if u_modified:
        u_tmp_after_cbf = u_rl_tmp + np.squeeze(u_bar)
        acc_, angle_velocity_ = u_tmp_after_cbf
        # beta_ = np.arcsin(angle_velocity_/v)
        # front_wheel_angle_ = np.arctan(np.tan(beta_) * 2)
        angle_velocity_ = -angle_velocity_
        front_wheel_angle_ = np.arctan(2*angle_velocity_/(v+1e-9))
        front_wheel_angle_ = np.clip(front_wheel_angle_, env.action_space.low[1], env.action_space.high[1])
        acc_ = np.clip(acc_,env.action_space.low[0], env.action_space.high[0])
        u_after_cbf = np.array([acc_, front_wheel_angle_])
    else:
        u_after_cbf = u_rl

    return u_after_cbf