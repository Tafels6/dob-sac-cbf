import numpy as np
import casadi as ca

def cbf_casadi(env,obs,u_rl,sense_range,d_hat):
    px, py, v, theta = obs[[0,1,2,5]]
    d_hat_tmp = d_hat[[0,1,2,5]]

    u = ca.MX.sym('u', 2)  # [acc, omega]
    constraints = []
    lbg = []  # lower bound
    ubg = []  # upper bound

    '''
    x' = v * cosθ
    y' = v * sinθ
    v' = acc
    theta' = omega
    where: omega = velocity * arctan(0.5 * tan(front_wheel_angle))

         | v * cosθ |   | 0   0 |
    X' = | v * sinθ | + | 0   0 | * |  acc  |
         |    0     |   | 1   0 |   | omega |
         |    0     |   | 0   1 |
    x' =     f(x)     +     g(x)  *     u
    '''
    for other_car in env.other_cars:
        if np.linalg.norm(other_car[:2] - obs[:2]) >= sense_range:
            continue
        # other_car: oc
        x_oc, y_oc, v_oc, theta_oc, _ = other_car
        radius_oc = 0.5
        
        # mpc
        x_oc_tmp, y_oc_tmp, v_oc_tmp, theta_oc_tmp = x_oc, y_oc, v_oc, theta_oc
        px_tmp, py_tmp, v_tmp, theta_tmp = px, py, v, theta
        slip_angle = ca.arctan(0.5*ca.tan(u[1]))
        for _ in range(50):
            x_oc_tmp += v_oc_tmp * ca.cos(theta_oc_tmp) * env.dt * 1
            y_oc_tmp += v_oc_tmp * ca.sin(theta_oc_tmp) * env.dt * 1
        
            px_tmp += v_tmp*ca.cos(theta_tmp + slip_angle) * env.dt  * 1
            py_tmp += v_tmp*ca.sin(theta_tmp + slip_angle) * env.dt  * 1
            v_tmp += u[0] * env.dt * 1
            theta_tmp += v_tmp * ca.sin(slip_angle) * env.dt  * 1
            constraints += [
                (x_oc_tmp-px_tmp)**2 + (y_oc_tmp-py_tmp)**2 - radius_oc**2,
            ]
            lbg += [
                0,
            ]
            ubg += [
                np.inf,
            ]

    # safer geometric constraints
    px_geo, py_geo, v_geo, theta_geo = px, py, v, theta
    for _ in range(5):
        px_geo += v_geo*ca.cos(theta_geo + slip_angle) * env.dt  * 1
        py_geo += v_geo*ca.sin(theta_geo + slip_angle) * env.dt  * 1
        v_geo += u[0] * env.dt * 1
        theta_geo += v_geo * ca.sin(slip_angle) * env.dt  * 1

    # lane
    if abs(py) >= 4.5:
        hx = 0.5 * (3**2- px**2)
        hx_dot = -px * v * ca.cos(theta)
        lie_f_hx = -px * v * ca.cos(theta)
        lie_f_hx_dx = np.array([
            -v * ca.cos(theta),
            0,
            -px * ca.cos(theta),
            px * v * ca.sin(theta)
        ])
        lie_f2_hx = -v**2 * ca.cos(theta)**2
        lie_g_lie_f_hx = np.array([
            -px * ca.cos(theta),
            px * v * ca.sin(theta)
        ])
        phi0 = hx
        alpha1_phi0 = 1 * hx
        lie_f_alpha1_phi0 = hx_dot
        phi1 = hx_dot + alpha1_phi0
        alpha2_phi1 = 2 * (hx_dot + 1 * hx)**3

        constraints += [
            lie_f2_hx + ca.dot(lie_g_lie_f_hx, ca.vertcat(u[0],v*ca.arctan(0.5*ca.tan(u[1])))) + alpha2_phi1 + lie_f_alpha1_phi0 + ca.dot(lie_f_hx_dx, d_hat_tmp)
        ]
        lbg += [
            0,
        ]
        ubg += [
            np.inf,
        ]

        # geometric
        constraints += [
            px_geo
        ]
        lbg += [
            -3,
        ]
        ubg += [
            3,
        ]

    if abs(px) >= 4.5:
        hx = 0.5 * (3**2- py**2)
        hx_dot = -py * v * ca.sin(theta)
        lie_f_hx = -py * v * ca.sin(theta)
        lie_f_hx_dx = np.array([
            0,
            -v * ca.sin(theta)
            -py * ca.sin(theta)
            -py * v * ca.cos(theta)
        ])
        lie_f2_hx = -v**2 * ca.sin(theta)**2
        lie_g_lie_f_hx = np.array([
            -py * ca.sin(theta),
            -py * v * ca.cos(theta)
        ])

        lie_f_alpha1_phi0 = hx_dot
        alpha2_phi1 = 2 * (hx + hx_dot)**3

        constraints += [
            lie_f2_hx + ca.dot(lie_g_lie_f_hx, ca.vertcat(u[0], v*ca.arctan(0.5*ca.tan(u[1])))) + alpha2_phi1 + lie_f_alpha1_phi0 + ca.dot(lie_f_hx_dx, d_hat_tmp)
        ]
        lbg += [
            0,
        ]
        ubg += [
            np.inf,
        ]

        # geometric constraints
        constraints += [
            py_geo
        ]
        lbg += [
            -3,
        ]
        ubg += [
            3,
        ]

    if abs(px) <= 4.5 and abs(py) <= 4.5:
        combinations = [[1,1],[1,-1],[-1,1],[-1,-1]]
        for sign_x, sign_y in combinations:
            lane_arc = np.array([4.5*sign_x, 4.5*sign_y, 1.5])

            x_lane_arc_center, y_lane_arc_center, radius_lane_arc = lane_arc
            hx = 0.5 * ((px - x_lane_arc_center)**2 + (py - y_lane_arc_center)**2 - radius_lane_arc**2)
            hx_dot = (px - x_lane_arc_center) * v * np.cos(theta) +  (py - y_lane_arc_center) * v * np.sin(theta)
            lie_f_hx = (px - x_lane_arc_center) * v * np.cos(theta) +  (py - y_lane_arc_center) * v * np.sin(theta)
            lie_f_hx_dx = np.array([
                v * np.cos(theta),
                v * np.sin(theta)
                (px - x_lane_arc_center) * np.cos(theta) +  (py - y_lane_arc_center) * np.sin(theta)
                -(px - x_lane_arc_center) * v * np.sin(theta) +  (py - y_lane_arc_center) * v * np.cos(theta)
            ])
            lie_f2_hx = v**2
            lie_g_lie_f_hx = np.array([
                (px - x_lane_arc_center) * np.cos(theta) +  (py - y_lane_arc_center) * np.sin(theta),
                -(px - x_lane_arc_center) * v * np.sin(theta) +  (py - y_lane_arc_center) * v * np.cos(theta)
            ])
            lie_f_alpha1_phi0 = lie_f_hx  # alpha1_phi0 = hx
            alpha1_phi0 = hx
            alpha2_phi1 = 2 * (hx + hx_dot)**3

            constraints += [
                0*lie_f2_hx + ca.dot(lie_g_lie_f_hx, ca.vertcat(u[0], v*ca.arctan(0.5*ca.tan(u[1])))) + lie_f_alpha1_phi0 + alpha2_phi1 + ca.dot(lie_f_hx_dx, d_hat_tmp)
            ]
            lbg += [
                0,
            ]
            ubg += [
                np.inf,
            ]

            # geometric constraints
            constraints += [
                (px_geo-x_lane_arc_center)**2 + (py_geo-y_lane_arc_center)**2 - radius_lane_arc**2
            ]
            lbg += [
                0,
            ]
            ubg += [
                np.inf,
            ]

    P = np.eye(2)
    objective = 0.5 * ca.mtimes([(u - u_rl).T, P, (u - u_rl)])

    opts = {"ipopt.print_level": 0, "print_time": 0} 

    nlp = {"x": u, "f": objective, "g": ca.vertcat(*constraints)}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    solution = solver(lbx=[env.action_space.low[0], env.action_space.low[1]], ubx=[env.action_space.high[0], env.action_space.high[1]], lbg=lbg, ubg=ubg)

    u_opt = solution["x"]
    # print('done')
    return u_opt.full().squeeze(-1)
