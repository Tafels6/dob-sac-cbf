from cvxopt import matrix, solvers
import cvxpy as cp
from scipy.optimize import minimize
import numpy as np

# nlp  non-linear predictive approach
import casadi as ca
import numpy as np


def cbf_casadi(env,obs,u_rl):
    px, py, theta = obs
    v, omega = u_rl
    u = ca.MX.sym('u', 2)  # [acc, omega]
    constraints = []
    lbg = []  # 下界
    ubg = []  # 上界

    '''
    x' = v * cosθ
    y' = v * sinθ
    theta' = omega

         | cosθ   0 |
    X' = | sinθ   0 | * |   v   |
         |  0     1 |   | omega |
    x' =      g(x)    *     u

    '''
    for obstacle in env.obstacles:
        
        x_ob, y_ob, radius_ob = obstacle
        
        
        hx = 0.5 * ((px - x_ob)**2 + (py - y_ob)**2 - radius_ob**2)
        hx_dot = (px - x_ob) * (v * ca.cos(theta)) + (py - y_ob) * (v * ca.sin(theta))
        hx_dx = np.array([
            px - x_ob,
            py - y_ob,
            0,
        ])
        lie_f_hx = 0
        lie_g_hx = np.array([
            (px - x_ob) * ca.cos(theta) + (py - y_ob) * ca.sin(theta),
            0
        ])
        
        alpha_hx = 1 * hx
        
        constraints += [
            lie_f_hx + ca.dot(lie_g_hx, u_rl) + alpha_hx
        ]
        lbg += [
            0,
        ]
        ubg += [
            np.inf,
        ]


    P = np.eye(2)  # 定义二次项系数矩阵
    objective = 0.5 * ca.mtimes([(u - u_rl).T, P, (u - u_rl)])  # 定义目标函数

    # 设置求解器选项
    opts = {"ipopt.print_level": 0, "print_time": 0}  # 不显示求解过程

    # 设置和求解优化问题
    nlp = {"x": u, "f": objective, "g": ca.vertcat(*constraints)}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    solution = solver(lbx=[env.action_space.low[0], env.action_space.low[1]], ubx=[env.action_space.high[0], env.action_space.high[1]], lbg=lbg, ubg=ubg)

    # 提取解
    u_opt = solution["x"]
    # print('done')
    return u_opt.full().squeeze(-1)