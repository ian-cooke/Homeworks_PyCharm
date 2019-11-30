# imports
import numpy as np
from scipy import integrate
import HW1.macros as macros


# # #
# function that defines the rate of change for a relative motion problem
# INERTIAL FRAME ONLY
# considers the state vector as 3 6x1 state vectors stacked on top of each other
# first six states is chief, second is dep, third is desired dep
# Inputs:
#   t: time
#   x: current state
#   u: control
#   params: parameters needed
# Outputs:
#   dxdt: time rate of change
def kepler_J2_ODE(t, x, u, params):

    # get params
    mu = params['mu']
    J2 = params['J2']
    R = params['R']
    n_objs = params['n_objs']
    use_J2 = params['use_J2']
    n_deps = params['n_deps']
    n_states = params['n_states']

    # preallocate
    dxdt = np.matrix(np.zeros(((2 * n_deps + 1) * n_states, 1)))

    # loop
    for k in range(2 * n_deps + 1):

        # extract and assign
        r_vec = x[6 * k : 6 * k + 1]
        X = r_vec[0]
        Y = r_vec[1]
        Z = r_vec[2]
        rdot_vec = x[6 * k + 3 : 6 * k + 6]
        r = np.linalg.norm(r_vec)

        # total acceleration
        rddot_vec = -mu / np.power(r, 3.0) * r_vec

        # perturbation due to J2
        if use_J2:
            J2_vec = np.matrix([[X / r * (5 * np.power(Z / r, 2.0) - 1)], [Y / r * (5 * np.power(Z / r, 2.0) - 1)], [Z / r * (5 * np.power(Z / r, 2.0) - 3)]])
            p_J2 = 1.5 * J2 * mu / np.power(r, 2.0) * np.power(R / r, 2.0) * J2_vec
            rddot_vec = rddot_vec + p_J2

        # control
        if np.mod(k+1, 2) == 0:
            rddot_vec = rddot_vec + u

        # dxdt
        dxdt[n_states * k : n_states * k + 6] = np.concatenate((rdot_vec, rddot_vec))

# # #
# function that simulates a ff scenario with full nonlinear control
# Inputs:
#   t: time
#   x: current state
#   u: control
#   params: parameters needed
# Outputs:
#   dxdt: time rate of change
def nonlinear_control_sim(x_chief_N_0, x_dep_N_0, x_dep_d_N_0, tspan, constant_sep, params):

    # params
    K_p = params['K_p']
    K_d = params['K_d']
    mu = params['mu']
    J2 = params['J2']
    R = params['R']
    n_objs = params['n_objs']
    use_J2 = params['use_J2']
    n_deps = params['n_deps']
    n_states = params['n_states']
    x_dep_d_N_sep = params['x_dep_d_N_sep']

    # preallocate
    n_steps = np.size(tspan)
    x_chief_N = np.matrix(np.zeros((n_states, n_steps)))
    x_dep_N = np.matrix(np.zeros((n_states, n_steps)))
    x_dep_d_N = np.matrix(np.zeros((n_states, n_steps)))
    u_N = np.matrix(np.zeros((3, n_steps)))

    # assign first index
    x_chief_N[:, 0] = x_chief_N_0
    x_dep_N_0[:, 0] = x_dep_N_0
    x_dep_d_N_0[:, 0] = x_dep_d_N_0

    # loop
    for k in range(n_steps-1):

        # calculate the control
        r_d = np.linalg.norm(x_dep_N[0:3, k])
        r_d_vec = x_dep_N[0:3, k]
        r_dd = np.linalg.norm(x_dep_d_N[0:3, k])
        r_dd_vec = x_dep_d_N[0:3, k]
        delta_r = r_d_vec - r_dd_vec
        delta_rdot = x_dep_N[3:6, k] - x_dep_d_N[3:6, k]
        u_N[0:3, k] = -(-mu/np.power(r_d, 3.0) * r_d_vec + mu/np.power(r_dd, 3.0) * r_dd_vec) - K_p * delta_r - K_d * delta_rdot

        # enforce numerical precision condition
        if np.linalg.norm(u_N[:, k]) < 1.0e-12:
            u_N[:, k] = np.matrix(np.zeros((3, 1)))

        # integrate
        x_0 = np.concatenate((x_chief_N[:, k], x_dep_N[:, k], x_dep_d_N[:, k]), axis=0)
        x_sol = integrate.odeint(func=kepler_J2_ODE, t=np.array(tspan[k], tspan[k+1]), y0=x_0, tfirst=True, args=(u_N[:, k], params,),
                                 rtol=1.0e-12, atol=1.0e-12)

        # assign
        x_chief_N[:, k+1] = np.matrix(np.transpose(x_sol[-1, 0:6]))
        x_dep_N[:, k+1] = np.matrix(np.transpose(x_sol[-1, 6:12]))
        if constant_sep:
            x_dep_d_N[:, k+1] = macros.convert_eci_hill('hill', 'eci', x_chief_N[:, k+1], x_dep_d_N_sep)
        else:
            x_dep_d_N[:, k+1] = np.matrix(np.transpose(x_sol[-1, 12:18]))

