import numpy as np

def kepler_J2_ODE(t, x, params):
    # get params
    mu_E = params['mu_E']
    J2 = params['J2']
    R_E = params['R_E']
    n_objs = params['n_objs']

    dxdt = np.zeros((n_objs * 6))

    for k in range(n_objs):
        r_vec = x[6*k:3+6*k]
        X = r_vec[0]
        Y = r_vec[1]
        Z = r_vec[2]
        rdot_vec = x[3+6*k:6+6*k]
        r = np.linalg.norm(r_vec)
        Z_r_2 = np.power(Z / r, 2.0)
        p_J2 = 1.5 * J2 * mu_E / np.power(r, 2.0) * np.power(R_E / r, 2.0) * \
               np.array([X/r*(5.0*Z_r_2 - 1.0), Y/r*(5.0*Z_r_2 - 1.0), Z/r*(5.0*Z_r_2 - 3.0)])
        rddot_vec = -mu_E / np.power(r, 3.0)*r_vec + p_J2
        dxdt[6*k:6*k+6] = np.concatenate((rdot_vec, rddot_vec))

    return dxdt

