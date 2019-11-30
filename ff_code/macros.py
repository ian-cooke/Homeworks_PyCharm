# imports
import numpy as np


# # #
# convert_rv_kep
# translate between the state vector and classical keplerian orbit elements
# Inputs:
#   input_flag - type of state being inputted
#   x - the numbers
#   delta_t - time elapsed since t_0
#   output_flag - type of state being outputted
def convert_rv_kep(input_flag, x_vec, delta_t, output_flag, radians=False, output_M=False):

    # define some constants
    mu_E = 398600.0  # [km^3/s^2] standard gravitational parameter of the Earth

    # Handle various cases, if none then it just returns
    # keplerian to state vector
    if input_flag == 'keplerian' and output_flag == 'cartesian':
        # parse
        a = x_vec[0]  # [km]
        ecc = x_vec[1]  # [none]
        inc = np.deg2rad(x_vec[2])  # [deg]
        Omega = np.deg2rad(x_vec[3])  # [deg]
        omega = np.deg2rad(x_vec[4])  # [deg]
        M_0 = np.deg2rad(x_vec[5])  # [deg]

        if ecc < 1.0:
            M = M_0 + np.sqrt(mu_E / np.power(a, 3.0)) * delta_t
            f = convert_M_to_f(M, 6, ecc)
        else:
            n = np.sqrt(mu_E / np.power(-a, 3.0))
            N = M_0 + n*delta_t
            f = convert_M_to_f(N, 6, ecc)

        theta = omega + f

        p = a * (1.0 - np.power(ecc, 2.0))

        h = np.sqrt(mu_E * p)

        r = p / (1.0 + ecc*np.cos(f))

        r_x = r * (np.cos(Omega)*np.cos(theta) - np.sin(Omega)*np.sin(theta)*np.cos(inc))
        r_y = r * (np.sin(Omega)*np.cos(theta) + np.cos(Omega)*np.sin(theta)*np.cos(inc))
        r_z = r * (np.sin(theta)*np.sin(inc))

        v_x = -mu_E / h * (np.cos(Omega)*(np.sin(theta) + ecc*np.sin(omega))
                           + np.sin(Omega)*(np.cos(theta) + ecc*np.cos(omega))*np.cos(inc))
        v_y = -mu_E / h * (np.sin(Omega)*(np.sin(theta) + ecc*np.sin(omega))
                           - np.cos(Omega)*(np.cos(theta) + ecc*np.cos(omega))*np.cos(inc))
        v_z = mu_E / h * (np.cos(theta) + ecc*np.cos(omega))*np.sin(inc)

        x_out = np.array([r_x, r_y, r_z, v_x, v_y, v_z])

        return x_out

    # state vector to keplerian
    elif input_flag == 'cartesian' and output_flag == 'keplerian':
        # parse
        x = x_vec[0]
        y = x_vec[1]
        z = x_vec[2]
        xd = x_vec[3]
        yd = x_vec[4]
        zd = x_vec[5]

        r_vec = np.array([x, y, z])
        v_vec = np.array([xd, yd, zd])

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        one_over_a = 2.0 / r - np.power(v, 2.0) / mu_E
        a = 1.0 / one_over_a

        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)

        ecc_vec = np.cross(v_vec, h_vec) / mu_E - r_vec / r
        ecc = np.linalg.norm(ecc_vec)

        ihat_e = ecc_vec / ecc
        ihat_h = h_vec / h
        ihat_p = np.cross(ihat_h, ihat_e)

        PN = np.column_stack((ihat_e, ihat_p, ihat_h))

        Omega = np.arctan2(PN[2, 0], -PN[2, 1])
        inc = np.arccos(PN[2, 2])
        omega = np.arctan2(PN[0, 2], PN[1, 2])
        ihat_r = r_vec / r
        f = np.arctan2(np.dot(np.cross(ihat_e, ihat_r), ihat_h), np.dot(ihat_e, ihat_r))
        if ecc < 1.0:
            E = 2.0*np.arctan(np.tan(f/2.0) / np.sqrt((1.0 + ecc)/(1.0 - ecc)))
            M = E - ecc*np.sin(E)
            n = np.sqrt(mu_E / np.power(a, 3.0))
        else:
            H = 2.0*np.arctanh(np.tan(f/2) / np.sqrt((ecc + 1.0) / (ecc - 1.0)))
            M = ecc*np.sinh(H) - H
            n = np.sqrt(mu_E / np.power(-a, 3.0))

        if output_M:
            M_0 = M
        else:
            M_0 = M - n * delta_t
            if M_0 < 0:
                M_0 = M_0 + 2 * np.pi

        if radians:
            return np.array([a, ecc, inc, Omega, omega, M_0])
        else:
            return np.array([a, ecc, np.rad2deg(inc), np.rad2deg(Omega), np.rad2deg(omega), np.rad2deg(M_0)])


    # flags are the same
    elif input_flag == output_flag:
        return x_vec
    # inputs are wrong
    else:
        raise ValueError('Incorrect input or output flags')


# # #
# subroutine for newton's method to solve keplers equation for E (eccentric anomaly)
# Inputs:
#   x_0 - [deg] initial guess
#   n_iter - [none] number of iterations to be completed
#   ecc - eccentricity of orbit
# Outputs:
#   x_k - final solution
def convert_M_to_f(x_0, n_iter, ecc):

    # iterate
    x_k = x_0
    for k in range(n_iter):
        # elliptic case
        if ecc < 1.0:
            x_k = x_k - (x_0 - (x_k - ecc*np.sin(x_k)))/-(1.0 - ecc*np.cos(x_k))
        # hyperbolic case
        else:
            x_k = x_k - (x_0 - (ecc*np.sinh(x_k) - x_k))/-(ecc*np.cosh(x_k) - 1)

    # elliptic case
    if ecc < 1.0:
        f = 2.0 * np.arctan(np.sqrt((ecc + 1.0)/(1.0 - ecc)) * np.tan(x_k / 2.0))
    # hyperbolic case
    else:
        f = 2.0 * np.arctan(np.sqrt((ecc + 1.0) / (ecc - 1.0)) * np.tanh(x_k / 2.0))

    return f


# # #
# subroutine to convert between hill and eci frames w/ flags
# Inputs:
#   input_flag - input frame
#   output_flag - output frame
#   x_chief_N - 6x1 vector of inertial chief states
#   x_dep_in - 6x1 vector of deputy states (either hill or eci, depending on flags)
# Outputs:
#   x_dep_out - 6x1 vector of deputy states, converted
#   dcm - frame transformation matrix
def convert_eci_hill(input_flag, output_flag, x_chief_N, x_dep_in):

    # chief
    r_chief_N = x_chief_N[0:3]
    rdot_chief_N = x_chief_N[3:6]

    if input_flag == 'eci' and output_flag == 'hill':

        # deputy
        r_dep_N = x_dep_in[0:3]
        rdot_dep_N = x_dep_in[3:6]

        # angular momentum
        h_chief_N = np.cross(r_chief_N, rdot_chief_N)

        # for angular velocity
        fdot = np.linalg.norm(h_chief_N) / np.power(np.linalg.norm(r_chief_N), 2.0)
        omega_ON_O = np.array([0.0, 0.0, fdot])

        # O-frame unit vectors
        ohat_r_N = r_chief_N / np.linalg.norm(r_chief_N)
        ohat_h_N = h_chief_N / np.linalg.norm(h_chief_N)
        ohat_t_N = np.cross(ohat_h_N, ohat_r_N)

        # eci to hill dcm
        dcm = np.transpose(np.column_stack((ohat_r_N, ohat_t_N, ohat_h_N)))

        # convert
        rho_O = np.dot(dcm, r_dep_N - r_chief_N)
        rhoprime_O = np.dot(dcm, rdot_dep_N - rdot_chief_N) - np.cross(omega_ON_O, rho_O, axis=0)

        # concatenate
        x_dep_out = np.concatenate((rho_O, rhoprime_O))

    elif input_flag == 'hill' and output_flag == 'eci':

        # deputy
        rho_O = x_dep_in[0:3]
        rhoprime_O = x_dep_in[3:6]

        # angular momentum
        h_chief_N = np.cross(r_chief_N, rdot_chief_N)

        # for angular velocity
        fdot = np.linalg.norm(h_chief_N) / np.power(np.linalg.norm(r_chief_N), 2.0)
        omega_ON_O = np.array([0.0, 0.0, fdot])

        # O-frame unit vectors
        ohat_r_N = r_chief_N / np.linalg.norm(r_chief_N)
        ohat_h_N = h_chief_N / np.linalg.norm(h_chief_N)
        ohat_t_N = np.cross(ohat_h_N, ohat_r_N)

        # hill to eci dcm
        dcm = np.column_stack((ohat_r_N, ohat_t_N, ohat_h_N))

        # convert
        r_dep_N = r_chief_N + np.dot(dcm, rho_O)
        rdot_dep_N = rdot_chief_N + np.dot(dcm, rhoprime_O + np.cross(omega_ON_O, rho_O, axis=0))

        # concatenate
        x_dep_out = np.concatenate((r_dep_N, rdot_dep_N))

    elif input_flag == output_flag:
        x_dep_out = x_dep_in
        dcm = np.identity(3)

    else:
        raise ValueError('Incorrect Input or Output Flags')

    return x_dep_out, dcm

# # #
# subroutine to convert ODE solution between hill and eci
# Inputs:
#   input_flag - input frame
#   output_flag - output frame
#   x_sol_in - in states (chief always inertial, dep can be either)
#   params - the params dict
# Outputs:
#   x_sol_out - outputted solution
def convert_ODEsol(input_flag, output_flag, x_sol_in, params):

    # preallocate
    N, n = x_sol_in.shape
    x_sol_out = np.zeros((N, n))
    n_deps = params['n_deps']
    n_states = params['n_states']

    # inertial chief same as before
    x_sol_out[:, :n_states] = x_sol_in[:, :n_states]

    # hill to eci
    if input_flag == 'hill' and output_flag == 'eci':

        # loop over deps
        for i in range(n_deps):

            ind = (i + 1) * n_states

            # loop over time
            for j in range(N):

                # extract states
                x_chief_N = x_sol_in[j, :n_states]
                x_dep_O = x_sol_in[j, ind:ind+n_states]

                # convert to eci and store
                x_dep_N, dcm = convert_eci_hill('hill', 'eci', x_chief_N, x_dep_O)
                x_sol_out[j, ind:ind+n_states] = x_dep_N

    # eci to hill
    elif input_flag == 'eci' and output_flag == 'hill':

        # loop over deps
        for i in range(n_deps):
            ind = (i + 1) * n_states

            # loop over time
            for j in range(N):
                # extract states
                x_chief_N = x_sol_in[j, :n_states]
                x_dep_N = x_sol_in[j, ind:ind + n_states]

                # convert to eci and store
                x_dep_O, dcm = convert_eci_hill('eci', 'hill', x_chief_N, x_dep_N)
                x_sol_out[j, ind:ind + n_states] = x_dep_O

    # the same
    elif input_flag == output_flag:
        x_sol_out = x_sol_in

    # error
    else:
        raise ValueError('Incorrect Input or Output Flags')

    return x_sol_out

# # #
# subroutine to compute Gauss's control matrix B(oe)
# Inputs:
#   elems - orbit elements
#   x_N - current sat state
#   t - current time
#   params - standard set of parameters
# Outputs:
#   B - Gauss control matrix
def compute_gauss_matrix(elems, x_N, params):

    # params extract
    R = params['R']
    mu = params['mu']

    # need to convert M to f first
    a_re = elems[0]
    a = a_re * R
    ecc = elems[1]
    inc = elems[2]
    Omega = elems[3]
    omega = elems[4]
    M = elems[5]
    if M < np.pi:
        guess = M + ecc/2.0
    else:
        guess = M - ecc/2.0
    f = convert_M_to_f(guess, 6, ecc)

    # ang momentum
    h = np.linalg.norm(np.cross(x_N[:3], x_N[3:]))
    p = a*(1 - np.power(ecc, 2.0))
    eta = np.sqrt(p / a)
    r = np.power(h, 2.0) / (mu * (1.0 + ecc*np.cos(f)))
    theta = omega + f

    # preallocate
    B = np.zeros((6, 3))

    # assign
    B[0, 0] = 2*np.power(a, 2.0)*ecc*np.sin(f) / (h*R)
    B[1, 0] = p*np.sin(f) / h
    B[4, 0] = -p*np.cos(f) / (h*ecc)
    B[5, 0] = eta*(p*np.cos(f) - 2*r*ecc) / (h*ecc)
    B[0, 1] = 2*np.power(a, 2.0)*p / (h*r*R)
    B[1, 1] = ((p + r)*np.cos(f) + r*ecc) / h
    B[4, 1] = (p + r)*np.sin(f) / (h*ecc)
    B[5, 1] = -eta*(p + r)*np.sin(f) / (h*ecc)
    B[2, 2] = r*np.cos(theta) / h
    B[3, 2] = r*np.sin(theta) / (h*np.sin(inc))
    B[4, 2] = -r*np.sin(theta)*np.cos(inc) / (h*np.sin(inc))

    # return
    return B

# # #
# subroutine to compute Gauss's control matrix B(oe)
# Inputs:
#   elems - orbit elements
#   x_N - current sat state
#   t - current time
#   params - standard set of parameters
# Outputs:
#   B - Gauss control matrix
def compute_oe_control_mats(elems, x_in, params, P_0, N):

    # params extract
    mu = params['mu']
    J2 = params['J2']
    use_J2 = params['use_J2']
    dim_P = params['dim_P']
    R = params['R']

    # need to convert M to f first
    a = elems[0] * R
    ecc = elems[1]
    inc = elems[2]
    Omega = elems[3]
    omega = elems[4]
    M = elems[5]
    f = convert_M_to_f(M, 6, ecc)

    # ang momentum
    h = np.linalg.norm(np.cross(x_in[:3], x_in[3:]))
    p = np.power(h, 2.0) / mu
    eta = np.sqrt(1.0 - np.power(ecc, 2.0))
    r = np.power(h, 2.0) / (mu * (1.0 + ecc*np.cos(f)))
    theta = omega + f
    n = np.sqrt(mu / np.power(a, 3.0))

    # set r_eq to 1 for some reason
    r_eq = R

    # preallocate
    B = np.zeros((6, 3))
    A = np.zeros(6)

    # assign
    B[0, 0] = 2*np.power(a, 2.0)*ecc*np.sin(f) / (h*r_eq)
    B[1, 0] = p*np.sin(f) / h
    B[4, 0] = -p*np.cos(f) / (h*ecc)
    B[5, 0] = eta*(p*np.cos(f) - 2*r*ecc) / (h*ecc)
    B[0, 1] = 2*np.power(a, 2.0)*p / (h*r*r_eq)
    B[1, 1] = ((p + r)*np.cos(f) + r*ecc) / h
    B[4, 1] = (p + r)*np.sin(f) / (h*ecc)
    B[5, 1] = -eta*(p + r)*np.sin(f) / (h*ecc)
    B[2, 2] = r*np.cos(theta) / h
    B[3, 2] = r*np.sin(theta) / (h*np.sin(inc))
    B[4, 2] = -r*np.sin(theta)*np.cos(inc) / (h*np.sin(inc))

    # compute A
    if use_J2:
        A[3] = -1.5*J2*np.power((r_eq / p), 2.0)*n*np.cos(inc)
        A[4] = 0.75*J2*np.power((r_eq / p), 2.0)*n*(5.0*np.power(np.cos(inc), 2.0) - 1)
        A[5] = n + 0.75*J2*np.power((r_eq / p), 2.0)*n*eta*(3.0*np.power(np.cos(inc), 2.0) - 1)

    # compute P
    P_a1 = 0.024
    P_e1 = 0.020
    P_i1 = 0.005
    P_O1 = 0.005
    P_w1 = 0.040
    P_M1 = 0.010

    if dim_P == 1:
        P = P_0[0, 0]
    elif dim_P == 3:
        P = P_0
    elif dim_P == 6:
        P = P_0 + np.diag(np.array([P_a1*np.power(np.cos(f/2.0), N), P_e1*np.power(np.cos(f), N),
                                    P_i1*np.power(np.cos(theta), 2.0), P_O1*np.power(np.sin(theta), N),
                                    P_w1*np.power(np.sin(f), N), P_M1*np.power(np.sin(f), N)]))
    else:
        raise ValueError('Incorrect dimension for gain P')

    # return
    return B, A, P
