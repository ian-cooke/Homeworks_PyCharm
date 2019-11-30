from ff_code import macros, functions, plotting
import numpy as np

# constants
n_deps = 1
n_states = 6
mu = 398600.4415  # [km^3/s^2]
J2 = 1.082626925638815e-03
R = 6378.1363  # [km]
use_J2 = False
params = {'mu': mu,
          'n_deps': n_deps,
          'n_states': n_states,
          'J2': J2,
          'R': R,
          'use_J2': use_J2}

# Chief Elements and state
a = 7000.0  # km
e = 0.0
i = 0.0
Omega = 0.0
omega = 0.0
M_0 = 0.0
elems_chief = np.array([a, e, i, Omega, omega, M_0])
x_chief_N_0 = macros.convert_rv_kep('keplerian', elems_chief, 0.0, 'cartesian', normalize=False)

# desired deputy elems and state
rho_O_0_d = np.array([0.0, 1.0, 0.0])
rhoprime_O_0_d = np.array([0.0, 0.0, 0.0])
x_dep_d_O_0 = np.concatenate((rho_O_0_d, rhoprime_O_0_d))
print('Desired Initial Hill Frame Position/Velocity: ' + str(rho_O_0_d) + ' km, ' + str(rhoprime_O_0_d) + ' km/s')
x_dep_d_N_0, dcm = macros.convert_eci_hill('hill', 'eci', x_chief_N_0, x_dep_d_O_0)

# tspan
n_orbits = 3
dt = 1.0
T = 2*np.pi*np.sqrt(np.power(a, 3.0) / mu)
tspan = np.arange(0.0, n_orbits * T + dt, dt)

# do control
K_p = 0.0001
K_d = 0.06
params['K_p'] = K_p
params['K_d'] = K_d
# initial deputy perturbation
rho_O_0_perturb = np.array([-100.0, 100.0, 100.0])  # [km]
rhoprime_O_0_perturb = np.array([0.1, -0.1, 0.0])  # [km]
print('Initial deputy perturbation ' + str(rho_O_0_perturb) + ' km, ' + str(rhoprime_O_0_perturb) + ' km/s')
x_dep_O_0 = np.concatenate((rho_O_0_perturb, rhoprime_O_0_perturb), axis=0)
x_dep_N_0, dcm = macros.convert_eci_hill('hill', 'eci', x_chief_N_0, x_dep_O_0)
# simulate
constant_sep = True
params['x_dep_d_N_sep'] = x_dep_d_O_0
x_chief_N, x_dep_N, x_dep_d_N, u_N = functions.nonlinear_control_sim(x_chief_N_0, x_dep_N_0, x_dep_d_N_0, tspan, constant_sep, params)

# plotting
errors = x_dep_N - x_dep_d_N
delta_r = errors[:, 0:3]
delta_rdot = errors[:, 3:6]

plotting.plot_error_norm(tspan, delta_r, delta_rdot, T)
plotting.plot_control_norm(tspan, u_N, T)
plotting.plot_error_components(tspan, delta_r, delta_rdot, T)
x_sol_O = macros.convert_ODEsol('eci', 'hill', np.concatenate((x_chief_N, x_dep_N), axis=1), params)
x_dep_O = x_sol_O[:, 6:]
plotting.plot_orbit_hill(x_dep_O)