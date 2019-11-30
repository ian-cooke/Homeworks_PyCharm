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
          'n_deps'  : n_deps,
          'n_states': n_states,
          'J2'      : J2,
          'R'       : R,
          'use_J2'  : use_J2,
          'dim_P'   : 6}


# Chief Elements and state
a = 7555.0  # km
e = 0.05
i = 48.0
Omega = 0.0
omega = 10.0
M_0 = 120.0
elems_chief = np.array([a, e, i, Omega, omega, M_0])
x_chief_N_0 = macros.convert_rv_kep('keplerian', elems_chief, 0.0, 'cartesian')

# desired deputy elems and state
delta_elems_d = 100 * np.array([-0.00192995, 0.000576727, 0.006, 0.0, 0.0, 0.0])
elems_dep_d = elems_chief + delta_elems_d
print('Desired Element Difference: ' + str(delta_elems_d))
x_dep_d_N_0 = macros.convert_rv_kep('keplerian', elems_dep_d, 0.0, 'cartesian')

# tspan
n_orbits = 3.0
dt = 1.0
T = 2*np.pi*np.sqrt(np.power(a, 3.0) / mu)
tspan = np.arange(0.0, n_orbits * T + dt, dt)

# initial deputy perturbation
delta_elems = -delta_elems_d
elems_dep = elems_chief + delta_elems
print('Initial deputy element perturbation: ' + str(delta_elems))
x_dep_N_0 = macros.convert_rv_kep('keplerian', elems_chief, 0.0, 'cartesian')
# simulate
elems_dep_d = macros.convert_rv_kep('cartesian', x_dep_d_N_0, 0.0, 'keplerian', radians=True)
x_chief_N, x_dep_N, x_dep_d_N, delta_oe, u_N = functions.oe_control_sim(x_chief_N_0, x_dep_N_0, x_dep_d_N_0, tspan, params)

# plot results
plotting.plot_control_norm(tspan, u_N, T)
errors = x_dep_N - x_dep_d_N
delta_r = errors[:, :3]
delta_rdot = errors[:, 3:]
plotting.plot_error_norm(tspan, delta_r, delta_rdot, T)
plotting.plot_delta_elems(tspan, delta_oe, T)
x_sol_O = macros.convert_ODEsol('eci', 'hill', np.concatenate((x_chief_N, x_dep_N), axis=1), params)
x_dep_O = x_sol_O[:, 6:]
plotting.plot_orbit_hill(x_dep_O)
