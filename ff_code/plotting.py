# imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# font definition
font = {'sans-serif' : 'Helvetica',
        'weight' : 'bold',
        'size'   : 12}

plt.rcParams['axes.labelweight'] = 'bold'

# # #
# plot_error_norm
# plot the error norm
# Inputs:
#   tspan - time vector
#   delta_r - position error
#   delta_rdot - velocity error
#   T - orbit period
def plot_error_norm(tspan, delta_r, delta_rdot, T):

    plt.figure(figsize=(9, 6))
    plt.rc('font', **font)

    # plot position error
    plt.subplot(211)
    plt.xlabel('Orbit Fraction')
    plt.ylabel('delta r [km]')
    plt.plot(tspan / T, np.linalg.norm(delta_r, axis=1))
    plt.yscale('log')
    plt.grid(True)

    # plot velocity error
    plt.subplot(212)
    plt.xlabel('Orbit Fraction')
    plt.ylabel('delta rdot [km/s]')
    plt.plot(tspan / T, np.linalg.norm(delta_rdot, axis=1))
    plt.yscale('log')
    plt.grid(True)

    # show
    plt.show()

# # #
# plot_control_norm
# plot the control norm
# Inputs:
#   tspan - time vector
#   u - control
#   T - orbit period
def plot_control_norm(tspan, u, T):

    plt.figure(figsize=(9, 6))
    plt.rc('font', **font)

    # plot control
    plt.xlabel('Orbit Fraction')
    plt.ylabel('Control, u [km/s^2]')
    plt.plot(tspan / T, np.linalg.norm(u, axis=1))
    plt.yscale('log')
    plt.grid(True)

    # show
    plt.show()

# # #
# plot_error_components
# plot the error components vs time
# Inputs:
#   tspan - time vector
#   delta_r - position error
#   delta_rdot - velocity error
#   T - orbit period
def plot_error_components(tspan, delta_r, delta_rdot, T):

    plt.figure(figsize=(9, 6))
    plt.rc('font', **font)

    # plot position error
    plt.subplot(211)
    plt.xlabel('Orbit Fraction')
    plt.ylabel('delta r [km]')
    a = plt.plot(tspan / T, delta_r[:, 0])
    b = plt.plot(tspan / T, delta_r[:, 1])
    c = plt.plot(tspan / T, delta_r[:, 2])
    plt.grid(True)
    plt.legend((a[0], b[0], c[0]), ('dx', 'dy', 'dz'), loc=0)

    # plot velocity error
    plt.subplot(212)
    plt.xlabel('Orbit Fraction')
    plt.ylabel('delta rdot [km/s]')
    a = plt.plot(tspan / T, delta_rdot[:, 0])
    b = plt.plot(tspan / T, delta_rdot[:, 1])
    c = plt.plot(tspan / T, delta_rdot[:, 2])
    plt.grid(True)
    plt.legend((a[0], b[0], c[0]), ('dvx', 'dvy', 'dvz'), loc=0)

    # show
    plt.show()

# # #
# plot_orbit_hill
# plot the deputy orbit in the hill frame
# Inputs:
#   tspan - time vector
#   delta_r - position error
#   delta_rdot - velocity error
#   T - orbit period
def plot_orbit_hill(x_dep_O):

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    plt.rc('font', **font)

    # plot orbit in hill frame
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    ax.plot(x_dep_O[:, 0], x_dep_O[:, 1], x_dep_O[:, 2], linewidth=2.0)
    ax.scatter(0.0, 0.0, 0.0, s=50, color='green')
    #ax.set_xlim([-3.0, 3.0])
    #ax.set_ylim([-3.0, 3.0])
    ax.grid(True)
    plt.legend(loc=0)

    # show
    plt.show()

# # #
# plot_delta_elems
# plot the oe errors versus time
# Inputs:
#   tspan - time vector
#   delta_oe - mean element error vector
#   T - orbit period
def plot_delta_elems(tspan, delta_oe, T):

    plt.figure(figsize=(9, 6))
    plt.rc('font', **font)

    # plot semi-major axis over R error
    plt.subplot(231)
    plt.xlabel('Orbit Fraction')
    plt.ylabel('Normalized SMA Error')
    plt.plot(tspan / T, delta_oe[:, 0])
    plt.grid(True)

    # plot eccentricity error
    plt.subplot(232)
    plt.xlabel('Orbit Fraction')
    plt.ylabel('Eccentricity Error')
    plt.plot(tspan / T, delta_oe[:, 1])
    plt.grid(True)

    # plot inclination error
    plt.subplot(233)
    plt.xlabel('Orbit Fraction')
    plt.ylabel('Inlincation Error [rad]')
    plt.plot(tspan / T, delta_oe[:, 2])
    plt.grid(True)

    # plot RAAN error
    plt.subplot(234)
    plt.xlabel('Orbit Fraction')
    plt.ylabel('RAAN Error [rad]')
    plt.plot(tspan / T, delta_oe[:, 3])
    plt.grid(True)

    # plot argument of periapse error
    plt.subplot(235)
    plt.xlabel('Orbit Fraction')
    plt.ylabel('Arg Periapse Error [rad]')
    plt.plot(tspan / T, delta_oe[:, 4])
    plt.grid(True)

    # plot argument of periapse error
    plt.subplot(236)
    plt.xlabel('Orbit Fraction')
    plt.ylabel('Mean Anomaly [rad]')
    plt.plot(tspan / T, delta_oe[:, 5])
    plt.grid(True)

    # show
    plt.show()
