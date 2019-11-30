import numpy as np
from HW1 import convert_rv_kep

delta_t = 3600.0
# case 1
a = 8000.0
e = 0.1
inc = 30.0
Omega = 145.0
omega = 120.0
M_0 = 10.0

elems = np.array([a, e, inc, Omega, omega, M_0])

state_vec = convert_rv_kep.convert_rv_kep('keplerian', elems, delta_t, 'state')
print('Convert from first set of elements given to state vector:')
print('r_x = {0} km'.format(state_vec[0]))
print('r_y = {0} km'.format(state_vec[1]))
print('r_z = {0} km'.format(state_vec[2]))
print('v_x = {0} km/s'.format(state_vec[3]))
print('v_y = {0} km/s'.format(state_vec[4]))
print('v_z = {0} km/s'.format(state_vec[5]))
print('')

# case 2
delta_t = 3600.0
# case 1
a = -8000.0
e = 1.1
inc = 30.0
Omega = 145.0
omega = 120.0
M_0 = 10.0

elems = np.array([a, e, inc, Omega, omega, M_0])

state_vec = convert_rv_kep.convert_rv_kep('keplerian', elems, delta_t, 'state')
print('Convert from second set of elements given to state vector:')
print('r_x = {0} km'.format(state_vec[0]))
print('r_y = {0} km'.format(state_vec[1]))
print('r_z = {0} km'.format(state_vec[2]))
print('v_x = {0} km/s'.format(state_vec[3]))
print('v_y = {0} km/s'.format(state_vec[4]))
print('v_z = {0} km/s'.format(state_vec[5]))
print('')

# case 3
state = np.array([-1264.61, 8013.81, -3371.25, -6.03962, -0.204398, 2.09672])

elems_out = convert_rv_kep.convert_rv_kep('state', state, delta_t, 'keplerian')
print('Convert from state vector back to first set of elements')
print('a = {0} km'.format(elems_out[0]))
print('e = {0}'.format(elems_out[1]))
print('i = {0} deg'.format(elems_out[2]))
print('Omega = {0} deg'.format(elems_out[3]))
print('omega = {0} deg'.format(elems_out[4]))
print('M_0 = {0} deg'.format(elems_out[5]))
print('')


# case 4
state = np.array([18877, 27406.6, -19212.8, 3.55968, 6.35532, -4.18447])

elems_out = convert_rv_kep.convert_rv_kep('state', state, delta_t, 'keplerian')
print('Convert from state vector back to second set of elements')
print('a = {0}'.format(elems_out[0]))
print('e = {0}'.format(elems_out[1]))
print('i = {0}'.format(elems_out[2]))
print('Omega = {0}'.format(elems_out[3]))
print('omega = {0}'.format(elems_out[4]))
print('M_0 = {0}'.format(elems_out[5]))