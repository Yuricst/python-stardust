"""Equations of motion for CR3BP in rotating frame"""

import numpy as np
from numba import njit, float64


@njit
def gravity_gradient_cr3bp(rvec, mu, mu1, mu2):
    """Compute gravity gradient matrix for CR3BP in the rotating frame.

    Args:
        mu (float): CR3BP parameter, i.e. scaled mass of secondary body
        mu1 (float): scaled mass of first body to be used
        mu2 (float): scaled mass of second body to be used
        rvec (np.array): position vector of spacecraft

    Returns:
        np.array: 3-by-3 gravity gradient matrix
    """
    # unpack rvec
    x,y,z = rvec
    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x-(1-mu))**2 + y**2 + z**2)

    # define entries of the gravity gradient matrix
    r1_5 = r1**5
    r2_5 = r2**5
    G00 = 3*mu1*(mu+x)**2/r1_5- mu1/r1**3 + 3*mu2*(x-1+mu)**2/r2_5 - mu2/r2**3 + 1
    G01 = 3*mu1*y*(mu+x)/r1_5+ 3*mu2*y*(x-1+mu)/r2_5
    G02 = 3*mu1*z*(mu+x)/r1_5+ 3*mu2*z*(x-1+mu)/r2_5

    G10 = 3*mu1*y*(mu+x)/r1_5+ 3*mu2*y*(x-1+mu)/r2_5
    G11 = 3*mu1*y**2/r1_5+ 3*mu2*y**2/r2_5 - mu1/r1**3 - mu2/r2**3 + 1
    G12 = 3*mu1*y*z/r1_5+ 3*mu2*y*z/r2_5

    G20 = 3*mu1*z*(mu+x)/r1_5+ 3*mu2*z*(x-1+mu)/r2_5
    G21 = 3*mu1*y*z/r1_5+ 3*mu2*y*z/r2_5
    G22 = 3*mu1*z**2/r1_5+ 3*mu2*z**2/r2_5 - mu1/r1**3 - mu2/r2**3
    return np.array([[G00, G01, G02], [G10, G11, G12], [G20, G21, G22]])


@njit(float64[:](float64, float64[:], float64, float64, float64), cache=True)
def eom_rotating_cr3bp(t, state, mu, mu1, mu2):
    """Equation of motion in CR3BP in the rotating frame.
    This function is written for `scipy.integrate.solve=ivp()` and is compatible with njit.

    Args:
        t (float): time
        state (np.array): 1D array of Cartesian state, length 6
        mu (float): CR3BP parameter, i.e. scaled mass of secondary body
        mu1 (float): scaled mass of first body to be used
        mu2 (float): scaled mass of second body to be used

    Returns:
        (np.array): 1D array of derivative of Cartesian state
    """
    # unpack positions
    x = state[0]
    y = state[1]
    z = state[2]
    # unpack velocities
    vx = state[3]
    vy = state[4]
    vz = state[5]
    # compute radii to each primary
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)
    # setup vector for dX/dt
    deriv = np.zeros((6,))
    # position derivatives
    deriv[0] = vx
    deriv[1] = vy
    deriv[2] = vz
    # velocity derivatives
    deriv[3] =  2*vy + x - (mu1/r1**3) * (mu + x) - (mu2/r2**3) * (x - 1 + mu)
    deriv[4] = -2*vx + y - (mu1/r1**3) * y        - (mu2/r2**3) * y
    deriv[5] =           - (mu1/r1**3) * z        - (mu2/r2**3) * z
    return deriv


#@njit(float64[:](float64, float64[:], float64, float64), cache=True)
def eom_stm_rotating_cr3bp(t, state, mu, mu1, mu2):
    """Equation of motion in CR3BP in the rotating frame with STM.
    This function is written for `scipy.integrate.solve=ivp()` and is compatible with njit.
    """
    # state derivative
    deriv = np.zeros((42,), order='C')
    deriv[0:6] = eom_rotating_cr3bp(t, state[0:6], mu, mu1, mu2)
    
    # derivative of STM
    A = np.zeros((6,6), order='C')
    A[0:3,3:6] = np.eye(3)
    A[3,4] = 2
    A[4,3] = -2
    A[3:6,0:3] = gravity_gradient_cr3bp(state[0:3], mu, mu1, mu2)
    deriv[6:] = np.dot(A, state[6:].reshape(6,6)).reshape(36,)
    return deriv