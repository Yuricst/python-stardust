"""Validate dynamics"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import stardust


def test_cr3bp():
    # physical parameters
    mu = 1.215058560962404e-02
    mu1 = 1 - mu
    mu2 = mu

    # initial state
    rv0 = np.array([1.0809931218390707E+00,
          0.0000000000000000E+00,
          -2.0235953267405354E-01,
          1.0157158264396639E-14,
          -1.9895001215078018E-01,
          7.2218178975912707E-15])
    period_0 = 2.3538670417546639E+00
    sol0_ballistic = solve_ivp(stardust.eom_rotating_cr3bp, (0, period_0), rv0, args=(mu, mu1, mu2), 
                               method='RK45', rtol=1e-12, atol=1e-12)
    assert sol0_ballistic.success, "Integration failed"
    assert np.linalg.norm(sol0_ballistic.y[:,-1] - rv0) < 1e-11,\
        f"|| sol0_ballistic.y[:,-1] - rv0 || = {np.linalg.norm(sol0_ballistic.y[:,-1] - rv0)}"
    return


def test_cr3bp_stm():
    # physical parameters
    mu = 1.215058560962404e-02
    mu1 = 1 - mu
    mu2 = mu

    # initial state
    rv0 = np.array([1.0809931218390707E+00,
          0.0000000000000000E+00,
          -2.0235953267405354E-01,
          1.0157158264396639E-14,
          -1.9895001215078018E-01,
          7.2218178975912707E-15])
    period_0 = 2.3538670417546639E+00
    rv0stm = np.concatenate((rv0, np.eye(6).flatten()))
    sol0_ballistic = solve_ivp(stardust.eom_stm_rotating_cr3bp, (0, period_0), rv0stm, args=(mu, mu1, mu2), 
                               method='RK45', rtol=1e-12, atol=1e-12)
    assert sol0_ballistic.success, "Integration failed"
    assert np.linalg.norm(sol0_ballistic.y[0:6,-1] - rv0) < 1e-11,\
        f"|| sol0_ballistic.y[:,-1] - rv0 || = {np.linalg.norm(sol0_ballistic.y[:,-1] - rv0)}"
    return


if __name__ == "__main__":
    test_cr3bp()
    test_cr3bp_stm()
    print("Done!")