"""Test indirect OCP in CR3BP dynamics"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import stardust


def test_twostage_outerloop():
    # physical parameters
    LU = 389703     # km
    TU = 382981     # sec
    MU = 1000.0     # kg

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

    # final targeted state
    rvf = np.array([1.1648780946517576,
                    0.0,
                    -1.1145303634437023E-1,
                    0.0,
                    -2.0191923237095796E-1,
                    0.0])
    period_f = 3.3031221822879884
    solf_ballistic = solve_ivp(stardust.eom_rotating_cr3bp, (0, period_f), rvf, args=(mu, mu1, mu2),
                               method='RK45', rtol=1e-12, atol=1e-12)

    # construct problem
    args = (mu, mu1, mu2)
    tspan = [0, 2*period_0]
    N = 20

    # test outer loop
    print(f"\nTesting outer loop in dense Jacobian mode...")
    np.random.seed(0)       # set flag since we are using random initial guess
    prob = stardust.FixedTimeTwoStageLeastSquares(
        stardust.eom_stm_rotating_cr3bp,
        rv0,
        rvf,
        tspan,
        N = N,
        args = args,
        initial_nodes_strategy = 'random_path'
    )

    # run Jacobian function
    tstart = time.time()
    prob._outer_loop_jacobian_sparse()
    tend = time.time()
    print(f"Elapsed time = {tend - tstart} sec\n")
    jac_serial = copy.deepcopy(prob.J_outer)

    # run Jacobian function in parallel
    nprocs = 4
    tstart = time.time()
    J_entry_list = prob._outer_loop_jacobian_sparse_parallel(nprocs = nprocs)
    tend = time.time()
    print(f"Elapsed time = {tend - tstart} sec ({nprocs} procs)\n")
    jac_parallel = copy.deepcopy(prob.J_outer)

    # compare serial and parallel Jacobians
    diff = np.linalg.norm(jac_serial - jac_parallel)
    print(f"diff = {diff}")
    print(f"jac_serial[0:4,0:4] = \n{jac_serial[0:4,0:4]}")
    print(f"jac_parallel[0:4,0:4] = \n{jac_parallel[0:4,0:4]}")
    print(f"Difference[0:4,0:4] = \n{jac_serial[0:4,0:4] - jac_parallel[0:4,0:4]}")
    return


if __name__=="__main__":
    test_twostage_outerloop()
    print("Done!")
    plt.show()
    