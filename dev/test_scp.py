"""Test indirect OCP in CR3BP dynamics"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
import pygmo as pg

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import stardust


def test_twostage_cr3bp():
    # physical parameters
    LU = 389703     # km
    TU = 382981     # sec
    MU = 1000.0     # kg

    mu = 1.215058560962404e-02
    mu1 = 1 - mu
    mu2 = mu

    # initial state, propagation time
    rv0 = np.array([1.0809931218390707E+00,
          0.0000000000000000E+00,
          -2.0235953267405354E-01,
          1.0157158264396639E-14,
          -1.9895001215078018E-01,
          7.2218178975912707E-15])
    period = 2.3538670417546639E+00
    sol0_ballistic = solve_ivp(stardust.eom_rotating_cr3bp, (0, period), rv0, args=(mu, mu1, mu2), 
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
    tspan = [0, 1.2*period]
    
    # load solution
    t_nodes = np.loadtxt(os.path.join(os.path.dirname(__file__), 'test_t_nodes.txt'))
    nodes = np.loadtxt(os.path.join(os.path.dirname(__file__), 'test_nodes.txt'))
    ubars = np.loadtxt(os.path.join(os.path.dirname(__file__), 'test_ubars.txt'))

    N = len(t_nodes)
    prob = stardust.FixedTimeSCP(
        stardust.eom_stm_rotating_cr3bp,
        rv0,
        rvf,
        t_nodes,
        nodes,
        ubars,
        args = args,
    )

    # solve SOCP
    trust_region = [1e-3]*3 + [1e-3]*3
    success, xs_opt, us_opt = prob.solve_socp(trust_region, verbose_solver=True)
    print(f"xs_opt = {xs_opt}")
    print(f"us_opt = {us_opt}")

    #ubars_new = ubars + us_opt
    sols = prob.propagate_nonlinear(us_opt)

    # # test outer loop
    # print(f"Testing outer loop...")
    # tstart = time.time()
    # exitflag, iter_sols = prob.solve(maxiter = 10, save_all_iter_sols = True, verbose_inner = True)
    # tend = time.time()
    # print(f"Elapsed time = {tend - tstart} sec")
    # assert exitflag == 1

    # plot trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for _sol in sols:
        ax.plot(_sol.y[0,:], _sol.y[1,:], _sol.y[2,:], color='black', lw=0.5)

    ax.plot(sol0_ballistic.y[0,:], sol0_ballistic.y[1,:], sol0_ballistic.y[2,:], color='blue')
    ax.plot(solf_ballistic.y[0,:], solf_ballistic.y[1,:], solf_ballistic.y[2,:], color='green')
    stardust.plot_sphere_wireframe(ax, 1737/384400, [1-mu,0,0], color='grey')
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.set_aspect('equal', 'box')
    # fig.savefig(os.path.join(os.path.dirname(__file__), 'twostage_cr3bp_example.png'), dpi=300)
    return


if __name__=="__main__":
    test_twostage_cr3bp()
    print("Done!")
    plt.show()
    