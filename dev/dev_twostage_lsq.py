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

    case_option = 2

    if case_option == 0:
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
        initial_nodes_strategy = 'random_path'
        tspan = [0, 12.5 * 86400/TU]
        N = 20
        weights = [1,1,1] * N
        
    elif case_option == 1:
        # initial state
        rv0 = np.array([8.3203900980615830E-1,
                        0.0,
                        1.2603694134707036E-1,
                        0.0,
                        2.4012675449219933E-1,
                        0.0])
        period_0 = 2.7824079054366879
        sol0_ballistic = solve_ivp(stardust.eom_rotating_cr3bp, (0, period_0), rv0, args=(mu, mu1, mu2), 
                                method='RK45', rtol=1e-12, atol=1e-12)
        # sol0_shift = solve_ivp(stardust.eom_rotating_cr3bp, (0, period_0*0.5), rv0, args=(mu, mu1, mu2), 
        #                        method='RK45', rtol=1e-12, atol=1e-12)
        # rv0 = sol0_shift.y[:,-1]

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
        initial_nodes_strategy = 'linear'
        tspan = [0, 2.5 * 86400/TU]
        N = 30
        w_impulse = 1e-3
        weights = [w_impulse,w_impulse,w_impulse] + [1,1,1] * (N-2) + [w_impulse,w_impulse,w_impulse]
        
    elif case_option == 2:
        # initial state
        rv0 = np.array([1.0311462015039496,
                        0.0,
                        2.1429500507420918E-1,
                        0.0,
                        -2.9154668219324587E-1,
                        0.0])
        period_0 = 8.2030392340209257E+0
        sol0_ballistic = solve_ivp(stardust.eom_rotating_cr3bp, (0, period_0), rv0, args=(mu, mu1, mu2), 
                                method='RK45', rtol=1e-12, atol=1e-12)
        # sol0_shift = solve_ivp(stardust.eom_rotating_cr3bp, (0, period_0*0.1), rv0, args=(mu, mu1, mu2), 
        #                        method='RK45', rtol=1e-12, atol=1e-12)
        # rv0 = sol0_shift.y[:,-1]

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
        initial_nodes_strategy = 'linear'
        tspan = [0, 30 * 86400/TU]
        N = 60
        weights = [1,1,1] * N
        # w_impulse = 1e-3
        # weights = [w_impulse,w_impulse,w_impulse] + [1,1,1] * (N-2) + [w_impulse,w_impulse,w_impulse]

    # construct problem
    args = (mu, mu1, mu2)
    print(f"tspan = {tspan}")

    print(f"\nTesting outer loop in sparse Jacobian mode...")
    np.random.seed(0)       # set flag since we are using random initial guess
    prob = stardust.FixedTimeTwoStageLeastSquares(     # need to re-initialize
        stardust.eom_stm_rotating_cr3bp,
        rv0,
        rvf,
        tspan,
        N = N,
        args = args,
        initial_nodes_strategy = initial_nodes_strategy,
    )
    nprocs = 6
    tstart = time.time()
    exitflag, iter_sols = prob.solve(maxiter = 20,
                                     save_all_iter_sols = True, 
                                     verbose_inner = True,
                                     sparse_approx_jacobian = True,
                                     weights = weights,
                                     nprocs = nprocs)
    tend = time.time()
    print(f"Elapsed time = {tend - tstart} sec\n")
    print(f"exitflag = {exitflag}")

    # Plot Jacobian sparsity
    fig, ax = plt.subplots()
    ax.spy(prob.J_outer)
    ax.set_title('Jacobian Sparsity Pattern')

    # # export nodes to file
    # np.savetxt(os.path.join(os.path.dirname(__file__), 'test_t_nodes.txt'), prob.times)
    # np.savetxt(os.path.join(os.path.dirname(__file__), 'test_nodes.txt'), prob.nodes)
    # np.savetxt(os.path.join(os.path.dirname(__file__), 'test_ubars.txt'), prob.v_residuals)

    # plot trajectory
    fig, ax, sols_check = prob.plot_trajectory(use_itm_nodes=False, show_maneuvers=True,
                                               sphere_radius = 1737/384400, sphere_center=[1-mu,0,0])
    pos_error = np.linalg.norm(sols_check[-1].y[0:3,-1] - rvf[0:3])
    vel_error = np.linalg.norm(sols_check[-1].y[3:6,-1] + prob.v_residuals[-1] - rvf[3:])
    print(f"Final position error = {pos_error}")
    print(f"Final velocity error = {vel_error}")
    assert pos_error < 1e-11
    assert vel_error < 1e-11
    
    # in-between guesses
    for _sols in iter_sols:
        for _sol in _sols:
            ax.plot(_sol.y[0,:], _sol.y[1,:], _sol.y[2,:], color='black', lw=0.5)
    ax.plot(sol0_ballistic.y[0,:], sol0_ballistic.y[1,:], sol0_ballistic.y[2,:], color='blue')
    ax.plot(solf_ballistic.y[0,:], solf_ballistic.y[1,:], solf_ballistic.y[2,:], color='green')
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.set_aspect('equal', 'box')
    # fig.savefig(os.path.join(os.path.dirname(__file__), 'twostage_cr3bp_example.png'), dpi=300)

    # plot control
    fig_u, ax_u = prob.plot_deltaV()
    return


if __name__=="__main__":
    test_twostage_outerloop()
    print("Done!")
    plt.show()
    