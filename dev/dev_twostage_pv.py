"""Test indirect OCP in CR3BP dynamics"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from scipy.integrate import solve_ivp
from scipy.optimize import show_options

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
        tspan = [0, 28 * 86400/TU]
        N = 30
        weights = [1,1,1] * N
        # w_impulse = 1e-3
        # weights = [w_impulse,w_impulse,w_impulse] + [1,1,1] * (N-2) + [w_impulse,w_impulse,w_impulse]

    # construct problem
    args = (mu, mu1, mu2)
    print(f"tspan = {tspan}")

    print(f"\nTesting outer loop in sparse Jacobian mode...")
    np.random.seed(0)       # set flag since we are using random initial guess
    prob = stardust.FixedTimeTwoStagePrimerVector(     # need to re-initialize
        stardust.eom_stm_rotating_cr3bp,
        rv0,
        rvf,
        tspan,
        N = N,
        args = args,
        initial_nodes_strategy = initial_nodes_strategy,
    )
    # load pre-computed solution
    prob.times = np.loadtxt(os.path.join(os.path.dirname(__file__), f'test_t_nodes_case{case_option}.txt'))
    prob.nodes = np.loadtxt(os.path.join(os.path.dirname(__file__), f'test_nodes_case{case_option}.txt'))
    
    # # plot trajectory from loaded solution
    # fig, ax_3d, sols_check = prob.plot_trajectory(use_itm_nodes=False, show_maneuvers=True,
    #                                            sphere_radius = 1737/384400, sphere_center=[1-mu,0,0])


    # solve problem
    res = prob.solve_scipy(method = 'CG', maxiter = 500, verbose = True)

    # tstart = time.time()
    # exitflag, iter_sols = prob.solve(maxiter = 2,
    #                                  save_all_iter_sols = True, 
    #                                  verbose_inner = True)
    # tend = time.time()
    # print(f"Elapsed time = {tend - tstart} sec\n")
    # print(f"exitflag = {exitflag}")

    # print(f"Gradient after solve: \n{prob.gradient}")

    # plot trajectory
    fig, ax_3d, sols_check = prob.plot_trajectory(use_itm_nodes=False, show_maneuvers=True,
                                               sphere_radius = 1737/384400, sphere_center=[1-mu,0,0])
    # prob.plot_trajectory(use_itm_nodes=False, show_maneuvers=True, ax = ax_3d)
    pos_error = np.linalg.norm(sols_check[-1].y[0:3,-1] - rvf[0:3])
    vel_error = np.linalg.norm(sols_check[-1].y[3:6,-1] + prob.v_residuals[-1] - rvf[3:])
    ax_3d.plot(sol0_ballistic.y[0,:], sol0_ballistic.y[1,:], sol0_ballistic.y[2,:], color='blue')
    ax_3d.plot(solf_ballistic.y[0,:], solf_ballistic.y[1,:], solf_ballistic.y[2,:], color='green')
    ax_3d.set(xlabel="x", ylabel="y", zlabel="z")
    ax_3d.set_aspect('equal', 'box')
    # fig.savefig(os.path.join(os.path.dirname(__file__), 'twostage_cr3bp_example.png'), dpi=300)

    # plot control
    fig_u, ax_u = prob.plot_deltaV()

    # compute primer vector
    pi_times, pi_histories, fig, axs = prob.plot_primer_vector()
    return


if __name__=="__main__":
    test_twostage_outerloop()
    print("Done!")
    plt.show()
    