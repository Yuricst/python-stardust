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


def test_twostage_innerloop():
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
    tspan = [0, 1.2*period_0]
    N = 12
    prob = stardust.FixedTimeTwoStageLeastSquares(
        stardust.eom_stm_rotating_cr3bp,
        rv0,
        rvf,
        tspan,
        N = N,
        args = args,
    )
    prob.create_nodes()

    # test inner loop
    print(f"Testing inner loop...")
    sols = prob.inner_loop(prob.nodes[1:-1,0:3].flatten(), get_sols = True, maxiter = 10, verbose = True)
    # assert prob.inner_loop_success == True

    fig, ax, _ = prob.plot_trajectory(show_maneuvers=False)
    ax.plot(sol0_ballistic.y[0,:], sol0_ballistic.y[1,:], sol0_ballistic.y[2,:], color='blue')
    ax.plot(solf_ballistic.y[0,:], solf_ballistic.y[1,:], solf_ballistic.y[2,:], color='green')
    stardust.plot_sphere_wireframe(ax, 1737/384400, [1-mu,0,0], color='grey')
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Testing inner loop success: {prob.inner_loop_success}")
    return


def test_twostage_innerloop_remove_node():
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
    _solf_half = solve_ivp(stardust.eom_rotating_cr3bp, (0, period_f/2), rvf, args=(mu, mu1, mu2),
                                 method='RK45', rtol=1e-12, atol=1e-12)
    rvf = _solf_half.y[:,-1]

    # construct problem
    args = (mu, mu1, mu2)
    tspan = [0, 1.2*period_0]
    N = 12
    prob = stardust.FixedTimeTwoStageLeastSquares(
        stardust.eom_stm_rotating_cr3bp,
        rv0,
        rvf,
        tspan,
        N = N,
        args = args,
    )
    prob.create_nodes(
        strategy = 'linear',
    )

    # test inner loop
    print(f"Testing inner loop...")
    sols = prob.inner_loop(prob.nodes[1:-1,0:3].flatten(), get_sols = True, maxiter = 10, verbose = True)
    # assert prob.inner_loop_success == True

    fig, ax, _ = prob.plot_trajectory(show_maneuvers=False)
    ax.plot(sol0_ballistic.y[0,:], sol0_ballistic.y[1,:], sol0_ballistic.y[2,:], color='blue')
    ax.plot(solf_ballistic.y[0,:], solf_ballistic.y[1,:], solf_ballistic.y[2,:], color='green')
    stardust.plot_sphere_wireframe(ax, 1737/384400, [1-mu,0,0], color='grey')
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Testing inner loop success: {prob.inner_loop_success}")

    # remove node
    prob.remove_node(5)

    # test inner loop again
    print(f"Testing inner loop with 1 node removed...")
    sols = prob.inner_loop(prob.nodes[1:-1,0:3].flatten(), get_sols = True, maxiter = 10, verbose = True)
    # assert prob.inner_loop_success == True

    fig, ax, _ = prob.plot_trajectory(show_maneuvers=False)
    ax.plot(sol0_ballistic.y[0,:], sol0_ballistic.y[1,:], sol0_ballistic.y[2,:], color='blue')
    ax.plot(solf_ballistic.y[0,:], solf_ballistic.y[1,:], solf_ballistic.y[2,:], color='green')
    stardust.plot_sphere_wireframe(ax, 1737/384400, [1-mu,0,0], color='grey')
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Testing inner loop success: {prob.inner_loop_success}")
    return


def test_twostage_innerloop_add_node():
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
    _solf_half = solve_ivp(stardust.eom_rotating_cr3bp, (0, period_f/2), rvf, args=(mu, mu1, mu2),
                                 method='RK45', rtol=1e-12, atol=1e-12)
    rvf = _solf_half.y[:,-1]

    # construct problem
    args = (mu, mu1, mu2)
    tspan = [0, 1.2*period_0]
    N = 12
    prob = stardust.FixedTimeTwoStageLeastSquares(
        stardust.eom_stm_rotating_cr3bp,
        rv0,
        rvf,
        tspan,
        N = N,
        args = args,
    )
    prob.create_nodes(
        strategy = 'linear',
    )

    # test inner loop
    print(f"Testing inner loop...")
    sols = prob.inner_loop(prob.nodes[1:-1,0:3].flatten(), get_sols = True, maxiter = 10, verbose = True)
    # assert prob.inner_loop_success == True

    fig, ax, _ = prob.plot_trajectory(show_maneuvers=False)
    ax.plot(sol0_ballistic.y[0,:], sol0_ballistic.y[1,:], sol0_ballistic.y[2,:], color='blue')
    ax.plot(solf_ballistic.y[0,:], solf_ballistic.y[1,:], solf_ballistic.y[2,:], color='green')
    stardust.plot_sphere_wireframe(ax, 1737/384400, [1-mu,0,0], color='grey')
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Before adding node")

    # add node
    t_new_node = (prob.times[1] + prob.times[2])/2
    r_new_node = np.array([1.11, -0.1, -0.16])
    v_new_node = np.zeros(3)
    prob.add_node(t_new_node, r_new_node, v_new_node)

    # test inner loop again
    print(f"Testing inner loop with 1 node added...")
    sols = prob.inner_loop(prob.nodes[1:-1,0:3].flatten(), get_sols = True, maxiter = 10, verbose = True)
    # assert prob.inner_loop_success == True

    fig, ax, _ = prob.plot_trajectory(show_maneuvers=False)
    ax.plot(sol0_ballistic.y[0,:], sol0_ballistic.y[1,:], sol0_ballistic.y[2,:], color='blue')
    ax.plot(solf_ballistic.y[0,:], solf_ballistic.y[1,:], solf_ballistic.y[2,:], color='green')
    stardust.plot_sphere_wireframe(ax, 1737/384400, [1-mu,0,0], color='grey')
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.set_aspect('equal', 'box')
    ax.set_title(f"With added node")
    return


if __name__=="__main__":
    # test_twostage_innerloop_remove_node()
    test_twostage_innerloop_add_node()
    print("Done!")
    plt.show()
    