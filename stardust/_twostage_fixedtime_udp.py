"""Two-state shooting algorithm casted as User-Defined Problem
Expected to be used with pygmo
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime
from tqdm.auto import tqdm
import pygmo as pg

from ._misc import vbprint
from ._twostage_base import _BaseTwoStageOptimizer


class FixedTimeTwoStageUDP(_BaseTwoStageOptimizer):
    """Two-stage minimization user-defined problem (UDP)
    
    Args:
        eom_stm (callable): callable function that computes the dynamics for state & STM
        rv0 (np.array): initial position and velocity vector
        rvf (np.array): final position and velocity vector
        tspan (tuple): time span of trajectory
        N (int): number of nodes
        args (any): additional arguments to pass to `eom_stm`
        ivp_method (str): method for `scipy.integrate.solve_ivp`
        ivp_max_step (float): maximum step size for `scipy.integrate.solve_ivp`
        ivp_rtol (float): relative tolerance for `scipy.integrate.solve_ivp`
        ivp_atol (float): absolute tolerance for `scipy.integrate.solve_ivp`
    """
    def __init__(
        self,
        eom_stm,
        rv0,
        rvf,
        tspan,
        N = 10,
        args = None,
        ivp_method = 'RK45',
        ivp_max_step = 0.01,
        ivp_rtol = 1,
        ivp_atol = 1,
        initial_nodes_strategy = 'random_path',
        x_bounds = [0.05, 0.05]
    ):
        # inheritance
        super().__init__(
            eom_stm,
            rv0,
            rvf,
            tspan,
            N = N,
            args = args,
            ivp_method = ivp_method,
            ivp_max_step = ivp_max_step,
            ivp_rtol = ivp_rtol,
            ivp_atol = ivp_atol,
            initial_nodes_strategy = initial_nodes_strategy,
        )

        # overwrite Jacobian storages
        self.J_inner = np.zeros((self.n_seg,3,3))
        self.DF = np.zeros(3*(self.N-2),)

        # store bounds
        self.x_bounds = x_bounds
        return

    def get_bounds(self):
        rs_itm_flat = self.nodes[1:-1,0:3].flatten()    # flattened decision variables
        lb = rs_itm_flat - abs(self.x_bounds[0])
        ub = rs_itm_flat + abs(self.x_bounds[1])
        return (lb, ub)
    
    def get_nec(self):
        return 0
    
    def get_nic(self):
        return 0
    
    def fitness(
        self,
        rs_itm_flat,
        maxiter_inner = 10,
        eps_inner = 1e-11,
    ):
        """Fitness function for pygmo"""
        v_res_flat = self.inner_loop(rs_itm_flat,
                                     maxiter = maxiter_inner,
                                     eps_inner = eps_inner,
                                     verbose = False, get_sols = False)
        v_res_flat = v_res_flat.reshape(self.N, 3)
        return [np.sum(np.linalg.norm(v_res_flat, axis=1)),]
    
    # provide gradients
    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

