"""Primer-vector based two-stage optimizer"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime
from tqdm.auto import tqdm

from ._misc import vbprint
from ._twostage_base import _BaseTwoStageOptimizer


class FixedTimeTwoStagePrimerVector(_BaseTwoStageOptimizer):
    """Primer-vector-based two-stage minimization problem for direct-method multi-impulse trajectory design
    The resulting trajectory is approximatly a minimum-fuel multi-impulsive trajectory.

    Note: by setting `ivp_rtol` and `ivp_atol` to 1, the integration effectively becomes
    fixed time-step, with time-step given by `ivp_max_step`. 

    The `eom_stm` function must be of the form `eom_stm(t, y, *args)`, where `y` is t is the time, 
    y is the concatenated state and flatenned state-transition matrix (STM), and args are additional arguments.
    The function must return the concatenated state and flatenned STM derivatives. 
    For example, see: `stardust.eom_stm_rotating_cr3bp`.
    
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