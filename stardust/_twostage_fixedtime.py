"""Two-stage shooting algorithm

Inner loop: enforce dynamics constraints
Outer loop: minimize cost
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime

from ._misc import vbprint
from ._twostage_base import _BaseTwoStageOptimizer


class FixedTimeTwoStageOptimizer(_BaseTwoStageOptimizer):
    """Two-stage optimizer for direct-method multi-impulse trajectory design in fixed-time

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
        ivp_atol = 1
    ):
        # inheritance
        super().__init__(
            eom_stm,
            rv0,
            rvf,
            tspan,
            N,
            args,
            ivp_method,
            ivp_max_step,
            ivp_rtol,
            ivp_atol
        )

        # overwrite Jacobian storages
        self.J_inner = np.zeros((self.n_seg,3,3))
        self.J_outer = np.zeros((3*self.N, 3*(self.N-2)))
        return
    
    
    def _outer_loop_jacobian(self, rs_itm_flat, eps_fprime = 1e-7, maxiter_inner = 10, eps_inner = 1e-11):
        """Compute Jacobian for outer loop
        Note: the Jacobian is stored within the object's preallocated storage. 
        
        Args:
            rs_itm_flat (np.array): flattened position vector of nodes
            eps_fprime (float): step-size for finite difference
            maxiter_inner (int): maximum number of iterations for inner loop
            eps_inner (float): tolerance for inner loop

        Returns:
            None
        """
        J_outer_full = approx_fprime(rs_itm_flat, self.inner_loop, eps_fprime,
                                        maxiter_inner, eps_inner, False, False)
        self.J_outer[:,:] = J_outer_full
        return
    
    def solve(
        self,
        maxiter = 10,
        eps_outer = 1e-3,
        eps_fprime = 1e-7, 
        maxiter_inner = 10,
        eps_inner = 1e-11,
        verbose = True,
        verbose_inner = False,
        save_all_iter_sols = False,
        weights = None,
    ):
        """Outer loop to choose position vector of nodes
        
        Args:
            maxiter (int): maximum number of iterations
            eps_outer (float): tolerance exit condition based on cost progress
            eps_fprime (float): step-size for finite difference Jacobian
            maxiter_inner (int): maximum number of iterations for inner loop
            eps_inner (float): tolerance for inner loop (i.e. tolerance on position continuity)
            verbose (bool): whether to print progress of outer loop
            verbose_inner (bool): whether to print progress of inner loop
            save_all_iter_sols (bool): whether to save all intermediate solutions

        Returns:
            (int, list): exitflag, list of intermediate solutions
        """
        iter_sols = []
        dv_cost_iter = []
        dv_cost_last = 1e18
        exitflag = 0
        if weights is None:
            weights = np.ones(3*self.N)
        else:
            assert len(weights) == 3*self.N,\
                f"weights must be of length 3*N = {3*self.N}, but given {len(weights)}"

        for it in range(maxiter):
            # save current nodes if requested
            if save_all_iter_sols:
                iter_sols.append(self.inner_loop(maxiter = maxiter_inner, get_sols = True, verbose = verbose_inner))

            # get current nodes
            rs_itm_flat = self.nodes[1:-1,0:3].flatten()
            self._outer_loop_jacobian(rs_itm_flat, eps_fprime, maxiter_inner, eps_inner)
            
            # compute cost based on per-maneuver norm
            dv_cost = np.sum( (self.v_residuals * self.v_residuals).sum(axis=1)**0.5 )
            vbprint(f"Outer loop iteration {it} : cost = {dv_cost:1.4e}", verbose)
            dv_cost_iter.append(dv_cost)
            if np.abs(dv_cost_last - dv_cost) < eps_outer:
                vbprint(f"Outer loop converged to within tolerance {eps_outer} in {it} iterations!", verbose)
                exitflag = 1
                break
            else:
                dv_cost_last = dv_cost

            # update positions
            weighted_F = np.multiply(weights, self.v_residuals.flatten())
            rs_itm_new = rs_itm_flat - np.linalg.inv(np.transpose(self.J_outer)@self.J_outer) @ np.transpose(self.J_outer) @ weighted_F
            #self.v_residuals.flatten()
            self.nodes[1:-1,0:3] = rs_itm_new.reshape(self.N-2, 3)
        return exitflag, iter_sols