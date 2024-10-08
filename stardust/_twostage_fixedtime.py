"""Two-stage shooting algorithm

Inner loop: enforce dynamics constraints
Outer loop: minimize cost
"""

import copy
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
    
    
    def _outer_loop_jacobian(self, eps_fprime = 1e-7, maxiter_inner = 10, eps_inner = 1e-11):
        """Compute sparse Jacobian of dimension (3N, 3(N-2)) for outer loop

        The Jacobian has dimensions (3N, 3(N-2)).
        Note: the Jacobian is stored within the object's preallocated storage. 
        
        Args:
            rs_itm_flat (np.array): flattened position vector of nodes
            eps_fprime (float): step-size for finite difference
            maxiter_inner (int): maximum number of iterations for inner loop
            eps_inner (float): tolerance for inner loop

        Returns:
            None
        """
        rs_itm_flat = self.nodes[1:-1,0:3].flatten()
        self.J_outer[:,:] = approx_fprime(rs_itm_flat, self.inner_loop, eps_fprime,
                                          maxiter_inner, eps_inner, False, False)
        return
    

    def _inner_loop_sparse(
        self,
        r_perturbed,
        rs_nodes,
        index_r: int,
        maxiter_inner = 10,
        eps_inner = 1e-11,
    ):
        """Wrap for inner_loop method where only part of the DV is returned, to be used by `approx_fprime`"""
        assert 0 < index_r < self.N-1, "index_r must be between 1 and N-2"
        rs_nodes[index_r,0:3] = r_perturbed
        rs_itm_flat = rs_nodes[1:-1,:].flatten()
        v_res_flat = self.inner_loop(rs_itm_flat, maxiter = maxiter_inner, eps_inner = eps_inner, verbose = False, get_sols = False)
        v_res_flat = v_res_flat.reshape(self.N, 3)
        return v_res_flat[index_r-1:index_r+2,:].flatten()
    
    
    def _outer_loop_jacobian_sparse(self, eps_fprime = 1e-7, maxiter_inner = 10, eps_inner = 1e-11):
        """Compute sparse Jacobian of dimension (3N, 3(N-2)) for outer loop
        
        The sparse Jaocbian is a faster approximation where we assume the position vector r_i
        only impacts DV_{i-1}, DV_i, and DV_{i+1}. 

        Note: the Jacobian is stored within the object's preallocated storage. 
        
        Args:
            rs_itm_flat (np.array): flattened position vector of nodes
            eps_fprime (float): step-size for finite difference
            maxiter_inner (int): maximum number of iterations for inner loop
            eps_inner (float): tolerance for inner loop

        Returns:
            None
        """
        self.J_outer[:,:] = np.zeros((3*self.N, 3*(self.N-2)))
        for i in range(1,self.N-1):
            r_perturbed = self.nodes[i,0:3].copy()
            jac_i = approx_fprime(r_perturbed, self._inner_loop_sparse, eps_fprime,
                                  self.nodes[:,0:3], i, maxiter_inner, eps_inner)
            self.J_outer[3*i-3:3*i+6, 3*(i-1):3*i] = jac_i
        return
    
    def solve(
        self,
        maxiter = 10,
        eps_outer = 1e-3,
        eps_fprime = 1e-7, 
        maxiter_inner = 10,
        eps_inner = 1e-11,
        verbose = True,
        verbose_inner = True,
        save_all_iter_sols = False,
        weights = None,
        sparse_approx_jacobian = True,
    ):
        """Outer loop to choose position vector of nodes

        The `weights` argument allows the user to penalize/neglect the cost of a certain
        maneuver component. For example, if the initial maneuver cost can be omitted, then
        consider using: `weights = [0,0,0] + [1,1,1] * N-1`. 
        
        Args:
            maxiter (int): maximum number of iterations
            eps_outer (float): tolerance exit condition based on cost progress
            eps_fprime (float): step-size for finite difference Jacobian
            maxiter_inner (int): maximum number of iterations for inner loop
            eps_inner (float): tolerance for inner loop (i.e. tolerance on position continuity)
            verbose (bool): whether to print progress of outer loop
            verbose_inner (bool): whether to print progress of inner loop
            save_all_iter_sols (bool): whether to save all intermediate solutions
            weights (list): weights for least-squares problem for each maneuver component

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

        # copy for best solution found so far
        best_nodes = copy.deepcopy(self.nodes)
        best_cost = 1e18
        i_best = -1

        for it in range(maxiter):
            # run inner loop to make sure there is enough nodes
            _sols_inner_loop = self.inner_loop(maxiter = maxiter_inner, get_sols = True, verbose = verbose_inner)
            if self.inner_loop_success is not True:
                vbprint(f"Inner loop failed to converge in outer loop iteration {it}!", verbose)
                break

            # save current nodes if requested
            if save_all_iter_sols:
                iter_sols.append(_sols_inner_loop)

            # get current nodes
            if sparse_approx_jacobian:
                print(f"  Computing outer-loop sparse Jacobian...")
                self._outer_loop_jacobian_sparse(eps_fprime, maxiter_inner, eps_inner)
            else:
                print(f"  Computing outer-loop dense Jacobian...")
                self._outer_loop_jacobian(eps_fprime, maxiter_inner, eps_inner)
            
            # compute cost based on per-maneuver norm
            dv_cost = np.sum( (self.v_residuals * self.v_residuals).sum(axis=1)**0.5 )
            vbprint(f"Outer loop iteration {it} : cost = {dv_cost:1.4e}", verbose)
            dv_cost_iter.append(dv_cost)

            # store best node if improved
            if dv_cost < best_cost:
                best_nodes[:,:] = copy.deepcopy(self.nodes)
                best_cost = dv_cost
                i_best = it

            # check exit condition
            if np.abs(dv_cost_last - dv_cost) < eps_outer:
                vbprint(f"Outer loop converged to within tolerance {eps_outer} in {it} iterations!", verbose)
                exitflag = 1
                break
            else:
                dv_cost_last = dv_cost

            # update positions
            weighted_F = np.multiply(weights, self.v_residuals.flatten())
            rs_itm_new = self.nodes[1:-1,0:3].flatten() - np.linalg.solve(self.J_outer.T @ self.J_outer, self.J_outer.T @ weighted_F)
            #self.v_residuals.flatten()
            self.nodes[1:-1,0:3] = rs_itm_new.reshape(self.N-2, 3)

        # if not converged, overwrite with best found nodes so far
        if (exitflag == 0) and (i_best != maxiter-1):
            print(f"Recovering solution from iteration {i_best}")
            self.nodes[:,:] = best_nodes
            self.propagate()                # call to store r_residuals and v_residuals
            dv_cost = np.sum( (self.v_residuals * self.v_residuals).sum(axis=1)**0.5 )
            print(f"Cost: {dv_cost:1.4e}")
        return exitflag, iter_sols