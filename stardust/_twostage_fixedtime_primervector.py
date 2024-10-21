"""Primer-vector based two-stage optimizer"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime, minimize
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
        self._reset_outerloop_storage()
        return
    
    def _reset_outerloop_storage(self, reset_innerloop = False):
        """Reset outerloop storage"""
        if reset_innerloop:
            self._reset_innerloop_storage()
        self.gradient = np.zeros(self.N + 3*(self.N - 2))
        return
    
    def outerloop_gradient(self, normalize = True):
        """Compute gradient of the objective function using primer-vector method
        
        The decision vector is assumed to be ordered as:
        `X = [t1, ..., tN, r2^T, ..., r(N-1)^T]`
        The result is stored in property `self.gradient`.

        Ref:
        - K. A. Bokelmann and R. P. Russell, “Optimization of Impulsive Europa Capture Trajectories using 
          Primer Vector Theory,” J. Astronaut. Sci., vol. 67, no. 2, pp. 485–510, Jun. 2020,
          doi: 10.1007/s40295-018-00146-z.
        - B. A. Conway, "Spacecraft Trajectory Optimization," Cambridge University Press, 2010.
        """
        pi_times, pi_histories = self.get_primer_vectors()

        # gradient with respect to times
        for i in range(self.N):
            if i == 0:
                pdot0 = pi_histories[i][3:6,0] @ pi_histories[i][0:3,0]             # Conway eqn (2.59)
                self.gradient[i] = -pdot0 * np.linalg.norm(self.v_residuals[i,:])
            elif i < self.N - 1:
                vi_plus  = self.nodes[i,3:6]                # velocity at node i immediately after DV
                vi_minus = vi_plus - self.v_residuals[i,:]  # velocity at node i immediately before DV
                self.gradient[i] = pi_histories[i-1][3:6,-1] @ vi_minus - pi_histories[i][3:6,0] @ vi_plus
            else:
                pdotf = pi_histories[i-1][3:6,-1] @ pi_histories[i-1][0:3,-1]       # Conway eqn (2.60)
                self.gradient[i] = -pdotf * np.linalg.norm(self.v_residuals[i,:])

        # gradient with respect to intermediate nodes' position vectors
        _gradients_r = np.zeros((self.N-2, 3))
        for i in range(1,self.N-1):
            _gradients_r[i-1,:] = pi_histories[i][3:6,0] - pi_histories[i-1][3:6,-1]
        self.gradient[self.N:] = _gradients_r.flatten()

        if normalize:
            self.gradient /= np.linalg.norm(self.gradient)
        return self.gradient
    
    def solve(
        self,
        maxiter = 20,
        eps_outer = 1e-3,
        eps_fprime = 1e-7, 
        maxiter_inner = 10,
        eps_inner = 1e-11,
        eps_inner_intermediate = 1e-11,
        descent_direction = 'steepest',
        verbose = True,
        verbose_inner = False,
        save_all_iter_sols = False,
    ):
        """Outer loop unconstrained minimization to choose times position vectors of nodes

        Args:
            maxiter (int): maximum number of iterations
            eps_outer (float): tolerance exit condition based on cost progress
            eps_fprime (float): step-size for finite difference Jacobian
            maxiter_inner (int): maximum number of iterations for inner loop
            eps_inner (float): tolerance for inner loop (i.e. tolerance on position continuity)
            eps_inner_intermediate (float): tolerance for inner loop during iteration of outer loop.
            verbose (bool): whether to print progress of outer loop
            verbose_inner (bool): whether to print progress of inner loop
            save_all_iter_sols (bool): whether to save all intermediate solutions
            
        Returns:
            (int, list): exitflag, list of intermediate solutions
        """
        # copy for best solution found so far
        best_nodes = copy.deepcopy(self.nodes)
        best_cost = 1e18
        i_best = -1
        
        # prepare iteration of outer-loop
        iter_sols = []
        dv_cost_last = 1e18
        exitflag = 0

        for it in range(maxiter):
            # run inner loop to make sure there is enough nodes
            _sols_inner_loop = self.inner_loop(maxiter = maxiter_inner,
                                               eps_inner = eps_inner_intermediate,
                                               get_sols = True,
                                               verbose = verbose_inner)
            dv_cost = np.sum(np.linalg.norm(self.v_residuals, axis=1))
            vbprint(f"Outer loop iteration {it} : cost = {dv_cost:1.4e}", verbose)
            
            if self.inner_loop_success is not True:
                vbprint(f"Inner loop failed to converge in outer loop iteration {it}!", verbose)
                break

            # save current nodes if requested
            if save_all_iter_sols:
                iter_sols.append(_sols_inner_loop)

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

            # compute gradient
            self.outerloop_gradient()

            # choose descent direction
            if descent_direction == 'steepest':
                search_dir = -self.gradient
            else:
                raise ValueError(f"Descent direction {descent_direction} not recognized")

            # compute step-size via line-search
            alpha = 0.3

            # perform update on times and nodes
            self.times += alpha * search_dir[0:self.N]
            rs_itm_flat = self.nodes[1:-1,0:3].flatten()            # flattened decision variables
            rs_itm_new = rs_itm_flat + alpha * search_dir[self.N:]
            self.nodes[1:-1,0:3] = rs_itm_new.reshape(self.N-2, 3)

        # if not converged, overwrite with best found nodes so far
        if (exitflag == 0) and (i_best != maxiter):
            vbprint(f"\nRecovering solution from iteration {i_best}", verbose)
            self.nodes[:,:] = best_nodes
            dv_cost = np.sum( (self.v_residuals * self.v_residuals).sum(axis=1)**0.5 )
            print(f"Cost: {dv_cost:1.4e}")
        
        # final "clean-up" run
        vbprint(f"\nFinal clean-up inner loop...", verbose)
        _sols_inner_loop = self.inner_loop(maxiter = maxiter_inner, get_sols = True, verbose = verbose_inner)
        if save_all_iter_sols:
            iter_sols.append(_sols_inner_loop)
        return exitflag, iter_sols
    
    def _cumulative_cost(self, X, maxiter_inner = 10):
        """Objective vector for minimization"""
        self.times = X[0:self.N]
        self.nodes[1:-1,0:3] = X[self.N:].reshape(self.N-2, 3)
        self.inner_loop(maxiter = maxiter_inner, verbose = False)
        dv_sum = np.sum(np.linalg.norm(self.v_residuals, axis=1))
        self.eval_f_count += 1
        print(f"   f eval {self.eval_f_count:3.0f}, objective = {dv_sum:1.4e}")
        return dv_sum
    
    def _gradient(self, X, maxiter_inner = 10):
        self.times = X[0:self.N]
        self.nodes[1:-1,0:3] = X[self.N:].reshape(self.N-2, 3)
        self.inner_loop(maxiter = maxiter_inner, verbose = False)
        return self.outerloop_gradient()
    
    def solve_scipy(self, method = 'BFGS', maxiter = 10, verbose = True):
        """Solve using scipy.optimize.minimize"""
        self.eval_f_count = 0
        X0 = np.concatenate([copy.deepcopy(self.times),
                             copy.deepcopy(self.nodes[1:-1,0:3].flatten())])
        res = minimize(
            self._cumulative_cost,
            X0,
            method = method,
            jac = self._gradient,
            options = {
                'maxiter': maxiter,
                'disp': verbose,
            }
        )
        self.times = res.x[0:self.N]
        self.nodes[1:-1,0:3] = res.x[self.N:].reshape(self.N-2, 3)
        return res