"""Two-stage shooting algorithm via unconstrained minimization outer-loop
This algorithm results in (approximately) a minimum-energy impulsive trajectory. 

Inner loop: enforce dynamics constraints
Outer loop: minimize impulse costs via unconstrained minimization
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime
from tqdm.auto import tqdm

from ._misc import vbprint
from ._twostage_base import _BaseTwoStageOptimizer


class FixedTimeTwoStageMinimizer(_BaseTwoStageOptimizer):
    """Two-stage minimization problem for direct-method multi-impulse trajectory design in fixed-time
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

        # overwrite Jacobian storages
        self.J_inner = np.zeros((self.n_seg,3,3))
        self.DF = np.zeros(3*(self.N-2),)
        return
    
    def _linesearch(self, c1=0.0001, c2=0.9, maxiter=10):
        """Compute step-size for unconstraind minimization"""
        return
    
    def _descent_direction(self):
        """Compute descent direction for unconstrained minimization"""
        return
    
    def _inner_loop_to_cumulative_cost(
        self, 
        rs_itm_flat,
        maxiter_inner = 20,
        eps_inner = 1e-11,
        verbose = False,
    ):
        """Compute cumulative cost for inner loop"""
        rs_itm_flat = self.nodes[1:-1,0:3].flatten()
        v_res_flat = self.inner_loop(rs_itm_flat, maxiter = maxiter_inner,
                                     eps_inner = eps_inner, verbose = verbose, get_sols = False)
        v_res_flat = v_res_flat.reshape(self.N, 3)
        return np.sum(np.linalg.norm(v_res_flat, axis=1))
    
    
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
        verbose_inner = True,
        save_all_iter_sols = False,
    ):
        """Outer loop unconstrained minimization to choose position vector of nodes

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
        iter_sols = []
        dv_cost_last = 1e18
        exitflag = 0

        # copy for best solution found so far
        best_nodes = copy.deepcopy(self.nodes)
        best_cost = 1e18
        i_best = -1
        
        # iterate
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
            rs_itm_flat = self.nodes[1:-1,0:3].flatten()    # flattened decision variables
            self.DF[:] = approx_fprime(
                rs_itm_flat, self._inner_loop_to_cumulative_cost, eps_fprime,
                maxiter_inner, eps_inner, False,
            )

            # choose descent direction
            if descent_direction == 'steepest':
                search_dir = -self.DF / np.linalg.norm(self.DF)
            else:
                raise ValueError(f"Descent direction {descent_direction} not recognized")

            # compute step-size via line-search
            alpha = 0.2

            # perform update on positions
            rs_itm_new = rs_itm_flat + alpha * search_dir
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
    
    def plot_linesearch(
        self,
        alphas,
        eps_fprime = 1e-7, 
        maxiter_inner = 10,
        eps_inner = 1e-11,
        eps_inner_intermediate = 1e-9,
        descent_direction = 'steepest',
        verbose_inner = False,
    ):
        # compute gradient
        rs_itm_flat = self.nodes[1:-1,0:3].flatten()    # flattened decision variables
        self.DF[:] = approx_fprime(
            rs_itm_flat, self._inner_loop_to_cumulative_cost, eps_fprime,
            maxiter_inner, eps_inner, True,
        )
        
        if descent_direction == 'steepest':
            search_dir = -self.DF / np.linalg.norm(self.DF)
        else:
            raise ValueError(f"Descent direction {descent_direction} not recognized")

        # compute new cost
        dv_costs = []
        for alpha in tqdm(alphas):
            rs_itm_new = rs_itm_flat + alpha * search_dir
            self.nodes[1:-1,0:3] = rs_itm_new.reshape(self.N-2, 3)
            self.inner_loop(maxiter = maxiter_inner,
                                               eps_inner = eps_inner_intermediate,
                                               get_sols = True,
                                               verbose = verbose_inner)
            dv_costs.append(np.sum(np.linalg.norm(self.v_residuals, axis=1)))

        # create plot
        fig, ax = plt.subplots(1,1,figsize=(7,5))
        ax.plot(alphas, dv_costs)
        ax.set(xlabel="Step-size", ylabel="DV cost", yscale="log")
        ax.grid(True, alpha=0.3)
        return alphas, dv_costs, fig, ax