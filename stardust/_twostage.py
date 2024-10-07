"""Two-stage shooting algorithm

Inner loop: enforce dynamics constraints
Outer loop: minimize cost
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime

from ._misc import vbprint


class TwoStageOptimizer:
    """Two-stage optimizer for direct-method multi-impulse trajectory design in fixed-time

    Note: by setting `ivp_rtol` and `ivp_atol` to 1, the integration effectively becomes
    fixed time-step, with time-step given by `ivp_max_step`. 
    
    Args:
        eom_stm (callable): callable function that computes the dynamics for state & STM
        rv0 (np.array): initial position and velocity vector
        rvf (np.array): final position and velocity vector
        N (int): number of nodes
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
        self.eom_stm = eom_stm
        self.rv0 = rv0
        self.rvf = rvf
        self.tspan = tspan
        self.N = N
        self.n_seg = N - 1

        self.args = args
        self.ivp_method = ivp_method
        self.ivp_max_step = ivp_max_step
        self.ivp_rtol = ivp_rtol
        self.ivp_atol = ivp_atol

        # initialize storage needed during solve process
        self.r_residuals = np.zeros((self.N, 3))
        self.v_residuals = np.zeros((self.N, 3))
        self.J_inner = np.zeros((self.n_seg,3,3))
        self.J_outer = np.zeros((3*self.N, 3*(self.N-2)))

        # initialize nodes
        self.create_nodes()
        return 
    
    
    def create_nodes(self, nodes_ig = None, strategy = 'linear'):
        """Create nodes for optimization
        
        Args:
            nodes_ig (np.array): initial guess for nodes
            strategy (str): strategy to connect nodes
        """
        # create time stamps of nodes and integration time-spans
        self.times = np.linspace(self.tspan[0], self.tspan[1], self.N)
        self.tspans = []
        for i in range(self.n_seg):
            self.tspans.append((self.times[i], self.times[i+1]))
        
        # nodes initial guess
        if nodes_ig is not None:
            assert nodes_ig.shape == (self.N, 6), "nodes_ig must be of shape (N, 6)"
            self.nodes = nodes_ig

        elif strategy == 'linear':
            # linearly connect rv0 and rvf
            self.nodes = np.linspace(self.rv0, self.rvf, self.N)
            # over-write velocity guess based on linear approximation
            for i in range(self.n_seg):
                vi_guess = (self.nodes[i+1,0:3] - self.nodes[i,0:3])/(self.times[i+1] - self.times[i])
                self.nodes[i,3:6] = vi_guess
        return
    
    
    def propagate(self, get_sols=True, dense_output=False):
        """Propagate nodes
        
        Args:
            get_sols (bool): whether to return solutions of each segment
            dense_output (bool): whether to use dense output on each solution object

        Returns:
            (list or None): list of solution objects of each segment or None
        """
        if get_sols:
            sols = []
        # initial node residuals
        self.r_residuals[0,:] = self.nodes[0,0:3] - self.rv0[0:3]
        self.v_residuals[0,:] = self.nodes[0,3:6] - self.rv0[3:6]

        # iterate through segments
        for i in range(self.n_seg):
            tspan = self.tspans[i]
            rv0 = np.concatenate((self.nodes[i], np.eye(6).flatten()))
            sol_i = solve_ivp(self.eom_stm, 
                            self.tspans[i], 
                            np.concatenate((self.nodes[i], np.eye(6).flatten())),
                            args=self.args,
                            dense_output=dense_output, method=self.ivp_method,
                            max_step=self.ivp_max_step, rtol=self.ivp_rtol, atol=self.ivp_atol)
            stm = sol_i.y[6:,-1].reshape(6,6)
            if get_sols:
                sols.append(sol_i)

            # store STM and residuals
            self.r_residuals[i+1,:] = self.nodes[i+1,0:3] - sol_i.y[0:3,-1]
            self.v_residuals[i+1,:] = self.nodes[i+1,3:6] - sol_i.y[3:6,-1]
            self.J_inner[i,:,:] = -stm[0:3,3:6]    # sensitivity of final position w.r.t. initial velocity
        if get_sols:
            return sols
        else:
            return None
        
    
    def inner_loop(self, rs_itm_flat = None, maxiter = 1, eps_inner = 1e-11, verbose = True, get_sols = False):
        """Enforce dynamics constraint by computing necessary velocity vectors
        If this function is called with `get_sols = False`, it returns the velocity residuals.
        This is used in the outer loop to compute the Jacobian with `approx_fprime`.
        
        Args:
            rs_itm_flat (np.array): flattened position vector of nodes
            maxiter (int): maximum number of iterations
            eps_inner (float): tolerance for inner loop (i.e. tolerance on position continuity)
            verbose (bool): whether to print progress
            get_sols (bool): whether to return solution objects

        Returns:
            (list or np.array): list of solution objects of each segment or velocity residuals
        """
        if rs_itm_flat is not None:
            assert rs_itm_flat.shape == (3*(self.N-2),),\
                f"rs_flat must be of shape 3*(N-2) = {3*(self.N-2)}, but given {rs_itm_flat.shape}"
            self.nodes[1:-1,0:3] = rs_itm_flat.reshape(self.N-2, 3)

        res_norms = np.zeros(self.n_seg)
        for it in range(maxiter):
            # propagate nodes
            sols = self.propagate(get_sols = get_sols)

            # update nodes
            for i in range(self.n_seg):
                # compute velocity correction
                J = self.J_inner[i,:,:]
                self.nodes[i,3:6] -= np.linalg.solve(J, self.r_residuals[i+1,:])
                #self.nodes[i,3:6] -= np.linalg.inv(J) @ self.r_residuals[i,:]
                res_norms[i] = np.linalg.norm(self.r_residuals[i,:])
            vbprint(f"    Inner loop {it}: max position residual norm: {max(res_norms):1.4e}", verbose)
            if max(res_norms) < eps_inner:
                vbprint(f"    Inner loop converged to within tolerance {eps_inner} in {it} iterations!", verbose)
                break
        if get_sols:
            return sols
        else:
            return self.v_residuals.flatten()
    
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
        #rs_itm_flat = self.nodes[1:-1,0:3].flatten()
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

        for it in range(maxiter):
            # save current nodes if requested
            if save_all_iter_sols:
                iter_sols.append(self.inner_loop(get_sols = True, verbose = verbose_inner))

            # get current nodes
            rs_itm_flat = self.nodes[1:-1,0:3].flatten()
            self._outer_loop_jacobian(rs_itm_flat, eps_fprime, maxiter_inner, eps_inner)
            
            dv_cost = np.linalg.norm(self.v_residuals)
            vbprint(f"Outer loop iteration {it} : cost = {dv_cost:1.4e}", verbose)
            dv_cost_iter.append(dv_cost)
            if np.abs(dv_cost_last - dv_cost) < eps_outer:
                vbprint(f"Outer loop converged to within tolerance {eps_outer} in {it} iterations!", verbose)
                exitflag = 1
                break
            else:
                dv_cost_last = dv_cost

            # update positions
            rs_itm_new = rs_itm_flat - np.linalg.inv(np.transpose(self.J_outer)@self.J_outer) @ np.transpose(self.J_outer) @ self.v_residuals.flatten()
            self.nodes[1:-1,0:3] = rs_itm_new.reshape(self.N-2, 3)
        return exitflag, iter_sols
    
    
    def plot_trajectory(self):
        """Plot trajectory in 3D space"""
        sols = self.propagate(get_sols = True, dense_output=False)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.nodes[:,0], self.nodes[:,1], self.nodes[:,2], 'o', color='black')
        ax.quiver(self.nodes[:,0], self.nodes[:,1], self.nodes[:,2],
                  self.v_residuals[:,0], self.v_residuals[:,1], self.v_residuals[:,2], color='red')
        for sol in sols:
            ax.plot(sol.y[0,:], sol.y[1,:], sol.y[2,:], color='black')
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        ax.set_aspect('equal', 'box')
        return fig, ax