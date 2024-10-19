"""Base class for two-stage optimizer

Inner loop: enforce dynamics constraints
Outer loop: minimize cost
"""

import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime

from ._misc import vbprint


class _BaseTwoStageOptimizer:
    """Base class for two-stage optimizer
    
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
        assert N > 1, "Number of nodes must be greater than 1"
        self.eom_stm = eom_stm
        self.rv0 = rv0
        self.rvf = rvf
        self.tspan = tspan
        self.N = N
        self.n_seg = N - 1
        self.nx = 6

        self.args = args
        self.ivp_method = ivp_method
        self.ivp_max_step = ivp_max_step
        self.ivp_rtol = ivp_rtol
        self.ivp_atol = ivp_atol

        # initialize storage needed during solve process
        self.r_residuals = np.zeros((self.N, 3))
        self.v_residuals = np.zeros((self.N, 3))

        # initialize nodes
        self.create_nodes(strategy = initial_nodes_strategy)
        self.inner_loop_success = True
        return 
    
    
    def create_nodes(self,
                     nodes_ig = None,
                     strategy = 'random_path',
                     sample_box = None,
                     offset_rectilinear = 0.01):
        """Create nodes for optimization
        
        Args:
            nodes_ig (np.array): initial guess for nodes
            strategy (str): strategy to connect nodes, either 'linear' or 'random_path' or 'random_box'
            offset_rectilinear (float): offset to avoid rectilinear motion in 'random' strategy
        """
        __ig_strategies = ['linear', 'random_path', 'random_box']
        assert strategy in __ig_strategies,\
            f"strategy {strategy} not recognized (allows: {__ig_strategies})"
        
        # create time stamps of nodes and integration time-spans
        self.times = np.linspace(self.tspan[0], self.tspan[1], self.N)
        self.tspans = []
        for i in range(self.n_seg):
            self.tspans.append((self.times[i], self.times[i+1]))
        
        # nodes initial guess
        if nodes_ig is not None:
            assert nodes_ig.shape == (self.N, 6), "nodes_ig must be of shape (N, 6)"
            self.nodes = nodes_ig
            return

        elif strategy == 'linear':
            # linearly connect rv0 and rvf
            self.nodes = np.linspace(self.rv0, self.rvf, self.N)

        elif strategy == 'random_box':
            #assert box is not None, "box must be provided to use 'random_box' strategy"
            assert sample_box.shape == (3,2), "box must be of shape (3,2)"

            # create random nodes
            self.nodes = np.zeros((self.N, 6))
            self.nodes[:,0:3] = np.multiply(np.random.rand(self.N,3), sample_box[:,1] - sample_box[:,0]) + sample_box[:,0]

        elif strategy == 'random_path':
            # get bounding box between rv0 and rvf
            bounds = [
                [np.min([el0, elf]), np.max([el0, elf])] 
                for (el0, elf) in zip(self.rv0, self.rvf)
            ]
            for bound in bounds:
                if bound[0] == bound[1]:
                    bound[0] -= offset_rectilinear  # offset to avoid rectilinear motion
                    bound[1] += offset_rectilinear  # offset to avoid rectilinear motion
            bounds = np.array(bounds)
            ranges = bounds[:,1] - bounds[:,0]

            # create random nodes
            self.nodes = np.multiply(np.random.rand(self.N,6), ranges) + bounds[:,0]
            self.nodes[0] = self.rv0
            self.nodes[-1] = self.rvf
        
        # over-write velocity guess based on linear approximation
        for i in range(self.n_seg):
            vi_guess = (self.nodes[i+1,0:3] - self.nodes[i,0:3])/(self.times[i+1] - self.times[i])
            self.nodes[i,3:6] = vi_guess
        return
    
    
    def propagate(self, get_sols=True, dense_output=False, use_itm_nodes=True, nodes = None):
        """Propagate nodes
        
        Args:
            get_sols (bool): whether to return solutions of each segment
            dense_output (bool): whether to use dense output on each solution object
            use_itm_nodes (bool): whether to use intermediate nodes to propagate segment-wise
            nodes (np.array): nodes to propagate; if not provided, the function uses self.nodes

        Returns:
            (list or None): list of solution objects of each segment or None
        """
        if nodes is None:
            nodes = self.nodes
        else:
            assert nodes.shape == self.nodes.shape, "nodes must be of shape (N, 6)"

        if get_sols:
            sols = []
        # initial node residuals
        self.r_residuals[0,:] = nodes[0,0:3] - self.rv0[0:3]
        self.v_residuals[0,:] = nodes[0,3:6] - self.rv0[3:6]

        # iterate through segments
        for i in range(self.n_seg):
            if use_itm_nodes or i == 0:
                x0 = np.concatenate((nodes[i], np.eye(6).flatten()))
            else:
                x0 = np.concatenate((sol_i.y[:6,-1] + np.concatenate(([0,0,0], self.v_residuals[i])),
                                     np.eye(6).flatten()))

            sol_i = solve_ivp(self.eom_stm, 
                            self.tspans[i],
                            x0,
                            args=self.args,
                            dense_output=dense_output, method=self.ivp_method,
                            max_step=self.ivp_max_step, rtol=self.ivp_rtol, atol=self.ivp_atol)
            stm = sol_i.y[6:,-1].reshape(6,6)
            if get_sols:
                sols.append(sol_i)

            # store STM and residuals
            self.r_residuals[i+1,:] = nodes[i+1,0:3] - sol_i.y[0:3,-1]
            self.v_residuals[i+1,:] = nodes[i+1,3:6] - sol_i.y[3:6,-1]
            self.J_inner[i,:,:] = -stm[0:3,3:6]    # sensitivity of final position w.r.t. initial velocity
        if get_sols:
            return sols
        else:
            return None
        
    
    def inner_loop(
        self,
        rs_itm_flat = None,
        maxiter = 30,
        eps_inner = 1e-11,
        verbose = True,
        get_sols = False,
        overwrite_nodes = True,
    ):
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
                f"rs_itm_flat must be of shape 3*(N-2) = {3*(self.N-2)}, but given {rs_itm_flat.shape}"
            self.nodes[1:-1,0:3] = rs_itm_flat.reshape(self.N-2, 3)

        # setups
        _nodes = copy.deepcopy(self.nodes)      # we deepcopy to avoid overwriting during Jacobian calculation of outer-loop
        self.inner_loop_success = False         # initially set to False
        res_norms = np.zeros(self.n_seg)
        for it in range(maxiter):
            # propagate nodes
            sols = self.propagate(get_sols = get_sols, nodes = _nodes)

            # update nodes
            for i in range(self.n_seg):
                # compute velocity correction
                J = self.J_inner[i,:,:]
                _nodes[i,3:6] -= np.linalg.solve(J, self.r_residuals[i+1,:])
                #self.nodes[i,3:6] -= np.linalg.inv(J) @ self.r_residuals[i,:]
                res_norms[i] = np.linalg.norm(self.r_residuals[i,:])
            vbprint(f"    Inner loop {it}: max position residual norm: {max(res_norms):1.4e}", verbose)
            if max(res_norms) < eps_inner:
                vbprint(f"    Inner loop converged to within tolerance {eps_inner} in {it} iterations!", verbose)
                self.inner_loop_success = True
                break

        if overwrite_nodes:
            self.nodes[:,:] = _nodes

        if get_sols:
            return sols
        else:
            return self.v_residuals.flatten()
    
    
    def solve(self):
        """Outer loop to choose position vector of nodes

        Returns:
            (int, list): exitflag, list of intermediate solutions
        """
        raise NotImplementedError("This method must be implemented in a subclass")
        return 0, []
    
    
    def plot_trajectory(self, use_itm_nodes=True, show_maneuvers=True):
        """Plot trajectory in 3D space"""
        sols = self.propagate(get_sols = True, dense_output=False, use_itm_nodes=use_itm_nodes)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.nodes[:,0], self.nodes[:,1], self.nodes[:,2], 'o', color='black')
        if show_maneuvers:
            ax.quiver(self.nodes[:,0], self.nodes[:,1], self.nodes[:,2],
                      self.v_residuals[:,0], self.v_residuals[:,1], self.v_residuals[:,2],
                      color='red')
        for sol in sols:
            ax.plot(sol.y[0,:], sol.y[1,:], sol.y[2,:], color='black')
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        ax.set_aspect('equal', 'box')
        return fig, ax, sols
    
    
    def plot_deltaV(self, VU = 1.0, ax=None, use_itm_nodes=True, figsize=(8,6)):
        """Plot delta-V's"""
        self.propagate(get_sols = False, dense_output=False, use_itm_nodes=use_itm_nodes)
        if ax is None:
            new_plot = True
            fig, ax = plt.subplots(1,1,figsize=figsize)
        else:
            new_plot = False
        ax.stem(self.times, np.linalg.norm(self.v_residuals, axis=1)*VU)
        ax.set(xlabel='Time', ylabel='Delta-V magnitude')
        if new_plot:
            return fig, ax
        else:
            return
    
    
    def get_trajectory(self):
        """Get trajectory segments and maneuvers
        
        Returns:
            (tuple): tuple containing
                - list of length-Ki segment times,
                - list of array with shape (6,Ni) of segment states, 
                - length-N list of maneuver times,
                - array with shape (N,3) of maneuver DVs, one maneuver in each row
        """
        sols = self.propagate(get_sols = True, dense_output=False)
        ts_segments = [sol.t for sol in sols]
        ys_segments = [sol.y for sol in sols]
        return ts_segments, ys_segments, self.times, self.v_residuals