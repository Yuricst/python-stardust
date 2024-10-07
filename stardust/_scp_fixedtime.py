"""Sequential Convex Program object for mass-optimal trajectory design"""


import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class FixedTimeSCP:
    """Sequential Convex Program object for fixed-time mass-optimal trajectory design
    
    Args:
        eom_stm (callable): callable function that computes the dynamics for state & STM
        rv0 (np.array): initial position and velocity vector
        rvf (np.array): final position and velocity vector
        tspan (tuple): time span of trajectory
        nodes
        func_B_matrix (callable): callable function that computes the B matrix for the segment
        args (any): additional arguments to pass to `eom_stm
    """
    def __init__(
        self,
        eom_stm,
        rv0,
        rvf,
        t_nodes,
        nodes,
        ubars,
        args = None,
        ivp_method = 'RK45',
        ivp_max_step = 0.01,
        ivp_rtol = 1,
        ivp_atol = 1
    ):
        assert len(rv0) == len(rvf) == 6, "Initial and final states must be length-6 vectors"
        assert nodes.shape == (len(t_nodes), 6), "State and control nodes must be (N, 6)"
        assert ubars.shape == (len(t_nodes), 3), "Control nodes must have shape (N, 3)"
        self.eom_stm = eom_stm
        self.rv0 = rv0
        self.rvf = rvf
        self.t_nodes = t_nodes
        self.nodes = nodes
        self.ubars = ubars
        self.N = nodes.shape[0]
        self.n_seg = self.N - 1
        self.args = args
        self.ivp_method = ivp_method
        self.ivp_max_step = ivp_max_step
        self.ivp_rtol = ivp_rtol
        self.ivp_atol = ivp_atol

        # initialize storage needed during solve process
        self.nx = 6
        self.As = np.zeros((self.n_seg, self.nx, self.nx))
        self.Bs = np.zeros((self.n_seg, self.nx, 3))
        return
    
    
    def get_discrete_time_linear_model(self, dense_output = False):
        """Construct discrete-time linear model for SCP"""
        for i in range(self.n_seg):
            # compute STM
            sol_i = solve_ivp(self.eom_stm, 
                            (self.t_nodes[i], self.t_nodes[i+1]),
                            np.concatenate((self.nodes[i], np.eye(6).flatten())),
                            args=self.args,
                            dense_output=dense_output, method=self.ivp_method,
                            max_step=self.ivp_max_step, rtol=self.ivp_rtol, atol=self.ivp_atol)
            stm = sol_i.y[6:,-1].reshape(6,6)

            # store A and B matrices
            self.As[i,:,:] = stm
            self.Bs[i,:,:] = stm @ np.concatenate((np.zeros((3,3)), np.eye(3)))
        return
    
    
    def propagate_nonlinear(self, us=None, use_itm_nodes=False, dense_output=False):
        """Propagate nonlinear dynamics"""
        if us is not None:
            assert us.shape == (self.N, 3), "Control input must have shape (N, 3)"
        sols = []
        # iterate through segments
        for i in range(self.n_seg):
            if use_itm_nodes or i == 0:
                x0 = np.concatenate((self.nodes[i], np.eye(6).flatten()))
            else:
                x0 = np.concatenate((sol_i.y[:6,-1], np.eye(6).flatten()))
                
            if us is not None:
                x0[3:6] += us[i,:]

            sol_i = solve_ivp(self.eom_stm, 
                            (self.t_nodes[i], self.t_nodes[i+1]),
                            x0,
                            args=self.args,
                            dense_output=dense_output,
                            method=self.ivp_method,
                            max_step=self.ivp_max_step,
                            rtol=self.ivp_rtol,
                            atol=self.ivp_atol)
            sols.append(sol_i)
        return sols
    

    def solve_socp(self, trust_region, solver=cp.SCS, verbose_solver=False, max_iters=100000):
        """Solve SOCP for mass-optimal trajectory design"""
        # create matrices
        self.get_discrete_time_linear_model()

        # create variables
        xs = cp.Variable((self.N, self.nx))         # state deviation
        us = cp.Variable((self.N, 3))               # control
        etas = cp.Variable(self.N)                  # slack for cost

        # dynamics constraints
        h_dynamics = []
        for i,(A,B) in enumerate(zip(self.As, self.Bs)):
            h_dynamics += [
                xs[i+1] == A @ xs[i,:] + B @ us[i,:]
            ]

        # initial and final states constraints
        h_boundaries = [
            self.nodes[0,:]    + xs[0,:] == self.rv0,
            self.nodes[-1,0:3] + xs[-1,0:3] == self.rvf[0:3],
            self.nodes[-1,3:6] + xs[-1,3:6] == self.rvf[3:6] + us[-1,:]
        ]

        # construct cvxpy constraints for objective
        objective_slack_constraints = [
            cp.SOC(t = etas[i],                 # cone constraints
                    X = np.eye(3) @ us[i,:] + self.ubars[i,:]) 
            for i in range(self.N)
        ]

        # trust region constraints
        trust_region = np.array(trust_region)
        h_trust_region = []
        for i in range(self.N):
            h_trust_region += [xs[i,:] <= trust_region,
                               xs[i,:] >= -trust_region]

        # solve convex program
        obj_cost = np.ones(self.N)    # cost vector
        prob = cp.Problem(cp.Minimize(obj_cost.T @ etas),
            h_dynamics + h_boundaries + objective_slack_constraints)# + h_trust_region)
        prob.solve(solver=solver, verbose=verbose_solver, max_iters=max_iters)

        if prob.status != cp.OPTIMAL:
            return 0, [], []
        
        # extract solution
        xs_opt = xs.value
        us_opt = us.value
        return 1, xs_opt, us_opt