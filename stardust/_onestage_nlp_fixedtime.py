"""Shooting NLP"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import Bounds, NonlinearConstraint


class FixedTimeShootingNLP:
    def __init__(
        self,
        eom_stm,
        rv0,
        rvf,
        t_nodes,
        nodes,
        us,
        args = None,
        ivp_method = 'RK45',
        ivp_max_step = 0.01,
        ivp_rtol = 1,
        ivp_atol = 1
    ):
        assert len(rv0) == len(rvf) == 6, "Initial and final states must be length-6 vectors"
        assert nodes.shape == (len(t_nodes), 6), "State and control nodes must be (N, 6)"
        assert us.shape == (len(t_nodes), 3), "Control nodes must have shape (N, 3)"
        self.eom_stm = eom_stm
        self.rv0 = rv0
        self.rvf = rvf
        self.t_nodes = t_nodes
        self.nodes = nodes
        self.us = us
        self.N = nodes.shape[0]
        self.n_seg = self.N - 1
        self.args = args
        self.ivp_method = ivp_method
        self.ivp_max_step = ivp_max_step
        self.ivp_rtol = ivp_rtol
        self.ivp_atol = ivp_atol
        self.nx = 6

        # initialize storage needed during solve process
        self.r_residuals = np.zeros((self.N, 3))
        self.v_residuals = np.zeros((self.N, 3))

        # store hash of latest set of nodes
        self.hash_nodes = hash(str(self.nodes.flatten()))
        return
    
    
    def propagate(self, get_sols=True, dense_output=False, use_itm_nodes=True):
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
            if use_itm_nodes or i == 0:
                x0 = np.concatenate((self.nodes[i], np.eye(6).flatten()))
            else:
                x0 = np.concatenate((sol_i.y[:6,-1] + np.concatenate(([0,0,0], self.v_residuals[i])),
                                     np.eye(6).flatten()))

            sol_i = solve_ivp(self.eom_stm, 
                            (self.t_nodes[i], self.t_nodes[i+1]),
                            x0,
                            args=self.args,
                            dense_output=dense_output, method=self.ivp_method,
                            max_step=self.ivp_max_step, rtol=self.ivp_rtol, atol=self.ivp_atol)
            stm = sol_i.y[6:,-1].reshape(6,6)
            if get_sols:
                sols.append(sol_i)

            # store STM and residuals
            self.r_residuals[i+1,:] = self.nodes[i+1,0:3] - sol_i.y[0:3,-1]
            self.v_residuals[i+1,:] = self.nodes[i+1,3:6] - sol_i.y[3:6,-1]
        if get_sols:
            return sols
        else:
            return None
        

    def objective(self, xs_flat):
        if self.hash_nodes != hash(str(xs_flat)):
            # update hash of nodes
            self.hash_nodes = hash(str(xs_flat))

            # unpack nodes & store them, then propagate
            self.nodes[:,:] = xs_flat[:self.N*self.nx].reshape(self.N, self.nx)
            self.propagate(get_sols=False, dense_output=False, use_itm_nodes=True)
        else:
            print(f"Using cached residuals")

        # compute objective
        dv_cost = np.sum( (self.v_residuals * self.v_residuals).sum(axis=1)**0.5 )
        return dv_cost
    

    def get_constraints(self, get_object=True):
        """Construct constraints object"""
        def rvf_constraint(xs_flat):
            if self.hash_nodes != hash(str(xs_flat)):
                # update hash of nodes
                self.hash_nodes = hash(str(xs_flat))

                # unpack nodes & store them, then propagate
                self.nodes[:,:] = xs_flat[:self.N*self.nx].reshape(self.N, self.nx)
                self.propagate(get_sols=False, dense_output=False, use_itm_nodes=True)
            else:
                print(f"Using cached residuals")
            return self.r_residuals[1:,:].flatten()
        
        if get_object:
            lb_c = np.zeros(3*self.n_seg)
            ub_c = np.zeros(3*self.n_seg)
            return NonlinearConstraint(rvf_constraint, lb_c, ub_c)
        else:
            return rvf_constraint
    

    def get_bounds(self, trust_region):
        """Construct bounds object"""
        assert len(trust_region) == 6, "Trust region must be a length-6 vector"
        region_rep = np.tile(trust_region, (self.N, 1))
        lb = (self.nodes - region_rep).flatten()
        ub = (self.nodes + region_rep).flatten()

        # overwrite initial position
        lb[0:3] = self.rv0[0:3]
        ub[0:3] = self.rv0[0:3]
        return Bounds(lb=lb, ub=ub, keep_feasible=False)

