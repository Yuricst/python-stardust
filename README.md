# stardust

> The Initial Guess is Random Noise

Stardust is a suite of two-stage optimization scheme for trajectory design in Newtonian dynamics [1]. 

The scheme makes use of the following assumptions:
- the control actions are discretized to $N$ points, and are assumed to be impulsive (i.e. instantaneous change in velocity)

The speed of the algorithm is dependent on $N$, since there is a Jacobian to be computed & least-squares problem with sizes that scale linearly with $N$. 
In practice, $N = 30 \sim 50$ is "manageable" (solves in the order of 10s of seconds using RK4). 


## Quick start

1. Clone this repository and `cd` to its root

2. Make sure the local python environemnt has dependencies: `cvxpy`, `numpy`, `scipy`, `matplotlib`, `numba`

3. Run tests with `pytest`

```bash
pytest tests
```

## How to use it

The package provides the following dynamics models:

- CR3BP

If you want to use your own dynamics model, the only thing you need to implement is the dynamics function, which computes the state and state-transition matrix (STM) derivatives. The function should have the following signature:

```python
def eom_stm(t, x_stm, *args):
    # unpack state and STM
    x = x_stm[:nx]                                        # state
    STM = x_stm[nx:].reshape(nx,nx)                       # STM

    # compute eom & store into 1D array to be returned
    deriv_x_stm = np.zeros(nx + nx*nx,)
    deriv_x_stm[:nx] = ...                                # state-derivative
    deriv_x_stm[nx:] = (Jacobian @ STM).reshape(nx*nx,)   # STM derivatives
    return deriv_x_stm
```

where `nx` is the dimension of the state. 


<p align="center">
  <img src="./tests/twostage_cr3bp_example.png" width="800" title="transfer">
</p>


## TODO

- [ ] Two-body dynamics
- [ ] Option to neglect initial and/or final maneuver cost (weighted least-squares)


## References

[1] N. L. Parrish, “Low Thrust Trajectory Optimization in Cislunar and Translunar Space,” University of Colorado, 2018.