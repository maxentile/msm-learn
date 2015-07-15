Open questions in molecular kinetics can be cast as machine learning tasks with constraints/assumptions imposed by physics.

# Transfer operator
Most generally, we want to model the dynamical propagator / transfer operator of a molecular system through its phase space. This object is a density $\rho(x,y,\tau)$ which is the probability density of the system occupying a point $y$ at time $t_0 + \tau$ given that the system occupied the point $x$ at time $t_0$.

For deterministic dynamics, this density is a dirac delta function: the position at any time in the future is exactly specified by its initial condition.

For stochastic dynamics, this density is more interesting.

If we consider the molecule of interest adiabatically decoupled from the 

## Simplifying assumptions
### Metastable sets
The free energy landscapes of biomolecules are frequently dominated by a relatively small number of long-lived states, and the system only slowly interconverts between these states. This motivates simplifying the dynamics as a jump process between these long-lived states.

###
