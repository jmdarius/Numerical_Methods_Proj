## Import Packages
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import timeit 
## Define System of Equations
# INPUT: Array of independent variables x = [x0,x1,...,xn]
# OUTPUT: Array of dependent Results res = [f0, f1, ..., fn]

def fs(x):
    # From http://fourier.eng.hmc.edu/e176/lectures/NM/node21.html Example 1
    # f0 = 3*x_0 - cos(x_1*x_2)--1.5
    # f1 = 4*x_0^2-625*x_1^2+2*x-1
    # f2 = 20*x_2+exp(-x_0*x_1)+9
    f0 = 3*x[0]-jnp.cos(x[1]*x[2])-3/2  # Equation 1
    f1 = 4*x[0]**2-625*x[1]**2+2*x[2]-1 # Equation 2
    f2 = 20*x[2]+jnp.exp(-x[0]*x[1])+9  # Equation 3
    res = [f0, f1, f2]                  # Create Array of Results
    return jnp.asarray(res)             # Return Jax Array

## Define Multivariate Newton Method
# INPUT: System of Functions = f, Initial Guess Array = x0, tolerance = tol, Maximum iterations = N
# OUTPUT: Solution Array = x
def multivariateNewton(f, x0, tol, N):
    x0 = jnp.asarray(x0).T          # Convert Input Array to Jax Array
    def J_inv(x):                   # Create Inverse Jacobian Function 
        jacobian = jax.jacfwd(f)    # Calculate the jacobian function from the provided systems with Forward Auto-differentiation
        J = jacobian(x)             # Calculate the Jacobian at x
        J_inv = jnp.linalg.inv(J)   # Calculate the Inverse Jacobian
        return jnp.asarray(J_inv)   # Return Inverse Jacobian at x as a Jax Array

    for i in range(1,N):            # Start Loop for Maximum Iterations
        x = jnp.subtract(x0, jnp.matmul(J_inv(x0), f(x0).T)) # Perform Newton Iteration: x_{n+1} = x_n-J^(-1)*f
        reltol = jnp.divide(jnp.linalg.norm(jnp.subtract(x,x0)),jnp.linalg.norm(x)) # Calculate: ||x_{n+1}-x_n||/||x_{n+1}||
        print(i, reltol)        # Print iteration and relTol
        if reltol < tol:        # Check for convergence
            print(x)            # Print Result
            return x            # Return Result 
        x0 = x                  # Update x0 for Next iteration
    print("Failed to converge") # Print Message if Convergence did not occur

res = multivariateNewton(fs, [1.,1.,1.], 1e-8, 20) # Perform Newton Method for System "fs" with guess  [x0,x1,x2] = [1,1,1] with tol = 1e-8 and 20 maximum iterations
print(fs(res))  # Print fs output for system