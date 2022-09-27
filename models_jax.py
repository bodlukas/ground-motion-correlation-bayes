# -------------------
# Models: JAX implementation
# -------------------
'''
This script contains functions to:
- compute distance / dissimilarity metrics from inputs
- evaluate the correlation models for a fixed set of parameters
- probabilistic correlation models that specify the 
    prior parameter distributions and the probabilistic structure

For a detailed explanation see notebook:
introductory_example_estimation.ipynb

The models are built in numpyro,
which in turn makes use of JAX  
to enable high-performance MCMC computations.
Therefore it is necessary to use jax.numpy instead
of numpy functions. 

For an application of the estimated models,
numpy implementations of the models are sufficient 
(see models_numpy.py).
'''

import numpy as np

import numpyro as numpyro
import jax as jax

# Change JAX default from float32 to float64
jax.config.update("jax_enable_x64", True)

# Import some functions directly for convenience
import jax.numpy as jnp
import numpyro.distributions as dist

# -------------------
# Distance metrics
# -------------------
'''
See models_numpy.py for documentation.
'''

def getEucDistanceFromPolar(X):
    sq_dist = (jnp.square(jnp.reshape(X[:,0],[-1,1])) + 
               jnp.square(jnp.reshape(X[:,0],[1,-1])) - 
        2 * (jnp.reshape(X[:,0],[-1,1]) * jnp.reshape(X[:,0],[1,-1])) * 
        jnp.cos(jnp.abs(jnp.reshape(X[:,1],[1,-1]) - jnp.reshape(X[:,1],[-1,1]) )) )
    sq_dist = jnp.clip(sq_dist, 0, np.inf)
    dist_mat = jnp.sqrt(sq_dist)
    dist_mat = dist_mat.at[jnp.diag_indices_from(dist_mat)].set(0.0)
    return dist_mat

def getAngDistanceFromPolar(X):
    cos_angle = jnp.cos( jnp.abs(jnp.reshape(X[:,1],[1,-1]) - 
                                jnp.reshape(X[:,1],[-1,1]) ))
    dist_mat =  jnp.arccos(jnp.clip(cos_angle, -1, 1))
    return dist_mat

def getSoilDissimilarity(X):
    sq_dist = jnp.square(jnp.reshape(X[:,2], [-1,1]) - jnp.reshape(X[:,2], [1,-1]))
    sq_dist = jnp.clip(sq_dist, 0, np.inf)
    dist_mat = jnp.sqrt(sq_dist)
    return dist_mat

# -------------------
# Deterministic correlation functions
# -------------------

'''
See models_numpy.py for documentation.
'''

def rhoE(X, LEt, gammaE, nugget=1e-6):
    distE = getEucDistanceFromPolar(X)
    K = jnp.exp(-1.0 * jnp.multiply(jnp.power(distE, gammaE), 1.0/LEt))

    K = K.at[jnp.diag_indices(X.shape[0])].add(nugget)
    return K

def rhoEA(X, LEt, gammaE, LA, nugget=1e-6):

    distE = getEucDistanceFromPolar(X)
    KE = jnp.exp(-1.0 * jnp.multiply(jnp.power(distE, gammaE), 1.0/LEt))

    distA = getAngDistanceFromPolar(X)
    distAdeg = jnp.multiply(distA, 180.0/np.pi)
    KA = ((1 + jnp.multiply(distAdeg, 1.0/LA)) * 
          jnp.power(1 - jnp.multiply(distAdeg, 1.0/180), 180/LA))
    
    K = KE * KA
    K = K.at[jnp.diag_indices(X.shape[0])].add(nugget)
    return K

def rhoEAS(X, LEt, gammaE, LA, LS, w, nugget=1e-6):
    
    distE = getEucDistanceFromPolar(X)
    KE = jnp.exp(-1.0 * jnp.multiply(jnp.power(distE, gammaE), 1.0/LEt))
    
    distA = getAngDistanceFromPolar(X)
    distAdeg = jnp.multiply(distA, 180.0/np.pi)
    KA = ((1 + jnp.multiply(distAdeg, 1.0/LA)) * 
          jnp.power(1 - jnp.multiply(distAdeg, 1.0/180), 180/LA))
    
    distS = getSoilDissimilarity(X)
    KS = jnp.exp(-1.0 * jnp.multiply(distS, 1.0/LS))
    
    K = KE * (w * KA + (1-w) * KS)
    K = K.at[jnp.diag_indices(X.shape[0])].add(nugget)
    return K

# -------------------
# Probabilistic correlation models
# -------------------

def modelE(X, eqids, z):
    # Define Prior Distributions
    LE = numpyro.sample("LE", dist.InverseGamma(concentration=2, rate=30))
    gamma2 = numpyro.sample("gamma2", dist.Beta(2,2))

    # Compute transformed parameters
    gammaE = numpyro.deterministic("gammaE", 2.0*gamma2)
    LEt = numpyro.deterministic("LEt", jnp.power(LE, gammaE))

    # Specify observational model for each event
    z = [numpyro.sample("z_{}".format(eqid), 
              dist.MultivariateNormal(0, rhoE(X[i], LEt, gammaE)), 
              obs = z[i]) for i,eqid in enumerate(eqids)]

def modelEA(X, eqids, z):
    # Define Prior Distributions
    LE = numpyro.sample("LE", dist.InverseGamma(concentration=2, rate=30))
    gamma2 = numpyro.sample("gamma2", dist.Beta(2,2))
    LAt = numpyro.sample("LAt", dist.Gamma(2, 0.25))

    # Compute transformed parameters
    gammaE = numpyro.deterministic("gammaE", 2.0*gamma2)
    LEt = numpyro.deterministic("LEt", jnp.power(LE, gammaE))
    LA = numpyro.deterministic("LA", 180/(4.0 + LAt))

    # Specify observational model for each event
    z = [numpyro.sample("z_{}".format(eqid), 
              dist.MultivariateNormal(0, rhoEA(X[i], LEt, gammaE, LA)), 
              obs = z[i]) for i,eqid in enumerate(eqids)]

def modelEAS(X, eqids, z):
    # Define Prior Distributions
    LE = numpyro.sample("LE", dist.InverseGamma(concentration=2, rate=30))
    gamma2 = numpyro.sample("gamma2", dist.Beta(2,2))
    LAt = numpyro.sample("LAt", dist.Gamma(2, 0.25))
    LS = numpyro.sample("LS", dist.InverseGamma(2, 100))
    w = numpyro.sample("w", dist.Beta(2,2))

    # Compute transformed parameters
    gammaE = numpyro.deterministic("gammaE", 2.0*gamma2)
    LEt = numpyro.deterministic("LEt", jnp.power(LE, gammaE))
    LA = numpyro.deterministic("LA", 180/(4.0 + LAt))

    # Specify observational model for each event
    z = [numpyro.sample("z_{}".format(eqid), 
              dist.MultivariateNormal(0, rhoEAS(X[i], LEt, gammaE, LA, LS, w)), 
              obs = z[i]) for i,eqid in enumerate(eqids)]