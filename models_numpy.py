# -------------------
# Models: numpy implementation
# -------------------

'''
This file contains several functions used to compute 
- distance metrics from inputs 
- correlation matrices for specified correlation models

These functions are implemented using numpy and can be
used to apply spatial correlation models for regional
seismic risk simulation studies with minimal additional 
packages required. 

For Bayesian inference we used numpyro which required 
several additional packages and a (virtual) linux os.
The corresponding models are specified in models_jax.py

Todo:
    - Test that the given inputs are valid. 
        (e.g., the lengthscale is positive)
        and include errors if not valid.

'''

import numpy as np

# -------------------
# Distance metrics
# -------------------

def getEucDistance(X):
    '''
    Computes a matrix with the Euclidean distance from 
    each input x in X to all other inputs in X.
    
    Inputs
    ------
    X: Input matrix (nk x 3)
        Column 0: Epicentral distance in km
        Column 1: Epicentral azimuth in radians
        Column 2: Vs30 values in m/s

    Outputs
    ------
    dist: Matrix with distance from each row in X to all 
        the other rows (nk x nk)
    '''
    sq_dist = (np.square(np.reshape(X[:,0],[-1,1])) + 
        np.square(np.reshape(X[:,0],[1,-1])) - 
        2 * (np.reshape(X[:,0],[-1,1]) * np.reshape(X[:,0],[1,-1])) * 
        np.cos(np.abs(np.reshape(X[:,1],[1,-1]) - np.reshape(X[:,1],[-1,1]) )) )
    sq_dist = np.clip(sq_dist, 0, np.inf)
    dist = np.sqrt(sq_dist)
    dist[np.diag_indices_from(dist)] = 0.0
    return dist

def getAngDistance(X):
    '''
    Computes a matrix with the angular distance from 
    each input x in X to all other inputs in X.

    Inputs
    ------
    X: Input matrix (nk x 3)
        Column 0: Epicentral distance in km
        Column 1: Epicentral azimuth in radians
        Column 2: Vs30 values in m/s

    Outputs
    ------
    dist: Matrix with Angular distance from each 
        row in X to all the other rows (nk x nk)
    '''
    cos_angle = np.cos( np.abs(np.reshape(X[:,1],[1,-1]) - 
                                np.reshape(X[:,1],[-1,1]) ))
    dist =  np.arccos(np.clip(cos_angle, -1, 1))
    dist[np.diag_indices_from(dist)] = 0.0
    return dist

def getSoilDissimilarity(X):
    '''
    Computes a matrix with the soil dissimilarity from 
    each input x in X to all other inputs in X.

    Inputs
    ------
    X: Input matrix (nk x 3)
        Column 0: Epicentral distance in km
        Column 1: Epicentral azimuth in radians
        Column 2: Vs30 values in m/s

    Outputs
    ------
    dist: Matrix with soil dissimilarities from each 
        row in X to all the other rows (nk x nk)
    '''
    sq_dist = np.square(np.reshape(X[:,2], [-1,1]) - np.reshape(X[:,2], [1,-1]))
    sq_dist = np.clip(sq_dist, 0, np.inf)
    dist = np.sqrt(sq_dist)
    dist[np.diag_indices_from(dist)] = 0.0
    return dist

# -------------------
# Deterministic correlation functions
# -------------------

def rhoE(X, LE, gammaE):
    '''
    Computes the correlation matrix using the
    isotropic model E conditional on parameters
    (LE, gammaE)

    Inputs
    ------
    X: Input matrix (nk x 3)
        Column 0: Epicentral distance in km
        Column 1: Epicentral azimuth in radians
        Column 2: Vs30 values in m/s
    LE: Euclidean lengthscale in km
    gammaE: Exponent

    Outputs
    ------
    K: Correlation matrix (nk x nk)
    '''    
    distE = getEucDistance(X)
    K = np.exp(- np.power(distE/LE, gammaE))
    return K

def rhoEA(X, LE, gammaE, LA):
    '''
    Computes the correlation matrix using the
    model EA conditional on parameters
    (LE, gammaE, LA)

    Inputs
    ------
    X: Input matrix (nk x 3)
        Column 0: Epicentral distance in km
        Column 1: Epicentral azimuth in radians
        Column 2: Vs30 values in m/s
    LE: Euclidean lengthscale in km
    gammaE: Exponent
    LA: Angular lengthscale in degrees

    Outputs
    ------
    K: Correlation matrix (nk x nk)
    '''   
    distE = getEucDistance(X)
    KE = np.exp(- np.power(distE/LE, gammaE))  

    distA = getAngDistance(X)
    distAdeg = distA * 180/np.pi
    KA = (1 + distAdeg/LA) * np.power(1 - distAdeg/180, 180/LA)

    K = KE * KA
    return K

def rhoEAS(X, LE, gammaE, LA, LS, w):
    '''
    Computes the correlation matrix using the
    model EAS conditional on parameters
    (LE, gammaE, LA, LS, w)

    Inputs
    ------
    X: Input matrix (nk x 3)
        Column 0: Epicentral distance in km
        Column 1: Epicentral azimuth in radians
        Column 2: Vs30 values in m/s
    LE: Euclidean lengthscale in km
    gammaE: Exponent
    LA: Angular lengthscale in degrees
    LS: Vs30 lengthscale in m/s
    w: weight parameter

    Outputs
    ------
    K: Correlation matrix (nk x nk)
    '''   
    distE = getEucDistance(X)
    KE = np.exp(- np.power(distE/LE, gammaE))  

    distA = getAngDistance(X)
    distAdeg = distA * 180/np.pi
    KA = (1 + distAdeg/LA) * np.power(1 - distAdeg/180, 180/LA)

    distS = getSoilDissimilarity(X)
    KS = np.exp(- distS/LS)
    
    K = KE * (w*KA + (1-w)*KS)
    return K