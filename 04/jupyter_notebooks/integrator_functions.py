import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from sympy import *
from tqdm import tqdm

def acf(data,n):
    data = data[0:n]
    
    # Mean
    mean = np.mean(data)

    # Variance
    var = np.var(data)

    # Normalized data
    ndata = data - mean

    acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:] 
    acorr = acorr / var / len(ndata)
    
    return acorr
   
def euler_integrator_step(Q, q, P, p, M, m, der_V, k_spring, dt):
        # force on bath particles
        f         = k_spring * ( Q - q )

        # force big particle
        F         = - der_V(Q) - np.sum( f )

        q  =  q + p / m * dt + f / m * dt ** 2
        Q    =  Q   + P   / M * dt + F / M * dt ** 2

        p  =  p + f * dt
        P  =  P + F * dt
        
        return Q, q, P, p
    
def vel_verlet_integrator_step(Q, q, P, p, M, m, der_V, k_spring, dt):
    
        # force on bath particles at time t_k
        f         = k_spring * ( Q - q )

        # force big particle      at time t_k
        F         = - der_V(Q) - np.sum( f )

        q  =  q + p / m * dt + 0.5 * f / m * dt ** 2
        Q    =  Q   + P   / M * dt + 0.5 * F / M * dt ** 2
        
        # force on bath particles at time t_k + 1
        f1        = k_spring * (Q - q)
        
        # force on big particle at time t_k + 1
        F1        = - der_V(Q) - np.sum( f )

        p  =  p + 0.5 * ( f + f1 ) * dt
        P    =  P   + 0.5 * ( F + F1 ) * dt
        
        return Q, q, P, p
    
def leap_frog_integrator_step(Q, q, P, p, M, m, der_V, k_spring, dt):

        # force 
        f       = k_spring * (Q - q)
        F       = - der_V(Q) - np.sum( f ) 

        # half step in momenta   
        ph = p + 0.5 * dt * f
        Ph = P   + 0.5 * dt * F

        # position step
        q    =  q + p / m * dt + 0.5 * f /m * dt**2
        Q      =  Q   + P / M * dt   + 0.5 * F /M * dt**2 

        # force 
        f       = k_spring * ( Q - q )
        F       = - der_V( Q) - np.sum( f ) 

        # half step in momenta
        p =  ph + 0.5 * f * dt
        P   =  Ph + 0.5 * F * dt
        
        return Q, q, P, p
    
def langevin_bbk_step(Q, P, M, gamma, beta, der_V, dt, R):
    L = 1 / (1 + 0.5 * gamma*dt)
    
    # Deterministic force
    F  =  - der_V(Q)
    
    # Random force 
    #R  =  np.random.normal()
    
    # update p_{n+1/2}
    Phalf = ( 1 - 0.5 * gamma * dt ) * P + 0.5 * F * dt + 0.5 * np.sqrt( 2 / beta * dt * gamma * M ) * R
    
    # update q_{n+1}
    Q  =  Q + Phalf / M * dt
    
    # Recalculate deterministic force (but not the random force)
    F  =  - der_V(Q)
    
    # update p_{n+1}
    P = ( Phalf + 0.5 * F * dt + 0.5 * np.sqrt( 2 / beta * dt * gamma * M ) * R ) / ( 1 + 0.5 * gamma * dt ) 

    return Q, P

def langevin_isp_step(Q, P, M, gamma, beta, der_V, dt , tau_lng, c1, c2, c3, R):

    # Deterministic force
    F  =  - der_V(Q)
    
    # Random force 
    #R  =  np.random.normal()
    
    # velocity
    vel = c1 * P / M + c2 * F / M + c3 * R
    
    # update q_{n+1}
    Q  =  Q + vel * dt

    # update p_{n+1}
    P = vel * M

    return Q, P
    