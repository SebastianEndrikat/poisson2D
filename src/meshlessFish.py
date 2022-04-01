#!/usr/bin/env python3

import scipy
import time
import numpy as np
from scipy.spatial import cKDTree as KDTree
from tools import printReplace
from tools import minimizeWeights
from tools import getAnW_poisson


def iteratePoisson(f,knownConst,R,KInds,ghostMap,Nf,K,alpha,maxit,TOL,verbose):
    '''
    Keep solving the system until converged. 
    '''
    
    knownf=np.zeros((Nf,K+2,1)) # the part of the known vector that is updated as f changes
    fmax=np.max(np.abs(f))
    ll=-1; res=9e9
    while ll<maxit and res>TOL:
        ll+=1
#        fold=np.copy(f)
        fmax0=fmax
        
        for ip in range(Nf): # only over the points for which I wanna find the deriv
            I=KInds[ip,:] # indeces of the K neighbor points
            knownf[ip,2:,0]=f[I]
        
        out=np.matmul(R,knownf+knownConst) # elementise matrix multiplication. matrices are in the last dimensions
        f[:Nf]=alpha*np.squeeze(out[:,0]) + (1.-alpha)*f[:Nf] # update values
        f[Nf:]=f[ghostMap[Nf:]] # update the ghost values
        
#        res=np.mean((f-fold)**2.)
        fmax=np.max(np.abs(f))
        res=np.abs(fmax-fmax0)/fmax
        if verbose:
            if np.mod(ll,100)==0:
                avgf=np.mean(f) # not a proper integral tho!
                printReplace('ll=%i, residual = %.6e, avgf=%.6e' %(ll,res,avgf))
    if verbose:
        printReplace('',done=True)
    return f,res

def solvePoisson(y,z,s,dyzero,dzzero,yb,zb,fb,f0=None,
                 yPeriodic=0.0,K=31,alpha=0.5,TOL=1e-10,maxit=1e6,
                 R=None,verbose=True):
    '''
    d2fdy2 + d2fdz2 = s ... solve for f
    
    y,z,s are 1D arrays
    s is the RHS of the poisson equ. That is, a constant source term.
    
    Zero-gradient boundaries:
    dyzero and dzzero are bool arrays of f.shape,
    they are true where the respective gradient is to be zero and false elsewhere
    
    Dirichlet boundaries:
    yb,zb,fb are 1D arrays for fixed boundaries. The given fb will not change.
    Points must not be the exact same as some in y,z. 
    ie. all points in ([y, yb], [z, zb]) must be unique
    Otherwise the closest neighbor has no distance and we get nans.
    
    if yPeriodic>0., points will be copied by that distance in y
    
    f0 is an initial field, optional
    
    K is the max number of neighbors. Will use fewest possible tho so cranking 
    this up only slows down the computation without changing the solution.
    The actually used number maxmaxK is printed. Next time around it would be 
    OK to set K to that number, which could save computational cost. The 
    algorithm might work for an even lower number, but then the result would be 
    wrong. So always try with very high values and move down from there, not
    up from lower values of K until it works, that could give wrong results.
    
    alpha is the relaxation factor. Must be positive and probably best <1.0
    
    final residual < TOL or number of iterations < maxit
    
    Could pass R from a previous run on the same mesh to save time
    '''
    
    K=int(K)
    Nf=len(y) # number of field points
    Nb=len(fb)# number of boundary points
    if np.any(f0==None):
        f=np.zeros(Nf) # initial field values
    else:
        f=f0
    
    # append dirichlet boundary, then potentially mirror for periodic boundary in y
    # dirichlet boundary points dont have zero gradients and will not be solved for
    # so whatever value is in fb will stay like that
    y=np.append(y,yb)
    z=np.append(z,zb)
    f=np.append(f,fb)
    
    # create ghost points for periodic boundary if yPeriodic>0.
    ghostMap=np.arange(Nf+Nb,dtype=int) # the first points are mapped to themselves
    if yPeriodic>0.0:
        ycut=np.min(y)+(yPeriodic/2.)
        sel0=(y<ycut)
        sel1=(y>ycut)
        
        y=np.append(np.append(y,y[sel0]+yPeriodic),y[sel1]-yPeriodic)
        z=np.append(np.append(z,z[sel0]),z[sel1])
        f=np.append(np.append(f,f[sel0]),f[sel1])
        ghostMap=np.append(np.append(ghostMap,ghostMap[sel0]),ghostMap[sel1])
        # added points are mapped to original points that will be updated
        # so now f=f[ghostMap] updates the ghost values if f[:Nf] have changed
    
    # get distances and integers of K points near every point:
    t0=time.time()
#    tree = KDTree( zip( y, z ) )  # python2
#    KDists,KInds = tree.query(zip(y,z) , K) # python2
    tree = KDTree( np.vstack((y,z)).T ) # python3
    KDists,KInds = tree.query(np.vstack((y,z)).T , K) # python3
    t1=time.time()
    if verbose:
        print('Found neighbors in %.8f seconds.' %(t1-t0) )
    
    # get matrix R
    if np.any(R==None): # didnt pass an R from a previous run
        t0=time.time()
        R=np.zeros((Nf,6,K+2))
        maxmaxK=0
        nUsedNeighbors=np.zeros(Nf)
        for ip in range(Nf): # only over the points for which I wanna find the deriv
            I=KInds[ip,:] # indeces of the K neighbor points
            A,W=getAnW_poisson(y[I],z[I],y[ip],z[ip])
            if dyzero[ip]: # zero-gradient BC
                A[1,1]=1.0 # gradient is whatever we put in the known-array 
                A[0,3]=0.
                A[0,4]=0. # we define a gradient instead of a rhs
            if dzzero[ip]: # zero-gradient BC
                A[1,2]=1.0 
                A[0,3]=0.
                A[0,4]=0. # we define a gradient instead of a rhs
            A,W,Keff=minimizeWeights(A,W,y[ip],z[ip]) # Keff includes the two constraints, not just neighbors!
            maxmaxK=np.max([maxmaxK,Keff])
            nUsedNeighbors[ip]=Keff-2 # two constraints
            R[ip,:,:]=scipy.matmul(scipy.matmul(scipy.linalg.inv(scipy.matmul(scipy.matmul(A.T,W),A)),A.T),W)
        t1=time.time()
        if verbose:
            print('Got R in %.8f seconds.' %(t1-t0) )
            print('Highest number of neighbors used anywhere: %i' %np.max(nUsedNeighbors) )
            print('Lowest  number of neighbors used anywhere: %i' %np.min(nUsedNeighbors) )
            print('Average number of neighbors used: %f' %np.mean(nUsedNeighbors) )
    
    # assemble the part of the known vectors that uses the constant source term
    knownConst=np.zeros((Nf,K+2,1))
    for ip in range(Nf): # only over the points for which I wanna find the deriv
        I=KInds[ip,:] # indeces of the K neighbor points
        if (not dyzero[ip]) and (not dzzero[ip]): # define RHS for this point instead of gradient BC
            knownConst[ip,0,0]= s[ip]
        knownConst[ip,1,0]= 0.0 # gradient value
    
    
    t0=time.time()
    f,res=iteratePoisson(f,knownConst,R,KInds,ghostMap,Nf,K,alpha,maxit,TOL,verbose)
    t1=time.time()
    
    if verbose:
        # solve the full system rather than just f:
        knownf=np.zeros((Nf,K+2,1)) # the part of the known vector that is updated as f changes
        for ip in range(Nf): # only over the points for which I wanna find the deriv
            I=KInds[ip,:] # indeces of the K neighbor points
            knownf[ip,2:,0]=f[I]
        out=np.matmul(R,knownf+knownConst) # elementise matrix multiplication. matrices are in the last dimensions
        
        print('Calculated f in %.6f minutes' %((t1-t0)/60.) )
        print('Final residual = %.6e' %res)
        
        d2fdy2=np.squeeze(out[:,3])
        d2fdz2=np.squeeze(out[:,4])
        err=np.mean((d2fdy2+d2fdz2-s)**2.)**0.5
        print('Average error between LHS and RHS = %.6e' %err )
        
        if np.sum(dyzero)>0:
            sel,=np.where(dyzero==True)
            print('Average dfdy=0 = %.6e' %np.mean((np.squeeze(out[:,1])[sel])**2.)**0.5 )
        if np.sum(dzzero)>0:
            sel,=np.where(dzzero==True)
            print('Average dfdz=0 = %.6e' %np.mean((np.squeeze(out[:,2])[sel])**2.)**0.5 )
    
    
    return f[:Nf], R # could also return gradients