#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:52:43 2020

@author: sebastian

Python is not a snake.
and now for something completely different:
"""




import numpy as np
#import glob
#import matplotlib.pyplot as plt
#import os, errno
import sys
#from scipy.spatial import cKDTree as KDTree
import scipy
import time


def printReplace(mystr,done=False):
    sys.stdout.write('\r')
    sys.stdout.write(mystr)
    sys.stdout.flush()
    if done: sys.stdout.write('\n') # line break after finished
    return


def getAnW_firstDerivs(x,y,x0,y0):
    K=len(x)
    A = np.zeros((K,3))
    W = np.zeros((K,K))
    alpha=1.
    dx=x[0]-x[1] # distance between the two closest points
    dy=y[0]-y[1] # a measure of the local grid size
    eps=alpha*((dx*dx)+(dy*dy))
    for i in range(K):
        dx = x[i]-x0
        dy = y[i]-y0
        A[i,0] = 1.0
        A[i,1] = dx
        A[i,2] = dy
        W[i,i] = eps/((dx*dx)+(dy*dy)+eps)
    return A,W

def getAnW_poisson(x,y,x0,y0):
    K=len(x)
    A = np.zeros((K+2,6))
    W = np.zeros((K+2,K+2))
    alpha=1. # weighting of neighbors
#    alpha=0.1
    dx=x[0]-x[1] # distance between the two closest points
    dy=y[0]-y[1] # a measure of the local grid size
    eps=alpha*((dx*dx)+(dy*dy))
    for i in range(K):
        dx = x[i]-x0
        dy = y[i]-y0
        A[i+2,0] = 1.0
        A[i+2,1] = dx
        A[i+2,2] = dy
        A[i+2,3] = 0.5*dx**2. 
        A[i+2,4] = 0.5*dy**2.
        A[i+2,5] = dx*dy
        W[i+2,i+2] = eps/((dx*dx)+(dy*dy)+eps)
    A[0,3]=1.0 # d2udy2 ... first entry of known-vector is the poisson rhs, known[0]=s
    A[0,4]=1.0 # d2udz2
#    A[1,1]=1.0 # in the event of dudy= known[1]
#    A[1,2]=1.0 # in the event of dudz= known[2]
    W[0,0]=1.0
    W[1,1]=1.0
    return A,W

def minimizeWeights(Afull,Wfull,x0,y0):
    # pass A and W with all neighbors.
    # will reduce number of neighbors to minium by setting some to zero
    # point for which this is one (x0,y0) only to print error
    K,Keff=Afull.shape # try to use only Keff neighbors. Remaining will be zeroed out
    
    theDetmax = np.linalg.det(np.matmul(np.matmul(Afull.T,Wfull),Afull))
        
    # now zero out all but Keff neighbors and see if it works. if not, zero out fewer
    Keff -= 1 # cuz it will be increased again
    theDet = 9e9 # start value
    while not (np.abs(theDetmax-theDet)/np.abs(theDetmax)) <= 0.99:
        # abs(theDetmax) is necessary for the rare case where it is negative e-26 or so
        Keff += 1
        if Keff > K:
            raise ValueError('Number of necessary neighbors for full rank exceeds '+
                             'available neighbors: %i at (%.6f, %.6f)' %(K,x0,y0))
        weight=np.copy(Wfull)
        amat=np.copy(Afull)
        for n in range(Keff,K):
            weight[n,n]=0.
            amat[n,:]=0.
        theDet = np.linalg.det(np.matmul(np.matmul(amat.T,weight),amat))
    return amat,weight,Keff

def getFirstDerivs(y,z,f,yPeriodic=0.0,K=31,R=None,verbose=True):
    # find first derivatives of f at every point in a point cloud
    # y,z,f are 1D arrays
    # if yPeriodic>0., points will be copied by that distance in y
    # K is the max number of neighbors. Will use fewest possible tho
    
    K=int(K)
    Nf=len(f)
    
    if yPeriodic>0.0:
        ycut=np.min(y)+(yPeriodic/2.)
        sel0=(y<ycut)
        sel1=(y>ycut)
        
        y=np.append(np.append(y,y[sel0]+yPeriodic),y[sel1]-yPeriodic)
        z=np.append(np.append(z,z[sel0]),z[sel1])
        f=np.append(np.append(f,f[sel0]),f[sel1])
        
#    plt.plot(y,z,'.g')
    
    tree = scipy.spatial.cKDTree( zip( y, z ) )  # build the tree
    # query tree and get distances and integers of K points near every point:
    queries = tree.query(zip(y,z) , K) 
    KInds = queries[1] # the integers, distances are in [0]. matrix of Nf by K indices
    
    # get R
    if np.any(R==None):
        t0=time.time()
        R=np.zeros((Nf,3,K))
        maxmaxK=0
        for ip in range(Nf): # only over the points for which I wanna find the deriv
            I=KInds[ip,:] # indeces of the K neighbor points
            A,W=getAnW_firstDerivs(y[I],z[I],y[ip],z[ip])
            A,W,Keff=minimizeWeights(A,W,y[ip],z[ip])
            maxmaxK=np.max([maxmaxK,Keff])
            R[ip,:,:]=scipy.matmul(scipy.matmul(scipy.linalg.inv(scipy.matmul(scipy.matmul(A.T,W),A)),A.T),W)
        t1=time.time()
        if verbose:
            print('Got R in %.8f seconds. Highest number of neighbors used anywhere: %i' %((t1-t0),maxmaxK) )
        # about maxmaxK: next time around it would be OK to set K to that 
        # number, which could save computational cost. The algorithm might work
        # for an even lower number but then the result would be wrong. So 
        # always try with very high values and move down from there, not
        # up from lower values of K until it works, that could give wrong results.
        
        
    # assamble the known part of the system
    known=np.zeros((Nf,K,1))
    for ip in range(Nf): # only over the points for which I wanna find the deriv
        I=KInds[ip,:] # indeces of the K neighbor points
        known[ip,:,0]=f[I]
        
        
    t0=time.time()
    out=np.matmul(R,known) # elementise matrix multiplication. matrices are in the last dimensions
    dfdy=np.squeeze(out[:,1])
    dfdz=np.squeeze(out[:,2])
    t1=time.time()
    if verbose:
        print('Elementwise matmul in %.8f seconds' %(t1-t0))
    
#    t0=time.time()
#    dfdy=np.zeros(Nf)
#    dfdz=np.zeros(Nf)
#    for ip in range(Nf): # only looping over the non-repeated points
#        # indeces of the K neighbor points
#        I=KInds[ip,:] 
#        
#        # if I want to find the derivatives, the known LHS is just the field values of the neighbors
#        known=f[I]
#        
#        out=np.matmul(R[ip,:,:],known)
#        dfdy[ip]=out[1]
#        dfdz[ip]=out[2]
#    t1=time.time()
#    print('Looped matmul in %.8f seconds' %(t1-t0))
    
    
#    Got R in 5.39548397 seconds
#    Elementwise matmul in 0.00095487 seconds
#    Looped matmul in      0.02349186 seconds
        
    return dfdy, dfdz, R

def iteratePoisson(f,knownConst,R,KInds,ghostMap,Nf,K,alpha,maxit,TOL,verbose):
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
        f[:Nf]=alpha*np.squeeze(out[:,0]) + (1.-alpha)*f[:Nf] # updated values
#        f[:Nf]=alpha*(np.sum(R[:,0,:]*(knownf+knownConst)[:,:,0],axis=1)) + (1.-alpha)*f[:Nf] # updated values
        f[Nf:]=f[ghostMap[Nf:]] # update the ghost values
        
#        res=np.mean((f-fold)**2.)
        fmax=np.max(np.abs(f))
        res=np.abs(fmax-fmax0)
        if verbose:
            if np.mod(ll,100)==0:
                printReplace('ll=%i, residual = %.6e' %(ll,res))
    if verbose:
        printReplace('',done=True)
    return f,res

def solvePoisson(y,z,s,dyzero,dzzero,yb,zb,fb,f0=None,
                 yPeriodic=0.0,K=31,alpha=0.5,TOL=1e-10,maxit=1e6,
                 R=None,verbose=True):
    '''
    d2fdy2 + d2fdz2 = s ... solve for f
    y,z,s are 1D arrays
    s is the RHS of the poisson eq. A constant source term
    
    Zero-gradient boundaries:
    dyzero and dzzero are bool arrays of f.shape,
    they are true where the respective gradient is to be zero and false elsewhere
    
    Dirichlet boundaries:
    yb,zb,fb are 1D arrays for fixed boundaries. the given fb will not change.
    Points must not be the exact same as some in y,z. 
    ie. all points in ([y, yb], [z, zb]) must be unique
    Otherwise the closest neighbor has no distance and we get nans.
    
    if yPeriodic>0., points will be copied by that distance in y
    
    f0 is an initial field, optional
    
    K is the max number of neighbors. Will use fewest possible tho so cranking 
    this up only slows down the computation without changing the solution.
    The actually used number maxmaxK is printed. next time around it would be 
    OK to set K to that number, which could save computational cost. The 
    algorithm might work for an even lower number but then the result would be 
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
    
    t0=time.time()
    tree = scipy.spatial.cKDTree( zip( y, z ) )  # build the tree
    # query tree and get distances and integers of K points near every point:
    KDists,KInds = tree.query(zip(y,z) , K) 
    t1=time.time()
    if verbose:
        print('Found neighbors in %.8f seconds.' %(t1-t0) )
    
    # get R
    if np.any(R==None):
        t0=time.time()
        R=np.zeros((Nf,6,K+2))
        maxmaxK=0
        for ip in range(Nf): # only over the points for which I wanna find the deriv
            I=KInds[ip,:] # indeces of the K neighbor points
            A,W=getAnW_poisson(y[I],z[I],y[ip],z[ip])
            if dyzero[ip]: # zero-gradient BC
                A[1,1]=1.0 # gradient is whatever we put in the known-array
            if dzzero[ip]: # zero-gradient BC
                A[1,2]=1.0 
            A,W,Keff=minimizeWeights(A,W,y[ip],z[ip]) # Keff invludes the two constraints, not just neighbors!
            maxmaxK=np.max([maxmaxK,Keff])
            R[ip,:,:]=scipy.matmul(scipy.matmul(scipy.linalg.inv(scipy.matmul(scipy.matmul(A.T,W),A)),A.T),W)
        t1=time.time()
        if verbose:
            print('Got R in %.8f seconds. Highest number of neighbors used anywhere: %i' %((t1-t0),maxmaxK) )
    
    # assemble the part of the known vectors that uses the constant source term
    knownConst=np.zeros((Nf,K+2,1))
    for ip in range(Nf): # only over the points for which I wanna find the deriv
        I=KInds[ip,:] # indeces of the K neighbor points
        knownConst[ip,0,0]= s[ip]
        knownConst[ip,1,0]= 0.0 # gradient in either y or z
    
    
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
        err=np.mean((d2fdy2+d2fdz2-s)**2.)
        print('Average squared error between LHS and RHS = %.6e' %err )
        
        if np.sum(dyzero)>0:
            sel,=np.where(dyzero==True)
            print('Average dfdy=0 = %.6e' %np.mean(np.squeeze(out[:,1])[sel]) )
        if np.sum(dzzero)>0:
            sel,=np.where(dzzero==True)
            print('Average dfdz=0 = %.6e' %np.mean(np.squeeze(out[:,2])[sel]) )
    
    
    return f[:Nf], R # could also return gradients




