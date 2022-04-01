#!/usr/bin/env python3

import scipy
import time
import numpy as np
from scipy.spatial import cKDTree as KDTree
from tools import minimizeWeights
from tools import getAnW_firstDerivs


def interpolate(y,z,f,ynew,znew,yPeriodic=0.0,K=31,R=None,verbose=True):
    '''
    Find zeroth derivatives of f at any points in a point cloud.
    
    The derivaties are calculated at *new* points, different from the data points
    y,z,f are 1D arrays of the known field
    ynew, znew are 1D arrays of the field to which we interpolate
    If yPeriodic>0., points will be copied by that distance in y
    K is the max number of neighbors. Will use fewest possible tho
    '''
    
    K=int(K)
    Nnew=len(ynew)
    
    # if the domain is periodic in y:
    if yPeriodic>0.0:
        ycut=np.min(y)+(yPeriodic/2.)
        sel0=(y<ycut)
        sel1=(y>ycut)
        
        y=np.append(np.append(y,y[sel0]+yPeriodic),y[sel1]-yPeriodic)
        z=np.append(np.append(z,z[sel0]),z[sel1])
        f=np.append(np.append(f,f[sel0]),f[sel1])
        
    # get distances and integers of K points near every point:
    tree = KDTree( list(zip( y, z )) )  # build the tree
    queries = tree.query( list(zip(ynew,znew)) , K) 
    KInds = queries[1] # the integers, distances are in [0]. matrix of Nf by K indices
    
    # get matrix R
    if np.any(R==None):
        t0=time.time()
        R=np.zeros((Nnew,3,K))
        maxmaxK=0
        for ip in range(Nnew): # only over the points for which I wanna find the deriv
            I=KInds[ip,:] # indeces of the K neighbor points
            A,W=getAnW_firstDerivs(y[I],z[I],ynew[ip],znew[ip])
            A,W,Keff=minimizeWeights(A,W,ynew[ip],znew[ip])
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
            
    # assemble the known part of the system
    known=np.zeros((Nnew,K,1))
    for ip in range(Nnew): # only over the points for which I wanna find the deriv
        I=KInds[ip,:] # indeces of the K neighbor points
        known[ip,:,0]=f[I]
        
    # solve the system:
    out=np.matmul(R,known) # elementise matrix multiplication. matrices are in the last dimensions
    fnew = np.squeeze(out[:,0])
    
        
    return fnew