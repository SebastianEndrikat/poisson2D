
import sys
import numpy as np


def printReplace(mystr,done=False):
    sys.stdout.write('\r')
    sys.stdout.write(mystr)
    sys.stdout.flush()
    if done: sys.stdout.write('\n') # line break after finished
    return


def getAnW_poisson(x,y,x0,y0):
    '''
    Define the coefficient matrix A and weight matrix W for the poisson solve.
    '''
    
    # init
    K=len(x)
    A = np.zeros((K+2,6))
    W = np.zeros((K+2,K+2))
    
    # weighting of neighbors ... find a balance between including enough 
    # points and minimizing spatial averaging
    alpha=0.1 
    
    # distance between the two closest points, a measure of the local grid size:
    dx=x[0]-x[1]
    dy=y[0]-y[1]
    d1=(dx**2. + dy**2.)**0.5
    
    for i in range(K):
        dx = x[i]-x0
        dy = y[i]-y0
        A[i+2,0] = 1.0
        A[i+2,1] = dx
        A[i+2,2] = dy
        A[i+2,3] = 0.5*dx**2. 
        A[i+2,4] = 0.5*dy**2.
        A[i+2,5] = dx*dy
        di=(dx**2. + dy**2.)**0.5
        W[i+2,i+2] = (alpha+1.)/(alpha+(di/d1)**2.)
    A[0,3]=1.0 # d2udy2 ... first entry of known-vector is the poisson rhs, known[0]=s
    A[0,4]=1.0 # d2udz2
#    A[1,1]=1.0 # in the event that dfdy= known[1]
#    A[1,2]=1.0 # in the event that dfdz= known[2]
    W[0,0]=1.0
    W[1,1]=1.0
    return A,W


def getAnW_firstDerivs(x,y,x0,y0):
    '''
    Define the coefficient matrix A and weight matrix W of the system that
    can be solved for the first derivatives. Second derivatives are not included
    here, hence A is only K by 3, not K by 6.
    '''
    K=len(x)
    A = np.zeros((K,3))
    W = np.zeros((K,K))
    alpha=1.
    dx=x[0]-x[1]
    dy=y[0]-y[1]
    eps=alpha*((dx*dx)+(dy*dy))
    for i in range(K):
        dx = x[i]-x0
        dy = y[i]-y0
        A[i,0] = 1.0
        A[i,1] = dx
        A[i,2] = dy
        W[i,i] = eps/((dx*dx)+(dy*dy)+eps)
    return A,W


def minimizeWeights(Afull,Wfull,x0,y0):
    '''
    Reduce number of neighbors to minium by setting some to zero.
    
    Pass A and W with all neighbors.
    The point for which this is one (x0,y0) is only used to print error msg.
    '''
    
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