#!/usr/bin/env python3


import numpy as np
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src')
from meshlessFish import solvePoisson

def linspace2D(x0,y0,x1,y1,n):
    
    dx=x1-x0
    dy=y1-y0
    dl=1./(n-1.)
    
    x=np.zeros(n)
    y=np.zeros(n)
    x[0]=x0
    y[0]=y0
    x[n-1]=x1
    y[n-1]=y1
    
    for i in range(1,n-1):
        x[i]=x[i-1] + dl*dx
        y[i]=y[i-1] + dl*dy
    
    return x,y

def stackLinspaces2D(X,Y,m):
    # corner points X,Y. number of points per segment in N
    # m is a factor to control the density
    d=( (X[1]-X[0])**2. + (Y[1]-Y[0])**2. )**0.5
    x,y=linspace2D(X[0],Y[0],X[1],Y[1],int(d*m))
    nseg=len(X)
    for i in range(1,nseg-1):
        d=( (X[i+1]-X[i])**2. + (Y[i+1]-Y[i])**2. )**0.5
        x0,y0=linspace2D(X[i],Y[i],X[i+1],Y[i+1],int(d*m))
        x=np.append(x[:-1],x0)
        y=np.append(y[:-1],y0)
    return x,y


# =============================================================================
# build domain boundary
# =============================================================================


X=[0.0, 1.0, 1.0, 0.0, 0.0]
Y=[0.0, 0.0, 1.2, 1.2, 0.0]
xb,yb=stackLinspaces2D(X,Y,m=100.) # control density of border points here
xb=xb[1:]; yb=yb[1:] # remove periodic point

plt.figure()
plt.plot(xb,yb,'k-')
plt.axis('scaled')


# =============================================================================
# build a very simple point cloud for this example
# =============================================================================

xgv=np.linspace(0,1.,51); ygv=np.linspace(0.,1.2,61)
Xf,Yf=np.meshgrid(xgv, ygv,indexing='ij')
x=Xf.ravel(); y=Yf.ravel()

path = mpltPath.Path(list(zip(xb,yb)))
isin = path.contains_points(list(zip(x,y)),radius=-np.diff(xgv)[0])
# the radius option makes sure that no points are on the boundary
# the poisson solver blows up if a field point coincides with a boundary point
x=x[isin]; y=y[isin] # throw out points outside of the domain

plt.plot(x,y,'g+',markersize=0.5)
plt.savefig('example0_domain.pdf',bbox_inches='tight')


# =============================================================================
# solve some poisson equation on this domain
# =============================================================================

n=len(x)
nb=len(xb)
dxzero=np.zeros(n,dtype=bool) # no dfdx=0 boundaries
dyzero=np.zeros(n,dtype=bool) # no dfdy=0 boundaries
fb=np.zeros(nb) # set f=0 for all boundary points
s=(x**2 + y**2) # the right-hand-side of the poisson equation

f,R=solvePoisson(x,y,s,dxzero,dyzero, xb,yb,fb,
                              yPeriodic=0.0, # no periodic boundary here
                              K=31, # max number of neighbors used
                              alpha=0.5, # relaxation
                              TOL=1e-10,maxit=1e6,
                              verbose=True)

print('max(abs(f))=%.10f' %np.max(np.abs(f)))
#%%
# simple plot
plt.figure()
plt.set_cmap('Reds_r')
plt.tricontourf(x,y,f,np.linspace(-0.06,0.0,11)[:-1],extend='min')
plt.plot(xb,yb,'k-')
plt.axis('scaled')
plt.axis('off')
plt.title('$\\nabla^2 f= x^2+y^2$\nwith $f=0$ on the boundary')
plt.savefig('example0_result.png',dpi=300,bbox_inches='tight')


print('Done.')
