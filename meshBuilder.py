#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 01:01:21 2019

@author: seb
"""

import numpy as np
import os
import errno
import matplotlib.pyplot as plt

# mesh made up of triangles



class meshBuilder():
    
    def __init__(self,caseDir):
        self.caseDir=caseDir
        self.mkdir(self.caseDir)
        
        self.vertices=np.array([])
        self.cellv=np.array([]) # cell vertice numbers
        self.cellb=np.array([]) # boundaries for each cell
        self.boundaries=np.array([]) # list of the boundaries that exist
        self.boundaryType=np.array([]) # their type
        
        return
    
    def mkdir(self,thedir):
        try:
            os.makedirs(thedir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        return
    
    def addVertex(self,x,y):
        if len(self.vertices)==0: # adding very first
            self.vertices=np.array([x,y])
        else:
            self.vertices=np.vstack((self.vertices, np.array([x,y])))
        return len(self.vertices)-1 # indexing of added vertex
    
    def addCell(self,v,bc=['','','']):
        # v contains the indices of vertices that make up this cell
        if len(self.cellv)==0: # adding very first
            self.cellv=v
        else:
            self.cellv=np.vstack((self.cellv,v))
        if len(self.cellb)==0: # adding very first
            self.cellb=bc
        else:
            self.cellb=np.vstack((self.cellb,bc))
        return
    
    def addBoundary(self,name,bctype):
        # name ... a string
        # bctype='dirichlet' or 'zeroGrad'
        self.boundaries=np.append(self.boundaries,name)
        self.boundaryType=np.append(self.boundaryType,bctype)
        return
    
    def splitCell(self,c):
        # c is the index of the cell
        v0=self.vertices[self.cellv[c][0]]
        v1=self.vertices[self.cellv[c][1]]
        v2=self.vertices[self.cellv[c][2]]
        cellx=np.array([v0[0], v1[0], v2[0], v0[0]]) # append first
        celly=np.array([v0[1], v1[1], v2[1], v0[1]])
            
        A=np.zeros(3)
        for i in range(3):
            dx=cellx[i]-cellx[i+1]
            dy=celly[i]-celly[i+1]
            A[i]=(dx**2. + dy**2.)**0.5 # length of the face
        isplit=np.argmax(A) # number of the longest face
        tol=8
        sel=(np.round(A,tol)==np.round(A,tol)[isplit])
        if np.sum(sel)>1: # more than one face is the longest
            allisplit=np.where(sel)[0] # numbers of the longest faces
            n=len(allisplit)
            dys=np.zeros(n)
            for i in range(n):
                dys[i]=np.abs(celly[allisplit[i]]-celly[allisplit[i]+1])
            isplit=allisplit[np.argmin(dys)] # pick the face that is the most horizontal
        xnew=0.5*(cellx[isplit]+cellx[isplit+1])
        ynew=0.5*(celly[isplit]+celly[isplit+1])
        vinew=self.addVertex(xnew,ynew) # index of the new vertex
        bcnew=self.cellb[c][isplit] # same bc as the split face
        
        # find vertex indeces in a system where the split is between 2 and 0
        bc2=self.cellb[c][isplit]
        vi2=self.cellv[c][isplit]
        if isplit<2:
            vi0= self.cellv[c][isplit+1]
            bc0= self.cellb[c][isplit+1]
        else:
            vi0= self.cellv[c][0]
            bc0= self.cellb[c][0]
        if isplit>0:
            vi1= self.cellv[c][isplit-1]
            bc1= self.cellb[c][isplit-1]
        else:
            vi1= self.cellv[c][2]
            bc1= self.cellb[c][2]
            
        # see if a neighbor cell is affected by the split:
        # before changing the verteces of this cell
        neighc=self.getNeighborCell(c,vi2,vi0)
        
        # add the new cell:
        self.addCell([vi0,vi1,vinew],bc=[bc0,'',bcnew])
        
        # change the old cell:
        self.cellv[c][0]=vi1
        self.cellv[c][1]=vi2
        self.cellv[c][2]=vinew
        self.cellb[c][0]=bc1
        self.cellb[c][1]=bc2
        self.cellb[c][2]='' # new internal face
            
        splitcellsno=np.array([c])
        
        # split the neighbor cell using the new vertex
        if neighc>=(-1): # a neighbor exists
            vi=self.cellv[neighc] # the three indices
            isplit=np.where(vi==vi0)[0]
            # roll the indices such that isplit becomes 0
            vi=np.roll(vi,-isplit)
            bci=np.roll(self.cellb[neighc],-isplit)
            
            # add the new cell:
            self.addCell([vi[0],vinew,vi[2]],bc=[bci[0],'',bci[2]])
            
            # change the old cell:
            self.cellv[neighc][0]=vinew
            self.cellv[neighc][1]=vi[1]
            self.cellv[neighc][2]=vi[2]
            self.cellb[neighc][0]=bci[0]
            self.cellb[neighc][1]=bci[1]
            self.cellb[neighc][2]=''
            
            splitcellsno=np.array([c,neighc])
        
        return splitcellsno # the cell or cells that it split
    
    def getNeighborCell(self,c,vno0,vno1):
        # pass the index of this cell and indices of two vertices that
        # make up the face for which is neighbor is sought
        cellswiththesev =  np.where(
                np.sum(np.logical_or(
                        self.cellv==vno0,
                        self.cellv==vno1),axis=1)==2)[0]
        neighborCell=-1 # doesnt exist
        for ci in cellswiththesev:
            if ci != c:
                neighborCell=ci
        return neighborCell # index or empty array if doesnt exist
    
    def refineMeshToDelta(self,dxmax,dymax):
        count=1 # to start loop
#        while count>0:
        for i in range(2): # testing
            listofsplit=np.array([]) # the cell indeces of those that have been split
            count=0
            ncells=len(self.cellv)
            for c in range(ncells):
                if not c in listofsplit:
                    v0=self.vertices[self.cellv[c][0]]
                    v1=self.vertices[self.cellv[c][1]]
                    v2=self.vertices[self.cellv[c][2]]
                    xmax=np.max([v0[0],v1[0],v2[0]])
                    xmin=np.min([v0[0],v1[0],v2[0]])
                    ymax=np.max([v0[1],v1[1],v2[1]])
                    ymin=np.min([v0[1],v1[1],v2[1]])
                    if (xmax-xmin)>dxmax or (ymax-ymin)>dymax:
                        splitcellsno=self.splitCell(c)
                        listofsplit=np.append(listofsplit,splitcellsno)
                        count +=1
        
        return
    
    def refineMeshNtimes(self,n):
        for i in range(n):
            listofsplit=np.array([]) # the cell indeces of those that have been split
            ncells=len(self.cellv)
            for c in range(ncells):
                if not c in listofsplit:
                    splitcellsno=self.splitCell(c)
                    listofsplit=np.append(listofsplit,splitcellsno)
        return
    
    def plotMesh(self):
        ncells=len(self.cellv)
        bccolors=['k','r','b','g','m','y']
        plt.figure()
        for c in range(ncells):
            v0=self.vertices[self.cellv[c][0]]
            v1=self.vertices[self.cellv[c][1]]
            v2=self.vertices[self.cellv[c][2]]
            bcno0=0
            bcno1=0
            bcno2=0
            if self.cellb[c][0] in self.boundaries:
                bcno0=int(np.where(self.cellb[c][0]==self.boundaries)[0]) +1
            if self.cellb[c][1] in self.boundaries:
                bcno1=int(np.where(self.cellb[c][1]==self.boundaries)[0]) +1
            if self.cellb[c][2] in self.boundaries:
                bcno2=int(np.where(self.cellb[c][2]==self.boundaries)[0]) +1
            plt.plot([v0[0],v1[0]], [v0[1],v1[1]], bccolors[bcno0]+'.-')
            plt.plot([v1[0],v2[0]], [v1[1],v2[1]], bccolors[bcno1]+'.-')
            plt.plot([v2[0],v0[0]], [v2[1],v0[1]], bccolors[bcno2]+'.-')
        return
    
    def finalize(self):
        # write the mesh and the finite volume coeffs to disk
        ncells=len(self.cellv)
        
        x=np.zeros(ncells)
        y=np.zeros(ncells)
        vol=np.zeros(ncells)
        
        for c in range(ncells):
            v0=self.vertices[self.cellv[c][0]]
            v1=self.vertices[self.cellv[c][1]]
            v2=self.vertices[self.cellv[c][2]]
            cellx=np.array([v0[0], v1[0], v2[0], v0[0]]) # append first
            celly=np.array([v0[1], v1[1], v2[1], v0[1]])
            
            # centroid of triangles is the avg of the vertices
            x[c]=np.mean(cellx[:3])
            y[c]=np.mean(celly[:3])
            
            # area of triangles
            tmp=0.
            for i in range(3): # all faces
                tmp += ( (cellx[i]*celly[i+1]) - (celly[i]*cellx[i+1]) )
            vol[c]=0.5*np.abs(tmp)
        
        neighbor=np.zeros((ncells,3))
        cf=np.zeros((ncells,3)) # coefficient for the three neighbors
        cp=np.zeros(ncells) # coefficient of that cell
        
        for c in range(ncells):
            v0=self.vertices[self.cellv[c][0]]
            v1=self.vertices[self.cellv[c][1]]
            v2=self.vertices[self.cellv[c][2]]
            cellx=np.array([v0[0], v1[0], v2[0], v0[0]]) # append first
            celly=np.array([v0[1], v1[1], v2[1], v0[1]])
            
            
            # face normals
            nx=np.zeros(3)
            ny=np.zeros(3)
            for i in range(3):
                va=np.array([cellx[i  ],celly[i  ],0])
                vb=np.array([cellx[i+1],celly[i+1],0])
                vc=vb-va
                n=np.cross(vc,np.array([0,0,1])) # if positive direction
                n /= (n[0]**2. + n[1]**2. + n[2]**2. )**0.5 # normalize normal vector
                nx[i]=n[0]
                ny[i]=n[1]
                
            # face areas (lengths actually in 2D)
            A=np.zeros(3)
            for i in range(3):
                dx=cellx[i]-cellx[i+1]
                dy=celly[i]-celly[i+1]
                A[i]=(dx**2. + dy**2.)**0.5 # length of the face
                
            # neighboring cells
            for i in range(3):
                if self.cellb[c][i]=='': # internal face
                    vno0=self.cellv[c][i]
                    if i==2:
                        vno1=self.cellv[c][0]
                    else:
                        vno1=self.cellv[c][i+1]
                    neighbor[c,i]=self.getNeighborCell(self,c,vno0,vno1)
                else: # boundary face
                    boundno=np.where(self.cellb[c][i]==self.boundaries)[0]
                    if self.boundaryType[boundno]=='zeroGrad':
                        # point to cell 0, the value wont be used anyway as
                        # the coefficient will be zero for zeroGrad
                        neighbor[c,i]=0
                    elif self.boundaryType[boundno]=='dirichlet':
                        # the how-many-th dirichlet BC is this:
                        dirichno=np.sum((self.boundaryType=='dirichlet')[:boundno])
                        # point to a ghost cell that will hold this value
                        neighbor[c,i]=ncells+dirichno
            
            
            # coefficients:
            eps=1e-6
            for i in range(3):
                if self.cellb[c][i]=='': # internal face
                    dx=x[neighbor[c,i]] - x[c]
                    dy=y[neighbor[c,i]] - y[c]
                    if np.abs(dx)>eps:
                        cf[c,i] += (-A[i]*nx[i]/dx/vol[i])
#                        cp[c]   += (-A[i]*nx[i]/dx/vol[i])
                    if np.abs(dy)>eps:
                        cf[c,i] += (-A[i]*ny[i]/dy/vol[i])
#                        cp[c]   += (-A[i]*ny[i]/dy/vol[i])
                else: # boundary face
                    boundno=np.where(self.cellb[c][i]==self.boundaries)[0]
                    if self.boundaryType[boundno]=='zeroGrad':
                        cf[c,i] = 0.0
#                        cp[c]  += 0.0
                    elif self.boundaryType[boundno]=='dirichlet':
                        dx=(0.5*(cellx[i]+cellx[i+1])) - x[c]
                        dy=(0.5*(celly[i]+celly[i+1])) - y[c]
                        if np.abs(dx)>eps:
                            cf[c,i] += (-A[i]*nx[i]/dx/vol[i])
#                            cp[c]   += (-A[i]*nx[i]/dx/vol[i])
                        if np.abs(dy)>eps:
                            cf[c,i] += (-A[i]*ny[i]/dy/vol[i])
#                            cp[c]   += (-A[i]*ny[i]/dy/vol[i])
            cp[c] = np.sum(cf[c,:])
            
            # write x,y,vol, cf, cp, and a dirichlet boundary file
                            
            
        
        return
    
    

# =============================================================================
# =============================================================================
if __name__ == "__main__":
    print('Executing this as a script...')
    
    m=meshBuilder('./TI60')
    
    # define boundaries
    m.addBoundary('wall','dirichlet') # fixed to something
    m.addBoundary('top','zeroGrad') # zero gradient
   
    # add at least four points
    top=0.5/np.tan(30.*np.pi/180)
    m.addVertex(0.,top)
    m.addVertex(0.5,0.)
    m.addVertex(1.,top)
    m.addVertex(0.5,top)
    
    # connect points to cells that may have boundary conditionsa
    # gotta have at least two cells (could change code to require one some time)
    # vertices have to go counter-clockwise!
    m.addCell([0,1,3],['wall','','top'])
    m.addCell([1,2,3],['wall','top',''])
    
    # refine mesh:
#    m.refineMeshToDelta(dxmax=0.1,dymax=0.1)
    m.refineMeshNtimes(2)
    
#    m.splitCell(1)

    m.plotMesh() # at least 2 cells have to exist
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1])
#    plt.xlim([0.2,0.4])
#    plt.ylim([0.6,0.8])
#    plt.plot([m.vertices[10][0],m.vertices[34][0]], 
#             [m.vertices[10][1],m.vertices[34][1]],'g-',linewidth=5)
    
#    plt.plot(m.vertices[10][0],m.vertices[10][1],'r+')
    
#    m.finalize()