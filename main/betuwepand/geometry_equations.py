# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:36:41 2020

@author: meer_an
"""

#%% External modules

import numpy as np

#%% Own modules

#%% Functions

def shoelace(x,y):
    r"""
    The shoelace formula is a mathematical algorithm to determine the area
    of a simple polygon whose points (vertices) are described by their 
    Cartesian coordinates in the plane.
    
    See: https://en.wikipedia.org/wiki/Shoelace_formula
    
    The implemented equation is for 2 dimension x and y.
    """
    
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def intersection(x1,y1,x2,y2):
    r"""
    This function calculates the intersection(s) between two lines. Output
    is 2 arrays, with the x-coordinates of the intersection in the first
    array and the y-coordinates in the second array.
    
    https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    
    """
    
    S1,S2,S3,S4 = rect_inter_inner(x1,x2)
    S5,S6,S7,S8 = rect_inter_inner(y1,y2)

    C1 = np.less_equal(S1,S2)
    C2 = np.greater_equal(S3,S4)
    C3 = np.less_equal(S5,S6)
    C4 = np.greater_equal(S7,S8)

    ii,jj = np.nonzero(C1 & C2 & C3 & C4)
    
    n     = len(ii)

    dxy1 = np.diff(np.c_[x1,y1],axis=0)
    dxy2 = np.diff(np.c_[x2,y2],axis=0)
    
    T            = np.zeros((4,n))
    AA           = np.zeros((4,4,n))
    AA[0:2,2,:]  = -1
    AA[2:4,3,:]  = -1
    AA[0::2,0,:] = dxy1[ii,:].T
    AA[1::2,1,:] = dxy2[jj,:].T

    BB      = np.zeros((4,n))
    BB[0,:] = -x1[ii].ravel()
    BB[1,:] = -x2[jj].ravel()
    BB[2,:] = -y1[ii].ravel()
    BB[3,:] = -y2[jj].ravel()

    for i in range(n):
        
        try:
            
            T[:,i] = np.linalg.solve(AA[:,:,i],BB[:,i])
            
        except:
            
            T[:,i] = np.NaN


    in_range = (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0 = T[2:,in_range]
    xy0 = xy0.T
    
    return xy0[:,0] , xy0[:,1]


def rect_inter_inner(x1,x2):
    r"""
    Supportive function for intersection().
    """
    
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1],x1[1:]]
    X2 = np.c_[x2[:-1],x2[1:]]    
    S1 = np.tile(X1.min(axis=1),(n2,1)).T
    S2 = np.tile(X2.max(axis=1),(n1,1))
    S3 = np.tile(X1.max(axis=1),(n2,1)).T
    S4 = np.tile(X2.min(axis=1),(n1,1))

    return S1,S2,S3,S4


#%% Test lines
    
if __name__ == "__main__":
    
    area = shoelace(x = np.array([0,1,2]),
                    y = np.array([0,5,0]))
    
    print('area = ',area)
    
    intersections = intersection(x1 = np.array([0,1,2,0]),
                                 y1 = np.array([0,5,0,0]),
                                 x2 = np.array([0,1,1]),
                                 y2 = np.array([2,2,-1]))
    
    print('intersections are: ',intersections)

