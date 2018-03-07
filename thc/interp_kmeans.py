import numpy
#import time
from timeit import default_timer as timer
import sys

# assigns a centriod to every point in the grid
# very inefficient for now, improve later
def classify(rgrid,c):

    ngs = rgrid.shape[0]  
    nmu = c.shape[0]    
    X = numpy.zeros(ngs,dtype=numpy.int32)
    X[:] = -1
    # simple double loop for now
    for ni in range(ngs):
        X[ni] = numpy.argmin(numpy.sqrt(numpy.sum((c-rgrid[ni,:])**2,axis=1))) 
    return X            

# calculates the new centroids and measures the distance with respect to the old ones
def centroids(rgrid,w,X,nmu):
    ngs = rgrid.shape[0]
    c = numpy.zeros((nmu,3),dtype=numpy.float64)
    wtot = numpy.zeros(nmu,dtype=numpy.float64)
    for ni in range(ngs):
        #assert(w[ni] > 0.0) 
        wtot[X[ni]] += w[ni]
        c[X[ni],:] += w[ni]*rgrid[ni,:]     
    for n in range(nmu):
        assert(wtot[n]>0.0)
        c[n,:] /= wtot[n]
    return c

# finds the closest point in the grid for every centroid and returns the set of indices
def map2grid(rgrid,c):
    X = classify(c,rgrid)
    X = numpy.sort(X)
    # look for repeated
    for i in range(X.shape[0]-1):
        assert( X[i] != X[i+1] ) 
    return X


# rgrid[ngs, 3]: grid points
# w[ngs]: weight function, e.g. electron density
# c[nmu,3]: initial guess for interpolating points - defines nmu
# FIX FIX FIX: Add PBC
def IPts_k_means(rgrid, w, c, maxIT=100, thres=1e-6):

    assert(rgrid.ndim == 2)
    assert(rgrid.shape[1] == 3)
    assert(w.ndim==1)
    assert(c.ndim==2)
    assert(c.shape[1] == 3)
    assert(rgrid.shape[0]==w.shape[0])
    assert(thres > 0.0)

    ngs = rgrid.shape[0] 
    nmu = c.shape[0] 

    for t in range(maxIT):
        X = classify(rgrid,c)
        c_new = centroids(rgrid,w,X,nmu)
        d = numpy.linalg.norm(c_new-c)
#        print ' iteration, distance:',t,d
#        sys.stdout.flush()
        if d < thres:
            return map2grid(rgrid,c_new)
        c = c_new
    
    return map2grid(rgrid,c)
