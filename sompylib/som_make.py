import numexpr as ne
import numpy as np
from time import time
import scipy.spatial as spdist
import tables as tb
import timeit
import sys
from math import sqrt
from sklearn.externals.joblib import Parallel, delayed
from numba import autojit, double, jit
from sklearn.externals.joblib import load, dump
import tempfile
import shutil
import os

#reload(sys.modules['sompylib.som_structure'])
#reload(sys.modules['sompylib.som_operations'])
# reload(sys.modules['sompylib.som_train'])
from sompylib.som_train import init_map
from sompylib.som_structure import SOM as som
#from sompylib.som_operations import grid_dist, neighbor, neighbor_para
import itertools
from scipy.sparse import csr_matrix

# in order to control memory limit, it divide dlen to batches, blen
# we will call this function in parallel for different number of jobs
def chunk_based_bmu_find(x, y, Y2):
    dim = x.shape[1]
    dlen = x.shape[0]
    nnodes = y.shape[0]
    #nnodes = getattr(som, 'nnodes')
    #print Low , High
    bmu = np.zeros((dlen,2))
    blen = min(nnodes,dlen) 
    #print blen
    i0 = 0;     
    while i0+1<=dlen:
        Low =  (i0)
        High = min(dlen,i0+blen)
        i0 = i0+blen      
        ddata = x[Low:High+1]
        d = np.dot(y, ddata.T)
        d *= -2
        d += Y2.reshape(nnodes,1)
        bmu[Low:High+1,0] = np.argmin(d, axis = 0)
        bmu[Low:High+1,1] = np.min(d, axis = 0)  
    return bmu
    #return very_fast_dist_calc(x[Low:High],X2, y)

def para_bmu_find(som, x, njb = 1):
     
    y = getattr(som, 'codebook')
    #x = getattr(som, 'data')
    #x = training_data
    dlen = x.shape[0]
    #y = getattr(som, 'codebook')
    
    Y2 = None
    Y2 = np.einsum('ij,ij->i', y, y)
    bmu = None
    b = None
    
    #here it finds BMUs for chunk of data in parallel
    b  = Parallel(n_jobs=njb, pre_dispatch='3*n_jobs')(delayed(chunk_based_bmu_find)\
    (x[i*dlen // njb:min((i+1)*dlen // njb, dlen)],y, Y2) \
    for i in xrange(njb))
    
    bmu = np.asarray(list(itertools.chain(*b))).T
    b = None
    Y2 = None
    return bmu
    
# First find the Voronoi set of each node. It needs to calculate a matrix size nnodes. Super fast
def update_codebook_voronoi(som, training_data, bmu, H, radius):
    #bmu has shape of 2,dlen, where first row has bmuinds
    # we construct ud2 from precomputed UD2 : ud2 = UD2[bmu[0,:]]
    nnodes = getattr(som, 'nnodes')
    dlen = getattr(som ,'dlen')
    dim = getattr(som, 'dim')
    New_Codebook = np.empty((nnodes, dim))
    #print bmu[0]
    inds = bmu[0].astype(int)
    #print inds
    row = inds
    col = np.arange(dlen)
    val = np.tile(1,dlen)
    P = csr_matrix( (val,(row,col)), shape=(nnodes,dlen) )
    #assert( P.shape == (nnodes, dlen))
    S = P.dot(training_data)
    #assert( S.shape == (nnodes, dim))
    #assert( H.shape == (nnodes, nnodes))
    
    # H has nnodes*nnodes and S has nnodes*dim  ---> Nominator has nnodes*dim
    #print Nom
    Nom = np.empty((nnodes,nnodes))
    Nom =  H.T.dot(S)
    #assert( Nom.shape == (nnodes, dim))
    nV = np.empty((1,nnodes))
    nV = P.sum(axis = 1).reshape(1, nnodes)
    #print nV
    #assert(nV.shape == (1, nnodes))
    Denom = np.empty((nnodes,1))
    Denom = nV.dot(H.T).reshape(nnodes, 1)
    #assert( Denom.shape == (nnodes, 1))
    New_Codebook = Nom/Denom
    #assert (New_Codebook.shape == (nnodes,dim))
    #setattr(som, 'codebook', New_Codebook)
    return New_Codebook 
    del New_Codebook   

################################################################################
def train(dlen, msz0, msz1, dim):
    data = np.random.rand(dlen,dim)
    nnodes = msz0 * msz1
    mem = np.log10(dlen*nnodes*dim)
    print 'data len is %d and data dimension is %d' % (dlen, dim)
    print 'map size is %d, %d' %(msz0, msz1)
    print 'array size in log10 scale' , mem 
    if mem < 10.5:
        n_job = 8
    elif mem >= 10.5 and mem <= 11.2:
        n_job = 4
    elif mem > 11.2 and mem <= 11.4:
        n_job = 3
    elif mem > 11.4 and mem <= 11.65:
        n_job = 2
    else:
        n_job = "goes out of memory"
    
    print 'nomber of jobs: ', n_job
    sm = som('sm', data, mapsize = [msz0, msz1])
    init_map(sm)
    radius_ini = 3
    radius_fin = .2
    trainlen = getattr(sm, 'epochs')
    UD2 = getattr(sm, 'UD2')
    nnodes = getattr(sm, 'nnodes')
    dlen = getattr(sm, 'dlen')
    New_Codebook_V = np.empty((nnodes, dim))
    New_Codebook_V = getattr(sm, 'codebook')
    #print New_Codebook_V
    #in case of Guassian neighborhood
    #H = np.exp(-1.0*UD2/(2.0*radius**2)).reshape(nnodes, nnodes)
    #print 'initial codebook'
    init_codebook =  getattr(sm, 'codebook')
    
    shared_memory = 'yes'
    #shared_memory = 'no'
    print '******'
    print 'data is in shared memory?', shared_memory
    print '******'
    #print 'data size= ', sys.getsizeof(x)
    if shared_memory == 'yes':
        data = getattr(sm, 'data')
        Data_folder = tempfile.mkdtemp()
        data_name = os.path.join(Data_folder, 'data')
        dump(data, data_name)
        data = load(data_name, mmap_mode='r')
    else:
        data = getattr(sm, 'data')
    
    
    t0 = time()
    trainlen= 1
    radius = np.linspace(radius_ini, radius_fin, trainlen)
    for i in range(trainlen):
        #in case of Guassian neighborhood
        H = np.exp(-1.0*UD2/(2.0*radius[i]**2)).reshape(nnodes, nnodes)
        print 'epoch: ', i+1
        print '#########' 
        t1 = time()
        bmu = None
        bmu = para_bmu_find(sm, data, njb = n_job)
        print "*** Time elapsed:", round(time() - t1, 3)
        print '#########'

        
        New_Codebook_V = update_codebook_voronoi(sm, data, bmu, H, radius)
        #setattr(sm, 'codebook', New_Codebook_V)
        bmu = None
        #print New_Codebook_V

        #sys.stdout.flush()
    setattr(sm, 'codebook', New_Codebook_V)
    ts = round(time() - t0, 3)
    print
    print "*** Time elapsed:", ts
    del data
    
    #print 'initial codebook'
    #print init_codebook 
    #
    #print 
    #print 'new codebook by a Voronoi'
    #print np.around(getattr(sm, 'codebook') , decimals=3)
    #print 
    #print 'training Data' 
    #print np.around(getattr(sm, 'data') , decimals=3)

##################################################################

## It is super memory intensive, as it needs to store a (dlen*nnodes) size matrix 
#def update_codebook(som, bmu, UD2, radius):
#    #bmu has shape of 2,dlen, where first row has bmuinds
#    # we construct ud2 from precomputed UD2 : ud2 = UD2[bmu[0,:]]
#    nnodes = getattr(som, 'nnodes')
#    dlen = getattr(som ,'dlen')
#    inds = bmu[0].astype(int)
#    #print bmu[0]
#    #print inds
#    assert(inds.shape == (dlen,))
#    i =  np.arange(dlen)
#    ud2 = UD2[inds[i],:].reshape(dlen, nnodes)
#    assert( ud2.shape == (dlen, nnodes))
#    #print 'ud2'
#    #print ud2
#    #in case of Guassian neighborhood
#    H = np.exp(-1.0*ud2/(2.0*radius**2)).reshape(dlen, nnodes)
#    assert( H.shape == (dlen, nnodes))
#    
#    #print 'H'
#    #print H
#    # H has nnodes*dlen and data has dlen*dim  ---> Nominator has cd*dim
#    dim = getattr(som, 'dim')
#    Nom =  H.T.dot(getattr(som, 'data'))
#    #np.einsum('ij,jk->ik', H, getattr(som, 'data')).reshape(nnodes, dim)
#    assert( Nom.shape == (nnodes, dim))
#    
#    Denom = np.einsum('ij->i', H.T).reshape(nnodes,1)
#    assert( Denom.shape == (nnodes,1))
#    
#    New_Codebook = Nom/Denom
#    #setattr(som, 'codebook', New_Codebook)
#    
#    ##H = np.empty((dlen, cd))
#    #
#    #if bmu.shape[1] == dlen:
#    #    out  = Parallel(n_jobs=njb, pre_dispatch='3*n_jobs')(delayed(neighbor_para)(som, bmu[0,:], radius,i, sz) \
#    #    for i in xrange(sz))  
#    #    H = np.asarray(list(itertools.chain(*out)))
#
#                
#        #for i in range(dlen):    
#        #    H[i,:] = neighbor(som, bmu[0,i], radius)
#    return New_Codebook
#    
#
#
