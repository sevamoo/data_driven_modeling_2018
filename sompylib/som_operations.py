#module Operations includes all the required operations for training. 
#Maybe if number of functions are too much it is better to decompose this module to several


# Vahid Moosavi 2013 01 09  3:35 pm
import numpy as np
#from som_structure import SOM
#from sompylib.som_structure import SOM as som
# Grid distance claculation

def grid_dist(som,bmu_ind):
    """
    som and bmu_ind
    depending on the lattice "hexa" or "rect" we have different grid distance
    functions.
    bmu_ind is a number between 0 and number of nodes-1. depending on the map size
    bmu_coord will be calculated and then distance matrix in the map will be returned
    """
    try:
        lattice = getattr(som, 'lattice')
    except:
        lattice = 'hexa'
        print 'lattice not found! Lattice as hexa was set'
   
    if lattice == 'rect':
        return rect_dist(som,bmu_ind)
    elif lattice == 'hexa':
        try:
            msize =  getattr(som, 'mapsize')
            rows = msize[0]
            cols = msize[1]
        except:
            rows = 0.
            cols = 0.
            pass 
       
        #needs to be implemented
        print 'to be implemented' , rows , cols
        return np.zeros((rows,cols))

def rect_dist(som,bmu):
    #the way we consider the list of nodes in a planar grid is that node0 is on top left corner,
    #nodemapsz[1]-1 is top right corner and then it goes to the second row. 
    #no. of rows is map_size[0] and no. of cols is map_size[1]
    try:
        msize =  getattr(som, 'mapsize')
        rows = msize[0]
        cols = msize[1]
    except:
        pass 
        
    #bmu should be an integer between 0 to no_nodes
    if 0<=bmu<=(rows*cols):
        c_bmu = int(bmu%cols)
        r_bmu = int(bmu/cols)
        #changine the coordinate
    else:
      print 'wrong bmu'  
      
    #calculating the grid distance
    if np.logical_and(rows>0 , cols>0): 
        r,c = np.arange(0, rows, 1)[:,np.newaxis] , np.arange(0,cols, 1)
        dist2 = (r-r_bmu)**2 + (c-c_bmu)**2
        return dist2.ravel()
    else:
        print 'please consider the above mentioned errors'
        return np.zeros((rows,cols)).ravel()

def neighbor(som, bmu_ind,radius):
    neigh_func = getattr(som, 'neigh')
    dist2 = grid_dist(som, bmu_ind)
    if neigh_func == 'gaussian':
        #print 'neigh'
        #print np.exp(-1.0*dist2/(2.0*radius**2))
        return np.exp(-1.0*dist2/(2.0*radius**2))
        
    else:
        return 'please come back later!!'
        
def neighbor_para(som, bmus,radius,i, parts):
    neigh_func = getattr(som, 'neigh')
    wanted_parts = parts
    #print 'x2', X2[0].reshape(1,)
    #print 'Y2', Y
    #dim = x.shape[1]
    dlen = bmus.shape[0]
    cd = getattr(som, 'nnodes')
    Low = i*dlen // wanted_parts
    High =  (i+1)*dlen // wanted_parts-1
    High = min(High, dlen-1)
    #print Low , High
    dbmu = bmus[Low:High+1];
    #print len(ddata)
    H = np.zeros((High-Low+1,cd))
    for i in range(len(dbmu)):
        dist2 = grid_dist(som, dbmu[i])
        if neigh_func == 'gaussian':
            H[i,:] = np.exp(-1.0*dist2/(2.0*radius**2))    
        else:
            print 'please come back later!!'
    return H