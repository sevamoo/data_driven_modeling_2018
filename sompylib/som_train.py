#SOM Trainer

# <codecell>
# Vahid Moosavi 2013 01 09  2:35 pm
import numpy as np
from matplotlib import pyplot as plt

#from .som_structure import SOM


     
# Initialize map codebook: Weight vectors of SOM
def init_map(som):
    dim = 0
    n_nod = 0
    if  getattr(som, 'initmethod')=='random':
        #It produces random values in the range of min- max of each dimension based on a uniform distribution
        mn = np.tile(np.min(getattr(som,'data'), axis =0), (getattr(som, 'nnodes'),1))
        mx = np.tile(np.max(getattr(som,'data'), axis =0), (getattr(som, 'nnodes'),1))
        setattr(som, 'codebook', mn + (mx-mn)*(np.random.rand(getattr(som, 'nnodes'), getattr(som, 'dim'))))
    elif getattr(som, 'initmethod') == 'pca':
        #needs to be implemented: a separate function maybe
        setattr(som, 'codebook', np.random.rand(getattr(som, 'nnodes'), getattr(som, 'dim'))) 
    else:
        print 'please select a corect initialization method'
        print 'set a correct one in SOM. current SOM.initmethod:  ', getattr(som, 'initmethod')
        print "possible init methods:'random', 'pca'"
     
    
#   % For the case where there are unknown components in the data, each data
#% vector will have an individual mask vector so that for that unit, the 
#% unknown components are not taken into account in distance calculation.
#% In addition all NaN's are changed to zeros so that they don't screw up 
#% the matrix multiplications and behave correctly in updating step.
#Known = ~isnan(D);
#W1 = (mask*ones(1,dlen)) .* Known'; 
#D(find(~Known)) = 0;  
#
#% constant matrices
#WD = 2*diag(mask)*D';    % constant matrix
#dconst = ((D.^2)*mask)'; % constant in distance calculation for each data sample 
#                         % W2 = ones(munits,1)*mask'; D2 = (D'.^2); 	