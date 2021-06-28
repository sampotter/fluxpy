#!/usr/bin/env python3

# read view factor matrix
# calculates SVD, full or truncated
# stores SVD output

import numpy 
from flux.compressed_form_factors import CompressedFormFactorMatrix
from sklearn.utils.extmath import randomized_svd
import time


def xSVDcomputation(ff_filename='FF.bin', TRUNC=1, mode='full'):
    # perform singular value decomposition
    # ff_filename ... filename of viewfactor matrix, e.g. 'FF.bin'
    # TRUNC ... number of elements in truncation
    
    # load FF.bin previously generated
    FF = CompressedFormFactorMatrix.from_file(ff_filename)
    print('- loaded FF.bin')
    VF = FF # how do I get the viewfactors as a single square matrix?
    
    Nflat = VF.size   # dimension of square VF matrix = number of facets
    print('- Number of facets',Nflat)
    #assert VF.size == FF.shape_model.F.shape[0]

    
    print('performing SVD decoposition')
    start = time.time()
    if mode=='full': # full SVD 
        U,sigma,Vt = numpy.linalg.svd(vf, full_matrices=True, compute_uv=True, hermitian=False)
    else: # approximate and truncated SVD
        print('- Number of terms in truncated SVD',TRUNC)
        U,sigma,Vt = randomized_svd(vf, n_components=TRUNC, n_iter=5, random_state=None)
    end = time.time()
    print('Time for SVD:',end-start)
    

    print('- saving outputs U,sigma,Vt')
    numpy.savetxt('svd_sigma.dat',sigma, newline=' ')
    numpy.savetxt('svd_U.dat',U[:,0:TRUNC], fmt='%11.7f')
    numpy.savetxt('svd_V.dat',Vt[0:TRUNC,:], fmt='%11.7f')


    
if __name__ == '__main__':

    xSVDcomputation('FF.bin')
