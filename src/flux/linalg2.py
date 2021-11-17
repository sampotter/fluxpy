#!/usr/bin/env python3

# read view factor matrix
# calculates SVD, full or truncated
# stores SVD output

import numpy
import scipy 
from flux.compressed_form_factors import CompressedFormFactorMatrix
from sklearn.utils.extmath import randomized_svd
import os
import time


def xSVDcomputation(ff_filename='FF.bin', TRUNC=0, mode='full'):
    # perform Singular Value Decomposition
    # ff_filename ... filename of viewfactor matrix, e.g., 'FF.bin'
    # TRUNC ... number of elements in truncation, but 0 means all

    print('Input file name:',ff_filename)
    
    # load FF.bin previously generated
    VF = CompressedFormFactorMatrix.from_file(ff_filename).toarray()
    print('- loaded FF.bin')

    assert VF.shape[0] == VF.shape[1]  # matrix should be square
    Nflat = VF.shape[0]   # dimension of square VF matrix = number of facets
    print('- Number of facets:',Nflat)

    if TRUNC==0: # zero means all
        TRUNC = Nflat
    assert TRUNC>0

    start = time.time()
    if mode=='full':
        print('- performing full SVD decomposition', VF.shape)
        #U,sigma,Vt = numpy.linalg.svd(VF, full_matrices=True, compute_uv=True, hermitian=False)
        U,sigma,Vt = scipy.linalg.svd(VF, full_matrices=True, compute_uv=True)
    else:
        print('- performing approximate SVD with',TRUNC,'terms', VF.shape)
        U,sigma,Vt = randomized_svd(VF, n_components=TRUNC, random_state=None)

    # check that no negative singular values are present
    assert(len(sigma[sigma<0])==0)

    end = time.time()
    print('Time for SVD:',end-start)

    print('- saving outputs U,sigma,Vt')
    outdir = '../examples/'
    if TRUNC<Nflat:
        print('- output truncated to',TRUNC,'terms')
    numpy.savetxt(os.path.join(outdir,'svd_sigma.dat'), sigma[:TRUNC], newline=' ')
    numpy.savetxt(os.path.join(outdir,'svd_U.dat'), U[:,0:TRUNC], fmt='%11.7f')
    numpy.savetxt(os.path.join(outdir,'svd_V.dat'), Vt[0:TRUNC,:], fmt='%11.7f')


    
if __name__ == '__main__':

    #xSVDcomputation('../examples/FF.bin',TRUNC=200,mode='full')
    xSVDcomputation('../examples/FF.bin',TRUNC=200,mode='approx')
