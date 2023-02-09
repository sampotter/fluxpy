# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:50:53 2023

@author: heath
"""

## This file will be implementing the Single Area Integration shown in Equation (5)
## of 'Calculation of Obstructed View Factors by Adaptive Integration'

import numpy as np
import array
from flux.util import tic, toc

def make_ffm_sai(shape_model):
    # Extract the vertices, faces, normals, and area from the shape_model
    V = shape_model.V
    F = shape_model.F
    N = shape_model.N
    A = shape_model.A
    P = shape_model.P
    
    # Setting up the arrays that will eventually create the matrix
    if shape_model.dtype == np.float32:
        typecode = 'f'
    elif shape_model.dtype == np.float64:
        typecode = 'd'
    else:
        raise RuntimeError(f'unsupported dtype {shape_model.dtype}')

    data = array.array(typecode)
    indices = array.array('Q')
    ind_ptr = array.array('Q')
    
    num_faces = len(F)
    indptr = 0
    tic()
    for i in range(num_faces):
        ind_ptr.append(indptr)
        n1 = N[i]
        A1 = A[i]
        p1 = P[i]
        for j in range(num_faces):
            if i == j:
                continue
            num_edges_1 = len(V[F][i])
            for edge in range(num_edges_1):
                
            a = V[F][j][0]
            b = V[F][j][1]
            b_vect = b-p1
            a_vect = a-p1
            c_vect = np.cross(a_vect, b_vect)
            e = np.linalg.norm(c_vect)
            d = np.dot(a_vect, b_vect)
            
            
        
        