# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:23:14 2023

@author: heath
"""

# This file will compare the Form Factor Matrices that are outputted by PyViewFactor
# and by the midpoint approximation

import numpy as np
import scipy.sparse
from make_ffm_midpoint import obj_to_shapemodel
import matplotlib.pyplot as plt
import pyvista as pv
import pyviewfactor as pvf
from flux.shape import get_surface_normals
    
def make_cent_dist(shape_model):
    """ Function that calculated the distances between the centiods of 
    all faces in a given mesh.
    
    Parameters
    ----------
    shape_model : CgalTrimeshShapeModel
    """
    P = shape_model.P
    cent_dist = []
    num_faces = len(P)
    for i in range(num_faces):
        row = []
        for j in range(num_faces):
            dist = np.linalg.norm(P[i] - P[j])
            row.append(round(dist,2))
        cent_dist.append(row)
    return cent_dist

def make_error_mat(FF_exp, FF_analy):
    """ Function that, given the an experimental Form Factor Matrix
    and an analytical Form Factor Matrix, will compute the error between
    each form factor and organize it in a matrix
    
    Parameters
    ---------
    FF_exp : scipy.sparse.csr_matric
            The experimental form factor matrix that needs to be tested.
            
    FF_analy : scipy.sparse.csr_matrix
            The analytical form factor matrix to compare.
    """
    error_mat = []
    num_faces = len(FF_exp.toarray()[0])
    for i in range(num_faces):
        row = []
        for j in range(num_faces):
            exp = FF_exp.toarray()[i][j]
            analy = FF_analy.toarray()[i][j]
            if exp == analy == 0:
                error = 0
            elif exp == 0:
                error = 1
            elif analy == 0:
                error = 1
            else:
                error = abs((exp-analy)/analy)
            row.append(error)
        error_mat.append(row)
    return error_mat

if __name__ == '__main__':  
    # Loading the FF_midpoint and FF_PyViewFactor
    FF_midpoint = scipy.sparse.load_npz('FF_midpoint_6.npz')
    FF_pvf = scipy.sparse.load_npz('FF_pvf_6_correct.npz')
    FF_midpoint1 = scipy.sparse.load_npz('FF_4.npz')
    
    # Computing Error Matrix
    #error_mat = make_error_mat(FF_pvf, FF_midpoint)
    
    
    # Computing the Centriod Distance Matrix
    shape_model = obj_to_shapemodel('hemispherical_mesh_6.obj', '.')
    print(type(shape_model))
    #cent_dist = make_cent_dist(shape_model)
    
    # # Plot #1
    # plt.plot(cent_dist[200], error_mat[200], 'bo')
    # plt.title('Midpoint Approximation Error vs. Surface Centriod Distance')
    # plt.xlabel('Centriod Distance')
    # plt.ylabel('Error (%)')
    # plt.show()
    
    # Finding the maximum view factor from the center Triangle (N=)
    print('Max View Factor to Triangle 200 - PVF:', max(FF_pvf.toarray()[200]))
    #print('Max View Factor to Triangle 200 - Midpoint:', max(FF_midpoint.toarray()[200]))
    print('Location of Max View Factor - PVF', np.argmax(FF_pvf.toarray()[200]))
    #print('Location of Max View Factor - Midpoint', np.argmax(FF_midpoint.toarray()[200]))
    #print('PVF Value at Midpoint Max', FF_pvf.toarray()[200][200])
    
    
    
    # Checking Midpoint View Factors at Adjacent Triangles
    # All of the view factors are zero
    print('Midpoint VF at Adjacent:', FF_midpoint1.toarray()[256][177])
    print('Midpoint VF at Adjacent:', FF_midpoint1.toarray()[256][244])
    print('Midpoint VF at Adjacent:', FF_midpoint1.toarray()[256][254])
            
    #Checking PyViewFactor View Factors at Adjacent Triangles
    # All of the view factors are zero
    print('PVF VF at Adjacent:', FF_pvf.toarray()[256][177])
    print('PVF VF at Adjacent:', FF_pvf.toarray()[256][244])
    print('PVF VF at Adjacent:', FF_pvf.toarray()[256][254])
    
    # Creating a function that will take in three points and compute the normal
    def calc_norm(p1, p2, p3):
        A = p2-p1
        B = p3-p1
        N = np.cross(A,B)
        return N/np.linalg.norm(N)
    
    # Manually check the view factor calculation using PyViewFactor directly
    V = shape_model.V
    F = shape_model.F
    N = shape_model.N
    
    
    p1 = [-0.41363139,  0.04571755, -0.21954044]
    p2 = [-0.40387446, -0.21851461, -0.20336582]
    p3 = [-0.22601854, -0.15600933, -0.26049735]
    ps = [p1, p2, p3]
    p = pv.Triangle(ps)
    print('Calculated Normal - 256', p.cell_normals)
    q1 = [-0.41363139,  0.04571755, -0.21954044]
    q2 = [-0.22601854, -0.15600933, -0.26049735]
    q3 = [-0.27034409, 0.03967309, -0.26081175]
    qs = [q1, q2, q3]
    q = pv.Triangle(qs)
    print('Calculated Normal - 177', q.cell_normals)
    print('256-177 Visibility', pvf.get_visibility(p,q))
    print('Midpoint VF at Adjacent:', FF_midpoint1.toarray()[256][177])
    print('256-177:', pvf.compute_viewfactor(p, q, epsilon=1e-8))
    print('FFM View Factor:', FF_pvf.toarray()[256][177])
    
    # Computing View Factor of 256 - 244
    # This view factor becomes non-zero when you reverse the coordinates
    q1 = V[F][244][0]
    q2 = V[F][244][1]
    q3 = V[F][244][2]
    qs = [q1, q2, q3]
    q = pv.Triangle(qs)
    print('Cell_normals - 244', q.cell_normals)
    print('244 Normal', N[244])
    print('256-244 Visibility', pvf.get_visibility(p,q))
    print('256-244 (e):', pvf.compute_viewfactor(p, q, epsilon=1e-6))
    print('FFM View Factor:', FF_pvf.toarray()[256][244])
    
    # Computing View Factor of 256 - 254
    q1 = V[F][254][0]
    q2 = V[F][254][1]
    q3 = V[F][254][2]
    qs = [q1, q2, q3]
    qs.reverse()
    q = pv.Triangle(qs)
    print('256-254 Visibility', pvf.get_visibility(p,q))
    print('256-254:', pvf.compute_viewfactor(p, q))
    print('256-254 (e):', pvf.compute_viewfactor(p, q, epsilon=1e-6))
    print('FFM View Factor:', FF_pvf.toarray()[256][254])
    
    ## Testing View Factors outside of crater
    # Computing View Factor of 256 - 20
    # This view factor now correctly shows visibility as false and view factor=0
    q1 = V[F][20][0]
    q2 = V[F][20][1]
    q3 = V[F][20][2]
    qs = [q1, q2, q3]
    q = pv.Triangle(qs)
    print('256-20 Visibility', pvf.get_visibility(p,q))
    print('256-20:', pvf.compute_viewfactor(p, q))
    print('256-20 (e):', pvf.compute_viewfactor(p, q, epsilon=1e-6))
    print('FFM View Factor:', FF_pvf.toarray()[256][20])
    
    ## Testing non-adjacent view factors, still in crater
    # Computing View Factor of 256-255
    q1 = V[F][255][0]
    q2 = V[F][255][1]
    q3 = V[F][255][2]
    qs = [q1, q2, q3]
    #qs.reverse()
    q = pv.Triangle(qs)
    print('256-255 Visibility', pvf.get_visibility(p,q))
    print('256-255:', pvf.compute_viewfactor(p, q))
    print('256-255:', pvf.compute_viewfactor(p, q, epsilon = 1e-6))
    print('FFM View Factor:', FF_midpoint1.toarray()[256][255])
    
    ## Checking view factor of the same triangles
    # Computing View Factor of 256-256
    q1 = V[F][256][0]
    q2 = V[F][256][1]
    q3 = V[F][256][2]
    qs = [q1, q2, q3]
    #qs.reverse()
    q = pv.Triangle(qs)
    print('256-256 Visibility', pvf.get_visibility(p,q))
    print('256-256:', pvf.compute_viewfactor(p, q))
    print('FFM View Factor:', FF_pvf.toarray()[256][256])
    
    
    # Plot #2 - PVF Form Factor Plot
    # max_error = np.argmax(error_mat[238])
    # print('Midpoint:', FF_midpoint1.toarray()[238][max_error])
    # print('PVF', FF_pvf.toarray()[238][max_error])
    
    
    ## PyViewFactor Write-Up Analysis
    # FF_pvf = scipy.sparse.load_npz('FF_pvf_4_1.npz')
    # FF_midpoint1 = scipy.sparse.load_npz('FF_4.npz')
    
    # print(V[F][197])
    # print(V[F][max_vf])
    
    verts = shape_model.V[:, :2]
    faces = shape_model.F
    dif = FF_midpoint.toarray()[1086]-FF_pvf.toarray()[1086]
    max_dif = np.argmax(abs(dif))
    print(max_dif)
    print('PVF 1086 Max Diff:', FF_pvf.toarray()[1086][max_dif])
    print('MD 1086 Max Diff:', FF_midpoint.toarray()[1086][max_dif])
    p1 = V[F][1086][0]
    p2 = V[F][1086][1]
    p3 = V[F][1086][2]
    ps = [p1, p2, p3]
    p = pv.Triangle(ps) 
    q1 = V[F][max_dif][0]
    q2 = V[F][max_dif][1]
    q3 = V[F][max_dif][2]
    qs = [q1, q2, q3]
    q = pv.Triangle(qs)
    print('p', V[F][1086])
    print('q', V[F][max_dif])
    print('P Normal', p.cell_normals)
    print('Q Normal', q.cell_normals)
    print('PVF Manual:', pvf.compute_viewfactor(q, p, epsilon=1e-8))
    
    
    #trying subplots
    fig1, ax1 = plt.subplots()
    pyvf = ax1.tripcolor(*verts.T, dif, triangles=faces)
    #md = ax1.tripcolor(*verts.T, FF_midpoint.toarray()[197], triangles=faces)
    plt.scatter(*shape_model.P[1086, :2], s=30, color='red', zorder=2)
    plt.scatter(*shape_model.P[max_dif, :2], s=30, color='blue', zorder=2)
    fig1.colorbar(pyvf)
    plt.title('Difference between Midpoint and PVF View Factors')
    plt.show()


