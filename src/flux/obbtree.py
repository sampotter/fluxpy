import itertools as it
import numpy as np
import scipy.spatial

from flux.shape import get_face_areas, get_centroids


def get_obb_partition_2d(I, shape_model, num_samples=1000):

    # get the vertices of all indexed triangles
    vertices_idx = shape_model.F[I]
    cluster_vertices = shape_model.V[vertices_idx]
    cluster_vertices = cluster_vertices[:,:,:2]
    cluster_vertices_flat = cluster_vertices.reshape(-1,2)

    # compute the convex hull of the triangle vertices
    hull = scipy.spatial.ConvexHull(cluster_vertices_flat)

    # sample point over the surface of the convex hull
    hull_deln_simplices = scipy.spatial.Delaunay(cluster_vertices_flat[hull.vertices]).simplices
    hull_deln_simplex_coords = cluster_vertices_flat[hull.vertices][hull_deln_simplices]
    hull_deln_areas = np.abs(np.linalg.det(hull_deln_simplex_coords[:, :2, :] - hull_deln_simplex_coords[:, 2:, :])) / np.math.factorial(2)
    sampled_deln_triangles = np.random.choice(len(hull_deln_areas), size=num_samples, p=hull_deln_areas/hull_deln_areas.sum())
    sampled_interior = np.einsum('ijk, ij -> ik', hull_deln_simplex_coords[sampled_deln_triangles], scipy.stats.dirichlet.rvs([1]*(2 + 1), size=num_samples))
    
    # compute the mean and covariance of the sample
    mu = sampled_interior.mean(axis=0)
    cov = np.cov(sampled_interior.T)

    # eigenvectors of covariance define partition
    _, eigvecs = np.linalg.eig(cov)
    plane1 = np.array([(eigvecs[1,0]/eigvecs[0,0]), -1, (-1*(eigvecs[1,0]/eigvecs[0,0])*mu[0]) + mu[1]])
    plane2 = np.array([(eigvecs[1,1]/eigvecs[0,1]), -1, (-1*(eigvecs[1,1]/eigvecs[0,1])*mu[0]) + mu[1]])

    # centroids determine partition
    P = get_centroids(shape_model.V, shape_model.F)
    cluster_centroids = P[I,:2]
    padded_cluster_centroids = np.concatenate([cluster_centroids,np.ones((cluster_centroids.shape[0],1))], axis=1)
    plane1_dot = (plane1 * padded_cluster_centroids).sum(axis=1) > 0
    plane2_dot = (plane2 * padded_cluster_centroids).sum(axis=1) > 0
    
    Is = []
    Is.append(np.where(np.logical_and(plane1_dot, plane2_dot))[0])
    Is.append(np.where(np.logical_and(~plane1_dot, plane2_dot))[0])
    Is.append(np.where(np.logical_and(plane1_dot, ~plane2_dot))[0])
    Is.append(np.where(np.logical_and(~plane1_dot, ~plane2_dot))[0])
    return Is