# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:00:51 2023

@author: heath
"""
## This file will define mesh characeristics, create a HemisphericalCrater mesh,
## and save this mesh to the disk as an OBJ file

import argparse
parser = argparse.ArgumentParser(description='Writing OBJ file to disk to PyViewFactor')
parser.add_argument(
    '-p', type=int, default=6,
    help='target edge length: h = (2/3)**p')
parser.add_argument(
    '--e0', type=float, default=15,
    help='sun angle above horizon in [deg.]')
parser.add_argument(
    '--F0', type=float, default=1000,
    help='solar constant [W/m^2]')
parser.add_argument(
    '--rho', type=float, default=0.3,
    help='albedo')
parser.add_argument(
    '--emiss', type=float, default=0.99,
    help='emissivity')
parser.add_argument(
    '--tol', type=float, default=None,
    help='tolerance used when assembling compressed form factor matrix')
parser.add_argument(
    '--beta', type=float, default=40,
    help='Ingersoll crater angle measured from vertical [deg.]')
parser.add_argument(
    '--rc', type=float, default=0.8,
    help='Crater radius [m]')
parser.add_argument(
    '--outdir', type=str, default='.',
    help='Directory to write output files to')
parser.add_argument(
    '--blocks', type=bool, default=False,
    help='Make a plot of the form factor matrix blocks')
parser.add_argument(
    '--mprof', type=bool, default=False,
    help='Measure the memory use during assembly and save results')
#Contour Rim is set to True
#Contour Shadow is set to False
parser.add_argument(
    '--contour_rim', type=bool, default=True,
    help='Whether the rim of the crater should be contoured')
parser.add_argument(
    '--contour_shadow', type=bool, default=False,
    help='Whether the shadow line in the crater should be contoured')
args = parser.parse_args()
print(args.outdir)


import os.path
import numpy as np
import trimesh

from flux.ingersoll import HemisphericalCrater

if __name__ == '__main__':
    #if there is not a directory to write output files too, make a directory
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    #specifying some variables
    h = (2/3)**args.p
    e0 = np.deg2rad(args.e0)
    beta = np.deg2rad(args.beta)
    
    #creating HemisphericalCrater object with necessary inputs from the defined arguements
    hc = HemisphericalCrater(beta, args.rc, e0, args.F0, args.rho, args.emiss)
    print('- groundtruth temperature in shadow: %1.2f K' % (hc.T_gt,))

    # Create the triangle mesh
    # If the rim and shadow should be apart of the mesh
    if args.contour_rim and args.contour_shadow:
        #inside of HemisphericalCrater class there is a make_trimesh command
        #this command takes in h (refinement factor) and **kwargs
        V, F, parts = hc.make_trimesh(h, contour_rim=True, contour_shadow=True,
                                      return_parts=True)
    ## THIS IS WHERE WE WANT TO BE
    # If we just want the rim to be apart of the mesh
    elif args.contour_rim:
        V, F, parts = hc.make_trimesh(h, contour_rim=True, return_parts=True)
    elif args.contour_shadow:
        raise ValueError('must set --contour_rim if --contour_shadow is set')
    else:
        V, F = hc.make_trimesh(h)
        parts = None
    print('- created tri. mesh with %d faces' % (F.shape[0],))

    # Write the mesh to disk as an OBJ file
    filename = 'hemispherical_mesh_6_ex.obj'
    trimesh.Trimesh(V, F).export(os.path.join(args.outdir, filename))
    print('- wrote ' + filename)
    
    