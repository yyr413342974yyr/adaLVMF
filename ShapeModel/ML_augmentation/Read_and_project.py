#!/usr/bin/env python
# ===================================================================================================
# CONVERT VTK UNSTRUCTURED GRID INTO ABAQUS INPUT FILE
# 
# label 0 is for the background
# label 1 is for RV blood pool
# label 2 is for LV walls
# label 3 is for LV blood pool
#
# Author: Stefano Buoso, buoso@biomed.ee.ethz.ch
# Date  : October 2018
# ===================================================================================================

import os, shutil
import sys
import meshio
import numpy    as np
import _pickle  as pickle

def compute_amplitudes(Coords,PHI):
    """ This function computes the projection of the onto the modes
    Coords is a Nx3 numpy array containing the 3D coordinates of the N points
    PHI is a 3N*M matrix where M is the number of modes of the model
    """
    import numpy as np

    vector_POD = np.concatenate((Coords[:,0],Coords[:,1],Coords[:,2]),axis = 0)
    amplitudes = PHI.transpose().dot(vector_POD)

    return np.array(amplitudes)[0,:]


# INPUT SECTION
# ===========================================================================================================
# Folder path with input mesh and projection matrix

path_to_ML_augmentation_folder = 'D:\Human_data\Whole_atlas_cardiac_model\ML_augmentation'
vtk_file_path      = path_to_ML_augmentation_folder + '\LV_mean.vtk'
proj_matrix_path   = path_to_ML_augmentation_folder + '\Projection_matrix.dat'
new_vtk_file_path  = path_to_ML_augmentation_folder + '\LV_mean_modified.vtk'
rec_vtk_file_path  = path_to_ML_augmentation_folder + '\LV_mean_modified_rec.vtk'

with open(proj_matrix_path,'rb') as infile:
    PHI = pickle.load(infile)

mesh = meshio.read(vtk_file_path)
Coords = mesh.points
n_points = len(Coords[:,0])
# Here you do wathever you want to do to the mesh
# NewCoords = f(Coords)
New_Coords = Coords*2 # Here I am just scaling the model
new_amplitudes = compute_amplitudes(New_Coords,PHI)

# If you want to store the new vtk file
mesh_new = mesh
mesh_new.points = New_Coords
meshio.write(new_vtk_file_path,mesh_new,'vtk-ascii')

# If you want to store the file reconstructed with the computed amplitudes
proj_back = PHI.dot(new_amplitudes.T)
Rec_Coords      = np.zeros(Coords.shape)
Rec_Coords[:,0] = np.array(proj_back)[0,:][0:n_points]
Rec_Coords[:,1] = np.array(proj_back)[0,:][n_points:2*n_points]
Rec_Coords[:,2] = np.array(proj_back)[0,:][2*n_points:3*n_points]
mesh_rec = mesh
mesh_rec.points = Rec_Coords
meshio.write(rec_vtk_file_path,mesh_rec,'vtk-ascii')
