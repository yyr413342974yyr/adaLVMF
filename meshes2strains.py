import meshio
import os
import numpy as np
import sys
import vtk
from vtk import vtkUnstructuredGridReader
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from matplotlib import pyplot as plt
import re

def sorted_nicely( l ): 
	""" Sort the given iterable in the way that humans expect.""" 
	convert = lambda text: int(text) if text.isdigit() else text 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)

def calculateStrains(mesh_model_dir, mesh_folder, strain_mesh_folder, plot_folder, time_frames):

	initial_mesh = meshio.read( os.path.join(mesh_folder,"mesh_t=0.vtk") )
	for i in time_frames:

		current_mesh = meshio.read( os.path.join(mesh_folder,"mesh_t=%d.vtk" % (i,)) )

		f = os.path.join(strain_mesh_folder,"mesh_t=%d.vtk" % (i,))
		meshio.write(f, initial_mesh, file_format='vtk')
		os.system('meshio-ascii %s' % (f,))
		outFile = open(f,'a')

		outFile.write('VECTORS displacement float \n')
		for i, p in enumerate(current_mesh.points):
			dx, dy, dz = current_mesh.points[i] - initial_mesh.points[i]
			outFile.write('%1.4f %1.4f %1.4f\n' % (dx, dy, dz))
		outFile.close()

	mean_mesh = meshio.read( os.path.join(mesh_model_dir,'Mean','LV_mean.vtk') )

	mwd_vtk_path = os.path.join(strain_mesh_folder,'mean_with_displacements.vtk')
	meshio.write(mwd_vtk_path, mean_mesh, file_format='vtk')
	os.system('meshio-ascii ' + mwd_vtk_path)
	outFile = open(mwd_vtk_path,'a')

	outFile.write('VECTORS displacement float \n')
	for i, p in enumerate(mean_mesh.points):
		dx, dy, dz = initial_mesh.points[i] - mean_mesh.points[i]
		# print(dx, dy, dz)
		outFile.write('%1.4f %1.4f %1.4f\n' % (dx, dy, dz))
	outFile.close()

	#save to vtk_file_path

	# from vtk import *
	# from vtk.util.numpy_support import vtk_to_numpy
	# from vtk.util.numpy_support import numpy_to_vtk

	vtk_file_path = mwd_vtk_path
	vtk_out_path = os.path.join(strain_mesh_folder,'initial.vtk')
	displ_field_name = 'displacement'

	# Read vtk written from VMTK
	reader = vtkUnstructuredGridReader()
	reader.SetFileName(vtk_file_path)
	reader.ReadAllScalarsOn()
	reader.ReadAllVectorsOn()
	reader.Update()
	data = reader.GetOutput()

	n_points  = int(data.GetNumberOfPoints())
	n_cells   = data.GetNumberOfCells()
	Coords_0  = vtk_to_numpy(data.GetPoints().GetData()).copy()
	displ_field = vtk_to_numpy(data.GetPointData().GetArray(displ_field_name)) 
	x_l = vtk_to_numpy(data.GetPointData().GetArray('x_l')) 
	e_t = vtk_to_numpy(data.GetPointData().GetArray('e_t')) 
	e_l = vtk_to_numpy(data.GetPointData().GetArray('e_l')) 
	e_c = vtk_to_numpy(data.GetPointData().GetArray('e_c')) 

	Els = [] # line elements
	# Get elements from VTK
	for i in range(n_cells):
		n_nodes_el   = data.GetCell(i).GetPointIds().GetNumberOfIds()
		connectivity = [0]*n_nodes_el
		for n_sel in range(n_nodes_el):
			connectivity[n_sel] = int(data.GetCell(i).GetPointId(n_sel)) 
		Els.append(connectivity)

	Els = np.array(Els)

	data.GetPointData().SetActiveVectors(displ_field_name)

	grad = vtk.vtkGradientFilter()
	grad.SetInputData(data)
	grad.SetInputScalars(0,displ_field_name) # Zero b/c point data
	grad.SetResultArrayName("F")
	grad.Update()
	dmmy = grad.GetOutput()
	gradU = vtk_to_numpy(dmmy.GetPointData().GetArray('F'))

	DefGradient = np.zeros((n_points,3,3))
	#DefGradient
	DefGradient[:,0,0] = gradU[:,0].reshape(1,-1) + 1
	DefGradient[:,0,1] = gradU[:,1].reshape(1,-1)
	DefGradient[:,0,2] = gradU[:,2].reshape(1,-1)
	DefGradient[:,1,0] = gradU[:,3].reshape(1,-1)
	DefGradient[:,1,1] = gradU[:,4].reshape(1,-1) + 1
	DefGradient[:,1,2] = gradU[:,5].reshape(1,-1)
	DefGradient[:,2,0] = gradU[:,6].reshape(1,-1)
	DefGradient[:,2,1] = gradU[:,7].reshape(1,-1)
	DefGradient[:,2,2] = gradU[:,8].reshape(1,-1) + 1

	#check volume, should be the case that 
	# for d in DefGradient:
	#     print(np.linalg.det(d))# > 0 ?

	Coords = mean_mesh.points

	outFile = open(vtk_out_path,'w')
	outFile.write('# vtk DataFile Version 4.0\n')
	outFile.write('vtk output\n')
	outFile.write('ASCII\n')
	outFile.write('DATASET UNSTRUCTURED_GRID \n')
	outFile.write('POINTS '+str(Coords.shape[0])+' float\n')
	for j in range(Coords.shape[0]):
		outFile.write(str(Coords[j,0]+displ_field[j,0])+' ')
		outFile.write(str(Coords[j,1]+displ_field[j,1])+' ')
		outFile.write(str(Coords[j,2]+displ_field[j,2])+' ')
		outFile.write('\n')
	outFile.write( 'CELLS ' + str( Els.shape[0] ) + ' ' + str( (Els.shape[0]) * 5 ) )
	outFile.write('\n')
	for k in range( Els.shape[0] ):
		outFile.write( '4 ' )
		for j in range( 4 ):
			outFile.write( str( Els[k,j]) + ' ' )
		outFile.write('\n')
	# write cell types
	outFile.write( '\n\nCELL_TYPES ' + str( Els.shape[0] ) )
	for k in range( Els.shape[0] ):
		outFile.write( '\n10' )
	outFile.write('\nPOINT_DATA '+str(Coords.shape[0])+'\n')

	outFile.write('SCALARS x_l float \n')
	outFile.write('LOOKUP_TABLE x_l')
	for k in range(Coords.shape[0]):
		outFile.write('%1.4f \n' %(x_l[k],) )

	outFile.write('VECTORS e_t float \n')
	for k in range(Coords.shape[0]):
		e_t_new = DefGradient[k,:,:].dot(e_t[k,:].T)/np.linalg.det(DefGradient[k,:,:])
		outFile.write('%1.4f %1.4f %1.4f \n' %(e_t_new[0],e_t_new[1],e_t_new[2]) )

	outFile.write('VECTORS e_l float \n')
	for k in range(Coords.shape[0]):
		e_l_new = DefGradient[k,:,:].dot(e_l[k,:].T)/np.linalg.det(DefGradient[k,:,:])
		outFile.write('%1.4f %1.4f %1.4f \n' %(e_l_new[0],e_l_new[1],e_l_new[2]) )

	outFile.write('VECTORS e_c float \n')
	for k in range(Coords.shape[0]):
		e_c_new = DefGradient[k,:,:].dot(e_c[k,:].T)/np.linalg.det(DefGradient[k,:,:])
		outFile.write('%1.4f %1.4f %1.4f \n' %(e_c_new[0],e_c_new[1],e_c_new[2]) )
	outFile.close()



	def ComputeLagrangianStrainComponent(E, dir_sel):

		n_points = E.shape[0]
		strain_  = np.zeros((n_points,1))

		p0 = E[:,0,0]*dir_sel[:,0] + E[:,0,1]*dir_sel[:,1] + E[:,0,2]*dir_sel[:,2]
		p1 = E[:,1,0]*dir_sel[:,0] + E[:,1,1]*dir_sel[:,1] + E[:,1,2]*dir_sel[:,2]
		p2 = E[:,2,0]*dir_sel[:,0] + E[:,2,1]*dir_sel[:,1] + E[:,2,2]*dir_sel[:,2]

		strain_ = p0*dir_sel[:,0] + p1*dir_sel[:,1] + p2*dir_sel[:,2]

		return strain_

	# os.path.join(strain_mesh_folder,"mesh_t=%d.vtk" % (i,))
	cases_folder         = strain_mesh_folder
	vtk_for_directions   = vtk_out_path
	displ_field_name     = 'displacement'
	out_file             = 'output.txt' #text file for results

	cases  = sorted_nicely(next(os.walk(cases_folder))[2])

	print(cases)
	# sys.exit()

	# Read vtk written from VMTK
	reader = vtkUnstructuredGridReader()
	reader.SetFileName(vtk_for_directions)
	reader.ReadAllScalarsOn()
	reader.ReadAllVectorsOn()
	reader.Update()
	data = reader.GetOutput()

	n_points  = int(data.GetNumberOfPoints())
	n_cells   = data.GetNumberOfCells()
	Coords_0  = vtk_to_numpy(data.GetPoints().GetData()).copy()
	
	x_l = vtk_to_numpy(data.GetPointData().GetArray('x_l')) 


	e_t = vtk_to_numpy(data.GetPointData().GetArray('e_t')) 
	e_l = vtk_to_numpy(data.GetPointData().GetArray('e_l')) 
	e_c = vtk_to_numpy(data.GetPointData().GetArray('e_c')) 

	#x_l = vtk_to_numpy(data.GetPointData().GetArray('x_l'))
	#then can get index in certain range

	Els = [] # line elements
	# Get elements from VTK
	for i in range(n_cells):
		n_nodes_el   = data.GetCell(i).GetPointIds().GetNumberOfIds()
		connectivity = [0]*n_nodes_el
		for n_sel in range(n_nodes_el):
			connectivity[n_sel] = int(data.GetCell(i).GetPointId(n_sel)) 
		Els.append(connectivity)

	Els = np.array(Els)
	f_out = open(out_file,'w')

	# cases = ['p%d.vtk' % i for i in range(25)]

	Rs,Ls,Cs=[],[],[]

	for ind, case_name in enumerate(cases):

		if case_name in ['initial.vtk', 'mean_with_displacements.vtk']:
			continue
		if case_name[0] == '.':
			continue

		ML_vtk = vtk.vtkUnstructuredGridReader()
		print(cases_folder+'/'+case_name)
		ML_vtk.SetFileName(cases_folder+'/'+case_name)
		ML_vtk.ReadAllVectorsOn()
		ML_vtk.ReadAllScalarsOn()
		ML_vtk.Update()
		ML_  = ML_vtk.GetOutput() 

		ML_.GetPointData().SetActiveVectors(displ_field_name)
		displ_new = vtk_to_numpy(ML_.GetPointData().GetArray(displ_field_name))

		grad = vtk.vtkGradientFilter()
		grad.SetInputData(ML_)
		grad.SetInputScalars(0,displ_field_name) # Zero b/c point data
		grad.SetResultArrayName("F")
		grad.Update()
		dmmy = grad.GetOutput()
		gradU = vtk_to_numpy(dmmy.GetPointData().GetArray('F'))

		DefGradient = np.zeros((n_points,3,3))
		#DefGradient
		DefGradient[:,0,0] = gradU[:,0].reshape(1,-1) + 1
		DefGradient[:,0,1] = gradU[:,1].reshape(1,-1)
		DefGradient[:,0,2] = gradU[:,2].reshape(1,-1)
		DefGradient[:,1,0] = gradU[:,3].reshape(1,-1)
		DefGradient[:,1,1] = gradU[:,4].reshape(1,-1) + 1
		DefGradient[:,1,2] = gradU[:,5].reshape(1,-1)
		DefGradient[:,2,0] = gradU[:,6].reshape(1,-1)
		DefGradient[:,2,1] = gradU[:,7].reshape(1,-1)
		DefGradient[:,2,2] = gradU[:,8].reshape(1,-1) + 1

		#check volume, should be the case that np.linalg.det(DefGradient[i]) > 0

		C        = np.zeros((n_points,3,3))
		C[:,0,0] = DefGradient[:,0,0]*DefGradient[:,0,0] + DefGradient[:,1,0]*DefGradient[:,1,0] + DefGradient[:,2,0]*DefGradient[:,2,0]
		C[:,0,1] = DefGradient[:,0,0]*DefGradient[:,0,1] + DefGradient[:,1,0]*DefGradient[:,1,1] + DefGradient[:,2,0]*DefGradient[:,2,1]
		C[:,0,2] = DefGradient[:,0,0]*DefGradient[:,0,2] + DefGradient[:,1,0]*DefGradient[:,1,2] + DefGradient[:,2,0]*DefGradient[:,2,2]

		C[:,1,0] = DefGradient[:,0,1]*DefGradient[:,0,0] + DefGradient[:,1,1]*DefGradient[:,1,0] + DefGradient[:,2,1]*DefGradient[:,2,0]
		C[:,1,1] = DefGradient[:,0,1]*DefGradient[:,0,1] + DefGradient[:,1,1]*DefGradient[:,1,1] + DefGradient[:,2,1]*DefGradient[:,2,1]
		C[:,1,2] = DefGradient[:,0,1]*DefGradient[:,0,2] + DefGradient[:,1,1]*DefGradient[:,1,2] + DefGradient[:,2,1]*DefGradient[:,2,2]

		C[:,2,0] = DefGradient[:,0,2]*DefGradient[:,0,0] + DefGradient[:,1,2]*DefGradient[:,1,0] + DefGradient[:,2,2]*DefGradient[:,2,0]
		C[:,2,1] = DefGradient[:,0,2]*DefGradient[:,0,1] + DefGradient[:,1,2]*DefGradient[:,1,1] + DefGradient[:,2,2]*DefGradient[:,2,1]
		C[:,2,2] = DefGradient[:,0,2]*DefGradient[:,0,2] + DefGradient[:,1,2]*DefGradient[:,1,2] + DefGradient[:,2,2]*DefGradient[:,2,2] 

		E        = C
		E[:,0,0] -= 1.0
		E[:,1,1] -= 1.0
		E[:,2,2] -= 1.0
		
		E = E/2.0

		radial_strain = ComputeLagrangianStrainComponent(E, e_t)
		long_strain   = ComputeLagrangianStrainComponent(E, e_l)
		circ_strain   = ComputeLagrangianStrainComponent(E, e_c)

		#if you want to filter to a certain region the do: np.mean(radial_strain[indicies])

		for sl in np.linspace(0,1,10):
			central_inds = np.where( np.abs(x_l-sl) < 0.1 )
			print(np.mean(radial_strain[central_inds]), np.std(radial_strain[central_inds]))

		f_out.write('%1.4f %1.4f %1.4f ' %(np.mean(radial_strain),np.mean(long_strain),np.mean(circ_strain)))
		f_out.write('%1.4f %1.4f %1.4f\n ' %(np.std(radial_strain),np.std(long_strain),np.std(circ_strain)))

		Rs.append(np.mean(radial_strain))
		Ls.append(np.mean(long_strain))
		Cs.append(np.mean(circ_strain))

		#radial [0 to -0.5 ish at peak]
		#longditudinal [0 to 0.3 ish at peak]
		#longditudinal [0 to 0.3 ish at peak]
			
	f_out.close()

	Xs = list(time_frames)
	plt.plot(Xs,Rs, label='radial')
	plt.plot(Xs,Ls, label='long')
	plt.plot(Xs,Cs, label='circ')
	plt.legend()
	plt.title('strains')
	plt.savefig(os.path.join(plot_folder,'strains.png'))