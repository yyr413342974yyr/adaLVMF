import numpy as np 
import _pickle  as pickle
import pyvista as pv
import itertools
from scipy.ndimage import zoom
import meshio
import imageio
import os

# from matplotlib import pyplot as plt

from scipy.ndimage import rotate
from scipy.ndimage.morphology import binary_fill_holes as bfh
from scipy.ndimage.morphology import binary_dilation
from skimage.measure import label as cc_label

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def rotation_tensor(theta, phi, psi, n_comps):
	# print(theta, phi, psi)
	# from https://discuss.pytorch.org/t/constructing-a-matrix-variable-from-other-variables/1529/3
	one = torch.ones(n_comps, 1, 1).to(device)
	zero = torch.zeros(n_comps, 1, 1).to(device)
	rot_x = torch.cat((
		torch.cat((one, zero, zero), 1),
		torch.cat((zero, theta.cos(), theta.sin()), 1),
		torch.cat((zero, -theta.sin(), theta.cos()), 1),
	), 2)
	rot_y = torch.cat((
		torch.cat((phi.cos(), zero, -phi.sin()), 1),
		torch.cat((zero, one, zero), 1),
		torch.cat((phi.sin(), zero, phi.cos()), 1),
	), 2)
	rot_z = torch.cat((
		torch.cat((psi.cos(), psi.sin(), zero), 1),
		torch.cat((-psi.sin(), psi.cos(), zero), 1),
		torch.cat((zero, zero, one), 1)
	), 2)
	# print(rot_x)
	return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))

def dice_loss(pred, target, slice_weights=1):

	# numerator = 2 * torch.sum(pred * target)
	# denominator = torch.sum(pred + target)
	# return 1- (numerator + 0.00001) / (denominator + 0.00001)


	# numerator = 2 * torch.sum(pred * target, axis=(0,1,2,3))
	# denominator = torch.sum(pred + target, axis=(0,1,2,3))
	# dloss = (numerator) / (denominator)

	# # dloss = (1 - dloss) * (slice_weights/torch.sum(slice_weights))
	# dloss = (1 - dloss)# * slice_weights

	# return torch.mean(dloss)


	numerator = 2 * torch.sum(pred * target, axis=(0,1,2,3))
	denominator = torch.sum(pred + target, axis=(0,1,2,3))
	dloss = (numerator + 0.00001) / (denominator + 0.00001)

	# dloss = (1 - dloss) * (slice_weights/torch.sum(slice_weights))
	dloss = (1 - dloss) * slice_weights

	return torch.mean(dloss)

def meshFittingLoss(pred, modes, global_shifts, slice_shifts, rots, target, axis_parameters,
	modes_weigth, 
	global_shifts_weight, 
	slice_shifts_weight, 
	rotations_weight, 
	myo_weight, 
	bp_weight,
	slice_weights=1, use_sdf=True):
	'''
	yyr 增加axis_parameters_loss
	'''
	

	if use_sdf:
		d_loss = torch.mean( (pred[:,:1] - target[:,:1])**2 )# + 100*(torch.mean(pred[:,:1]) - torch.mean(target[:,:1]))**2

		# #calculated (weighted) dice losses:
		# d0 = dice_loss(pred[:,:1], target[:,:1], slice_weights)*myo_weight
		# d1 = dice_loss(pred[:,1:], target[:,1:], slice_weights)*bp_weight
		# # print(d0, d1)
		# d_loss = d0+d1  + 1000*(torch.mean(pred[:,:1]) - torch.mean(target[:,:1]))**2

		modes_loss = torch.mean(modes**2)*modes_weigth*0.01

		global_shift_loss = torch.mean(global_shifts**2)*global_shifts_weight*0.01
		slice_shift_loss = torch.mean(slice_shifts**2)*slice_shifts_weight*0.01
		rotation_loss = torch.mean(rots**2)*rotations_weight*0.01

		# yyr 增加axis_parameters_loss
		axis_parameters_loss = torch.mean( axis_parameters**2 ) * 0.01

		# yyr 多返回一个axis_parameters_loss
		return d_loss, modes_loss, global_shift_loss, rotation_loss, slice_shift_loss, axis_parameters_loss


	else:
		#calculated (weighted) dice losses:
		d0 = dice_loss(pred[:,:1], target[:,:1], slice_weights)*myo_weight
		d1 = dice_loss(pred[:,1:], target[:,1:], slice_weights)*bp_weight
		# print(d0, d1)
		d_loss = d0+d1

	

	#calculate mode's largest squared distance from origin:
	# modes_loss = torch.max(modes**2)*modes_weigth
	modes_loss = torch.mean(modes**2)*modes_weigth

	global_shift_loss = torch.mean(global_shifts**2)*global_shifts_weight
	slice_shift_loss = torch.mean(slice_shifts**2)*slice_shifts_weight
	rotation_loss = torch.mean(rots**2)*rotations_weight

	# yyr 增加axis_parameters_loss
	axis_parameters_loss = torch.mean( axis_parameters**2 ) * 0.01

	# yyr 多返回一个axis_parameters_loss
	return d_loss, modes_loss, global_shift_loss, rotation_loss, slice_shift_loss, axis_parameters_loss


def MSE_on_gt(pred, modes, shifts, rots, target, 
	dice_weigth, 
	modes_weigth, 
	shifts_weight, 
	rotations_weight, 
	myo_weight, 
	bp_weight, 
	target_shifts, 
	target_rot, 
	target_modes, 
	srw):

	#calculated (weighted) dice losses:
	d0 = dice_loss(pred[:,:1],target[:,:1])*myo_weight
	d1 = dice_loss(pred[:,1:],target[:,1:])*bp_weight
	d_loss = d0+d1

	#calculate mode's largest squared distance from origin:
	modes_loss = torch.max(modes**2)*modes_weigth
	# modes_loss = torch.mean(modes**2)*modes_weigth

	slice_shift_loss = torch.mean(shifts**2)*shifts_weight
	rotation_loss = torch.mean(rots**2)*rotations_weight

	# if target_shifts is None:
	# 	shift_rot_penalty = 0
	# 	mode_penalty = 0
	# else:
	# 	shift_rot_penalty = srw*torch.mean(rots - target_rot)**2 + srw*torch.mean(shifts - target_shifts)**2
	# 	mode_penalty = (srw*torch.mean(modes - target_modes)**2) * 50
	
	return d_loss + modes_loss + slice_shift_loss + rotation_loss
	# return 2*dice_weigth*(d0+d1)/(myo_weight+bp_weight) + torch.max(modes**2*modes_weigth) + torch.mean(shifts**2)*shifts_weight + torch.mean(rots**2)*rotations_weight + shift_rot_penalty + mode_penalty


def enlargeImage(im, scale=4):
	return np.repeat(np.repeat( im, repeats=scale, axis=0), repeats=scale, axis=1)

def makeErrorImage(gt_slices, pred_slices):

	# print(gt_slices.shape, pred_slices.shape)
	# sys.exit()

	error = np.zeros((gt_slices.shape[0],gt_slices.shape[1],3))
	error[...,0] = gt_slices*(1-pred_slices)
	error[...,1] = gt_slices*pred_slices
	error[...,2] = pred_slices*(1-gt_slices)
	error_im = np.concatenate([ np.tile(gt_slices[...,None], (1,1,3)), np.tile(pred_slices[...,None], (1,1,3)), error], axis=0)

	return (error_im*255).astype('uint8')

def ampsToPoints(amps, PHI=None):

	num_modes = amps.shape[0]
	
	if PHI is None:
		try:
			proj_matrix_path = os.path.join(model_dir,'ML_augmentation/Projection_matrix.dat')
			with open(proj_matrix_path,'rb') as infile:
				PHI = pickle.load(infile)
		except:
			proj_matrix_path = os.path.join(model_dir,'ML_augmentation/Projection_matrix.npy')
			PHI = np.load(proj_matrix_path)
		PHI = np.asmatrix(PHI[:,:num_modes])

	# if PHI is None:
		# proj_matrix_path = 'ShapeModel/ML_augmentation/Projection_matrix.dat'
		# with open(proj_matrix_path,'rb') as infile:
			# PHI = pickle.load(infile)
	proj_back = PHI.dot(amps)

	# yyr 调整一下shape
	if len(proj_back.shape) == 1:
		num_shape = proj_back.shape[0]
	elif len(proj_back.shape) == 2:
		num_shape = proj_back.shape[1]
	proj_back = proj_back.reshape(1, num_shape)


	n_points = proj_back.shape[1]//3    # 4800多个点
	# return proj_back
	Rec_Coords      = np.zeros((proj_back.shape[1]//3,3))
	# return proj_back
	Rec_Coords      = np.zeros((n_points,3))
	Rec_Coords[:,0] = np.array(proj_back)[0,:][0:n_points]
	Rec_Coords[:,1] = np.array(proj_back)[0,:][n_points:2*n_points]
	Rec_Coords[:,2] = np.array(proj_back)[0,:][2*n_points:3*n_points]

	assert np.array_equal(Rec_Coords, proj_back.reshape((3,-1)).T)
     

	return Rec_Coords

def voxelizeUniform(mesh, resolution, is_epi_pts, gridsize=128, offset=(70,70,128), bp_channel=False, blur=False):

	#mesh should be a mesh object
	meshio.write('test_tmp.vtk', mesh)
	model = pv.read('test_tmp.vtk')
	grid = pv.create_grid(model, dimensions=resolution)


	grid.spacing = (0.001, 0.001, 0.001) 
	grid.origin = (-grid.spacing[0]*resolution[0]/2, -grid.spacing[1]*resolution[1]/2, -offset[2])   
	sampled = grid.sample(model)

	array = sampled.get_array('vtkValidPointMask')#sampled.array_names[4])
	binary = np.reshape(np.round(array), sampled.dimensions, order='F').astype(bool)


	if bp_channel:

		filled = []
		for k in range(binary.shape[-1]):

			#we are moving through long-axis slices starting from bellow the apex, moving up towards the base, stopping at the valve plane.
			#we should see: 
			#0 or more empty slices,
			#0 or more slices with just myo,
			#1 or more slices with both myo and bp,
			#0 or more empty slices,
			# in that order.

			# 内膜，内外模
			if is_epi_pts == False:            
				filled.append( np.logical_xor(bfh(binary[...,k]), binary[...,k]) )
            
			# yyr   外膜  
			if is_epi_pts:
				binary[...,k] = bfh(binary[...,k])
				filled.append(binary[...,k])

			#for the slices towards the base the myocardium may not be a closed circle (and thus filling the hole wont work)
			#we catch those cases here and fill them. Note that this filling hack wont work if the first non-closed slice is 
			#very different from the last closed slice. However, for the current shape model it is ok.
			#[TODO] -> generalise the bloodpool filling method
			if k > 0 and np.sum(filled[k]) == 0 and np.sum(filled[k-1]) != 0:
				
				f = filled[k-1]+0
				small_bp = np.logical_xor(f, np.logical_and(f, binary[...,k]))
				filled[k] = np.logical_xor(bfh(np.logical_or(small_bp, binary[...,k])), binary[...,k])


			if np.sum(binary[...,k]) == 0:
				filled[k] = filled[k]*0

			'''
			#assert that together the myo and bp form a single connected component with no holes:
			myo_and_bp = np.logical_or(filled[k], binary[...,k])
			single_cc = (cc_label(myo_and_bp, return_num=True )[1] <= 1)
			assert single_cc
			no_holes = (np.sum(bfh(myo_and_bp)) == np.sum(myo_and_bp))
			assert no_holes

			#assert that: if the previous slice has a bloodpool and this slice has myo, then this slice also has bloodpool:
			if k > 0:
				if np.sum(filled[k-1])>0 and np.sum(binary[...,k]):
					assert np.sum(filled[k-1])>0
			'''

		filled = np.moveaxis(np.array(filled),0,-1)
        
		# yyr 内膜
		if is_epi_pts == False:   
			binary = filled

		return binary, filled



	return binary

def loadSSMV2(num_modes, cp_frequency, model_dir, is_epi_pts):

	#load the "average" mesh model:
	mean_mesh_file = os.path.join(model_dir,'Mean/LV_mean.vtk')
	mesh_1 = meshio.read(mean_mesh_file)
	#load the projection matrix:
	try:
		proj_matrix_path = os.path.join(model_dir,'ML_augmentation/Projection_matrix.dat')
		with open(proj_matrix_path,'rb') as infile:
			PHI = pickle.load(infile)
	except:
		proj_matrix_path = os.path.join(model_dir,'ML_augmentation/Projection_matrix.npy')
		PHI = np.asmatrix(np.load(proj_matrix_path))

	PHI = PHI[:,:num_modes] 

	n_points = PHI.shape[0]//3
	#load the mode bounds:
	try:
		mode_bounds = np.loadtxt(os.path.join(model_dir,'boundary_coeffs.txt'))
		mode_bounds = mode_bounds[:num_modes,:]
	except:
		mode_bounds = np.load(os.path.join(model_dir,'boundary_coeffs.npy'))
		mode_bounds = mode_bounds[:num_modes,:]
	# modes = []
	# for i in range(num_modes):
	# 	modes.append(meshio.read(os.path.join(model_dir,'Modes/Mode%d.vtk'% (i,))))

	#get mesh internal and external (i.e. surface) node indecies:
	# epi_pts是心外膜，endo_pts是心内膜
	epi_pts = np.load(os.path.join(model_dir,'Boundary_nodes/EPI_points.npy'))
	endo_pts = np.load(os.path.join(model_dir,'Boundary_nodes/ENDO_points.npy'))
	
	# exterior_points_index = np.concatenate([epi_pts,endo_pts])
	'''
     yyr 
     心外膜 epi_pts     [3603 3604 3605 ... 4761 4762 4763]
     心内膜 endo_pts    [   0    1    2 ... 1158 1159 1160]
    '''   
	if is_epi_pts:
		exterior_points_index = epi_pts
	else:        
		exterior_points_index = endo_pts   
    
    
	#take a subset of the control points based on cp_frequency (take one in every cp_frequency points):
	exterior_points_index = exterior_points_index[::cp_frequency]
	# starting_cp = mesh_1.points[exterior_points_index]

	#create the PCA matrix for a reduced set of points and reduced number of modes:
	PHI3 = np.reshape(np.array(PHI), (n_points,3,num_modes), order='F') #these values are (num_mesh_points, 3=xyz, num_modes) for full mesh model
	PHI3 = PHI3[exterior_points_index,:,:num_modes]
	PHI3 = np.reshape(PHI3, (-1,num_modes), order='F')

	#get the modes for the mean mesh: 也就是mode的均值
	mean_modes = np.dot( meshio.read(mean_mesh_file).points.reshape((-1,), order='F'), np.array(PHI))

	#the offset required to bring the mesh into "pixel space"
	
	# mesh_offset = np.array([70,70,128]) #possibly better for just SAX slices
	mesh_offset = np.array([0, 0, 0.11])
	starting_cp = mesh_1.points[exterior_points_index] + mesh_offset

	#mesh keypoints: (used for initially orienting the mesh)
	mesh_vpc = np.array([0.9,0.,0.])
	mesh_sax_normal = np.array([1.,0.,0.])
	mesh_rv_direction = np.array([0.,-0.75,-1.])
	mesh_axes = [mesh_vpc, mesh_sax_normal, mesh_rv_direction]

	return mesh_1, starting_cp, PHI3, PHI, mode_bounds, mean_modes, mesh_offset, exterior_points_index, mesh_axes

def bbox2_ND(img):
	N = img.ndim
	out = []
	for ax in itertools.combinations(reversed(range(N)), N - 1):
		nonzero = np.any(img, axis=ax)
		out.extend(np.where(nonzero)[0][[0, -1]])
	return tuple(out)

def loadMMWHS(sz=64):

	examps = []
	for k in range(1,21):
		examp = np.load('mmwhs_preprocessed/mmwhs_%d.npy' % (k,))[...,0]
		examp = np.moveaxis(examp, 0,2)
		examp[...,2:] = examp[...,2:][...,::-1]
		print(examp.shape)
		examp[...,0] = np.rot90(examp[...,0],-1)
		examp[...,1] = np.rot90(examp[...,1],-1)
		examp = examp[None]
		examp = zoom(examp, (1,0.5, 0.5, 1))
		examps.append(examp)

	# current_sz_x = examps[0].shape[1]
	# current_sz_y = examps[0].shape[2]
	# for k in range(len(examps)):
	# 	examps[k] = zoom(examps[k], (sz/current_sz_x, sz/current_sz_y, 1, 1), order=1)

	return np.array(examps)

# def loadAllACDC():
# 	labels = np.load('ACDC_ES_128x128_labels.npy', allow_pickle=True)
# 	examps, hr_examps, example_mm_resolution = loadTrainingData(labels, num_slices, sz)

def loadACDC():

	import nibabel as nib
	img = nib.load('/media/tom/scratch/data/ACDC_DataSet/training/patient001/patient001_4d.nii.gz')

	data = img.get_fdata()
	hdr = img.header

	return data, hdr

def loadTrainingData(num_slices, sz, dataset='acdc', bp_channel=False):

	#prepare the training data:
	#	1. make sure each volume has the required number of slices (slice_num)
	#	2. center the myocardium mask in the volume's (x,y) plane
	#	3. resize to the given number of voxels (sz)

	num_chans = 1
	if bp_channel:
		num_chans = 2

	slice_thickness = {1:10.0,2:10.0,3:10.0,4:10.0,5:10.0,6:10.0,7:10.0,8:10.0,9:10.0,10:10.0,11:10.0,12:10.0,13:10.0,14:10.0,15:10.0,16:10.0,17:10.0,18:10.0,19:10.0,20:10.0,21:10.0,22:10.0,23:10.0,24:10.0,25:10.0,26:10.0,27:10.0,28:10.0,29:10.0,30:10.0,31:10.0,32:10.0,33:10.0,34:10.0,35:5.0,36:10.0,37:10.0,38:10.0,39:10.0,40:10.0,41:10.0,42:10.0,43:6.5,44:10.0,45:10.0,46:10.0,47:10.0,48:10.0,49:10.0,50:10.0,51:10.0,52:10.0,53:10.0,54:10.0,55:10.0,56:10.0,57:10.0,58:10.0,59:10.0,60:10.0,61:10.0,62:10.0,63:10.0,64:10.0,65:10.0,66:10.0,67:10.0,68:10.0,69:10.0,70:10.0,71:10.0,72:10.0,73:10.0,74:10.0,75:5.0,76:10.0,77:10.0,78:10.0,79:10.0,80:10.0,81:5.0,82:5.0,83:10.0,84:5.0,85:5.0,86:10.0,87:10.0,88:5.0,89:10.0,90:10.0,91:10.0,92:5.0,93:7.0,94:5.0,95:5.0,96:5.0,97:10.0,98:10.0,99:5.0,100:10.0}

	if dataset == 'acdc' or dataset == 'acdc_ES':
		labels = np.load('../ACDC_ES_128x128_labels.npy', allow_pickle=True)
	if dataset == 'acdc_ED':

		from scipy.ndimage.measurements import center_of_mass as com

		# labels = np.load('../ACDC_ES_128x128_labels.npy', allow_pickle=True)
		# print(labels[0].shape)
		labels = np.load('../ACDC_ED_labels.npy', allow_pickle=True)
		# print(labels[0].shape)
		for i in range(100):
			_,cx,cy = com(labels[i][...,2])
			cx,cy=np.round(cx).astype('int'),np.round(cy).astype('int')
			labels[i] = labels[i][:,cx-64:cx+64,cy-64:cy+64]
	# 	print(labels[0].shape)
	# 	sys.exit()

	# print(labels[0].shape)

	examps = []
	current_sz_x = labels[0].shape[1]
	current_sz_y = labels[0].shape[2]
	for k in range(len(labels)):

		examp = labels[k]
		if bp_channel:
			examp = np.round(rotate(examp, -45, axes=(1,2), reshape=False)[:,:,:,1:3]).astype('int')
		else:
			examp = np.round(rotate(examp, -45, axes=(1,2), reshape=False)[:,:,:,1:2]).astype('int')

		if examp.shape[0] > num_slices:
			examp = examp[:num_slices]

		examp = np.concatenate([examp,np.zeros((num_slices-examp.shape[0],current_sz_x,current_sz_y,num_chans))], axis=0)
		examp = np.moveaxis(examp, 0,2)
		examp = examp[:,:,::-1]

		#center the example volume in-plane:
		r = bbox2_ND( 1*(examp>0))
		examp = np.roll(examp, shift=int(current_sz_x/2-(r[0]+r[1])/2), axis=0)
		examp = np.roll(examp, shift=int(current_sz_y/2-(r[2]+r[3])/2), axis=1)

		examps.append(examp)


	hr_examps = np.array(examps) + 0

	for k in range(len(examps)):
		examps[k] = zoom(examps[k], (sz/current_sz_x, sz/current_sz_y, 1, 1), order=1)
	examps = np.array(examps)

	example_mm_resolution = (-1, -1, 10)

	examps = np.moveaxis(examps, 4, 1)

	return examps, hr_examps, slice_thickness