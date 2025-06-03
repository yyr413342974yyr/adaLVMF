import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np

import pyvista as pv
import imageio
import meshio
import os

import _pickle  as pickle

from matplotlib import pyplot as plt

from scipy.ndimage import shift
import sys

from helper_functions import ampsToPoints, voxelizeUniform, makeErrorImage, loadSSMV2

import itertools
from scipy.ndimage.measurements import center_of_mass as com

def mkdir(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)

def slicewiseDice(arr1,arr2):
	arr1 = np.squeeze(arr1)
	arr2 = np.squeeze(arr2)

	slice_dice, has_target = [], []
	for i in range(len(arr1)):

		slice_dice.append( 2 * np.sum(arr1[i] * arr2[i]) / (np.sum(arr1[i]) + np.sum(arr2[i]) + 0.00001 ) )
		has_target.append( (np.sum(arr2[i])>0)*1 )


	return np.array(slice_dice), np.array(has_target)

def transformMeshAxes(mesh_axes, vol_shifts_out, myR):
	[mesh_vpc, mesh_sax_normal, mesh_rv_direction] = mesh_axes
	mesh_vpc_transformed  = np.squeeze(np.dot(mesh_vpc - vol_shifts_out, myR.T)*64)
	mesh_sax_normal_transformed  = np.squeeze(np.dot(mesh_sax_normal, myR.T)*64)
	mesh_rv_direction_transformed  = np.squeeze(np.dot(mesh_rv_direction, myR.T)*64)
	mesh_sax_normal_transformed = mesh_sax_normal_transformed / np.linalg.norm(mesh_sax_normal_transformed)
	mesh_rv_direction_transformed = mesh_rv_direction_transformed / np.linalg.norm(mesh_rv_direction_transformed)
	mesh_axes_transformed = [mesh_vpc_transformed, mesh_sax_normal_transformed, mesh_rv_direction_transformed]
	return mesh_axes_transformed

def getRotationMatrix(A, B):
	'''rotation matrix that rotates vector A to align with vector B'''
	A = A / np.linalg.norm(A)
	B = B / np.linalg.norm(B)
	c = np.dot(A,B)
	v = np.cross(A,B)
	v_x = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
	v_x_squared = np.dot(v_x,v_x)
	rotM = np.eye(3) + v_x + v_x_squared*(1/(1+c))
	return rotM

def getSlices(se, mesh, sz, use_bp_channel, mesh_offset, learned_inputs, ones_input, is_epi_pts):
	
	modes_output, global_offsets, x_shifts, y_shifts, global_rotations,_ = learned_inputs(ones_input)
	per_slice_offsets = torch.cat([y_shifts*0, y_shifts, x_shifts], dim=-1)

	if use_bp_channel:
		mean_arr, mean_bp_arr = voxelizeUniform(mesh, (sz,sz,sz), is_epi_pts=is_epi_pts, bp_channel=use_bp_channel, offset=mesh_offset)
		mean_arr_batch = torch.Tensor(np.concatenate([mean_arr[None,None], mean_bp_arr[None,None]], axis=1)).to(device)
	else:
		mean_arr = voxelizeUniform(mesh, (sz,sz,sz), is_epi_pts=is_epi_pts, bp_channel=use_bp_channel, offset=mesh_offset)
		mean_arr_batch = torch.Tensor(np.tile(mean_arr[None,None],(1,1,1,1,1))).to(device)
	return se([mean_arr_batch, global_offsets, per_slice_offsets, global_rotations])

def bbox2_ND(img):
	N = img.ndim
	out = []
	for ax in itertools.combinations(reversed(range(N)), N - 1):
		nonzero = np.any(img, axis=ax)
		out.extend(np.where(nonzero)[0][[0, -1]])
	return tuple(out)

def loadMMWHSvol(i, sz=128):

	arr = np.load('../MMWHS/p%d/label.npy' % (i,))[0]
	arr = np.rot90(arr, 2, (1,2))	
	arr_c = np.zeros(arr.shape+(3,), dtype='int')

	for k in range(1,4):
		arr_c[...,k-1] = (arr==k)*1
	r = np.prod(np.sum(arr_c, axis=(1,2)) >= 1000, axis=-1)*1

	arr = arr[np.where(r==1)[0][0]:]
	arr_c = arr_c[np.where(r==1)[0][0]:]

	_,_,xl,xu,yl,yu = bbox2_ND(arr==3)
	arr = arr[:, int((xl+xu)/2)-64:int((xl+xu)/2)+64, int((yl+yu)/2)-64:int((yl+yu)/2)+64]
	arr_c = arr_c[:, int((xl+xu)/2)-64:int((xl+xu)/2)+64, int((yl+yu)/2)-64:int((yl+yu)/2)+64]

	arr = (arr==2)*1

	z,x,y = com(arr)
	x, y = int(x), int(y)
	if arr.shape[0] < 128:
		arr = np.concatenate([arr,np.zeros((128-arr.shape[0],)+arr.shape[1:])], axis=0)
		arr_c = np.concatenate([arr_c,np.zeros((128-arr_c.shape[0],)+arr_c.shape[1:])], axis=0)

	if sz == 64:
		arr = arr[::2,::2,::2]
		arr_c = arr_c[::2,::2,::2]

	arr = np.moveaxis(arr, 0, 2)
	arr_c = np.moveaxis(arr_c, 0, 2)
	arr = arr[:,:,::-1]
	arr_c = arr_c[:,:,::-1]

	return arr# np.rot90(arr, 3, (0,1))	

def progress(errors, patience):

	# if we haven't done the minimum number of updates yet:
	if len(errors) < patience:
		return 1
	return errors[-1]-errors[-patience]


def getSlicesAtPositions(arr, slice_positions, global_shift=(0,0,0), slice_shifts=None, sz=128):
	''' accepts a 3D array and a set of slice position, and extractes the specified slices '''

	if slice_shifts != None:
		assert len(slice_shifts) == len(slice_positions)

	arr = np.round(shift(arr*1., global_shift, order=1)).astype('int')

	slices = []
	for j, sp in enumerate(slice_positions):

		arr_s = arr+0
		if slice_shifts != None:
			arr_s = shift(arr, slice_shifts[j], order=1)

		if sp[:3] == (0,90,90): #short axis slice
			i = int((sz-1)*(sp[3]+1)/2)
			slices.append( arr_s[...,i:i+1] )
		elif sp[:3] == (0,0,0):
			slices.append( np.moveaxis(arr_s[sz//2:sz//2+1], 0, -1) )
		elif sp[:3] == (90,0,0):
			slices.append( np.moveaxis(arr_s[:,sz//2:sz//2+1], 1, -1) )

	return np.concatenate(slices, axis=-1)


def randomMesh(num_modes=25):
	mesh,_,_,_,mode_bounds,mode_means,_,_ = loadSSMV2(num_modes, cp_frequency=50) #cp_frequency not actually used for theses required outputs
	normed_modes = np.concatenate([np.random.randn(num_modes)/3., np.zeros((25-num_modes,))]) #we do /3. so majority of samples are in range [-1,1]
	rescaled_modes = normed_modes*(mode_bounds[:,1] - mode_bounds[:,0])/2 + mode_means
	mesh.points = ampsToPoints(rescaled_modes)
	return mesh


class evalLearnedInputs():

	def __init__(self, learned_inputs, mode_bounds, mode_means, mesh, PHI):

		from copy import deepcopy

		self.learned_inputs = learned_inputs
		self.num_modes = learned_inputs.num_modes
		self.ones_input = torch.Tensor(np.ones((1,1))).to(device)
		self.mode_bounds = mode_bounds
		self.mode_means = mode_means
		self.mesh = deepcopy(mesh)
		self.PHI = PHI+0

	def __call__(self, just_mesh=True):

		with torch.no_grad():
			modes_output, volume_shift, x_shifts, y_shifts, volume_rotations,_ = self.learned_inputs(self.ones_input)
			modes_output = modes_output.cpu().numpy()
			volume_shift = volume_shift.cpu().numpy()[0]
			x_shifts = x_shifts.cpu().numpy()
			y_shifts = y_shifts.cpu().numpy()
			volume_rotations = volume_rotations.cpu().numpy()[0,0]

		normed_modes = np.concatenate([modes_output[0], np.zeros((25-self.num_modes,))])     
		rescaled_modes = normed_modes*(self.mode_bounds[:,1] - self.mode_bounds[:,0])/2 + self.mode_means
		gt_cp = ampsToPoints(rescaled_modes, self.PHI)
		self.mesh.points = gt_cp

		if just_mesh:
			return self.mesh

		return self.mesh, modes_output
		# return self.mesh, volume_shift, x_shifts, y_shifts, volume_rotations,


def saveMesh(filename, eli):

	# num_modes = learned_inputs.num_modes
	# ones_input = torch.Tensor(np.ones((1,1))).to(device)

	# with torch.no_grad():
	# 	modes_output, volume_shift, x_shifts, y_shifts, volume_rotations = learned_inputs(ones_input)
	# 	modes_output = modes_output.cpu().numpy()

	# normed_modes = np.concatenate([modes_output[0], np.zeros((25-num_modes,))])
	# rescaled_modes = normed_modes*(mode_bounds[:,1] - mode_bounds[:,0])/2 + mode_means
	# gt_cp = ampsToPoints(rescaled_modes)
	# mesh_1.points = gt_cp

	meshio.write(filename+'.vtk', eli())

def diceForAlign(gt, pred):

	gt_present = (np.sum(gt,(1,2), keepdims = True) > 1)*1.
	pred = pred * gt_present
	dice_score = np.sum(gt*pred)*2 / (np.sum(gt) + np.sum(pred))

	return dice_score

def gridpad(input_im, sub_im_size, pad_size, add_centeral_lines):

	#covert sub_im_size to a tuple inrequired:
	try:
		iter(sub_im_size)
	except TypeError: # not an iterable, assume a single value and thus square image
		sub_im_size = (sub_im_size,sub_im_size)

	#check the image passed is compatible with the stated sub_im_size:
	assert input_im.shape[0]%sub_im_size[0] == 0
	assert input_im.shape[1]%sub_im_size[1] == 0

	rows = input_im.shape[0]//sub_im_size[0]
	cols = input_im.shape[1]//sub_im_size[1]

	if add_centeral_lines:
		input_im[sub_im_size[0]//2::(sub_im_size[0])] = np.max(input_im)/2
		input_im[:,sub_im_size[1]//2::(sub_im_size[1])] = np.max(input_im)/2

	#make a new image of the required shape:
	padded_im = np.zeros( ((rows-1)*pad_size+input_im.shape[0], (cols-1)*pad_size+input_im.shape[1], input_im.shape[2]) )
	padded_im += np.max(input_im)

	for i in range(rows):
		for j in range(cols):
			sx,ex,sy,ey = i*(sub_im_size[0]+pad_size),i*(sub_im_size[0]+pad_size)+sub_im_size[0],j*(sub_im_size[1]+pad_size),j*(sub_im_size[1]+pad_size)+sub_im_size[1]
			print(sx,ex,sy,ey)
			padded_im[sx:ex,sy:ey] = input_im[i*sub_im_size[0]:(i+1)*sub_im_size[0], j*sub_im_size[1]:(j+1)*sub_im_size[1]]

	

	return padded_im


def evaluateOnLearnedInputExampleReAlign(input_data, gt_data, eli, sz, slice_positions, li_model, mean_arr_batch, mode='full', dice_for='approx', add_cenrtral_grid=True):

	#generate the mesh:
	mesh_1, volume_shift, x_shifts, y_shifts, volume_rotations = eli(just_mesh=False)
	#perform global shift:
	global_shift = -volume_shift[0,::-1]*(sz/2)
	mesh_1.points += global_shift
	#calculate slice shifts:
	slice_shifts = [(-x_shifts[0,i,0]*(sz/2), -y_shifts[0,i,0]*(sz/2), 0) for i in range(y_shifts.shape[1])]

	#voxelize the mesh
	arr = voxelizeUniform(mesh_1, (sz,sz,sz), bp_channel=False)

	#get the required slices, including slice shifts:
	vpm = getSlicesAtPositions(arr, slice_positions, slice_shifts=slice_shifts)
	#get the volume ready for visulaising (by reordering dimensions):
	vpm = np.moveaxis(np.squeeze(vpm), 2, 0)

	#get the required slices, without slice shifts:

	vpm_noshift = getSlicesAtPositions(arr, slice_positions)
	#get the volume ready for visulaising (by reordering dimensions):
	vpm_noshift = np.moveaxis(np.squeeze(vpm_noshift), 2, 0)

	#get the ground truth (input) data, which includes offsets:
	lk_input = input_data.cpu().numpy()[0,0]
	lk_input = np.moveaxis(lk_input, 2, 0)

	#get the ground truth (original) data:
	lk_gt = gt_data.cpu().numpy()[0,0]
	lk_gt = np.moveaxis(lk_gt, 2, 0)

	#make 'corrected input' by applying opposit shifts to the lk_input data:

	lk_input_corrected = lk_input + 0 
	for i in range(2,8):
		dx = np.round(slice_shifts[i][0]).astype('int')
		dy = np.round(slice_shifts[i][1]).astype('int')
		lk_input_corrected[i] = np.roll(np.roll(lk_input_corrected[i], -dx, 0), -dy, 1)

	#find the best offset:
	vpm_noshift_chunk = vpm_noshift[2:]
	target_chunk = lk_gt[2:]

	_, gx, gy = com(vpm_noshift_chunk)
	_, px, py = com(target_chunk)

	dx = np.round(gx - px).astype('int')
	dy = np.round(gy - py).astype('int')

	vpm_noshift_chunk = np.roll(np.roll(vpm_noshift_chunk, -dx, 1), -dy, 2)

	best_dice = diceForAlign(target_chunk, vpm_noshift_chunk)
	best_dxdy = (0,0)

	resgrid = []
	for dx in range(-25,26):
		resgrid.append([])
		for dy in range(-25,26):
			vpm_noshift_chunk_temp = np.roll(np.roll(vpm_noshift_chunk+0, dx, 1), dy, 2)
			this_dice = diceForAlign(target_chunk, vpm_noshift_chunk_temp)
			resgrid[-1].append(this_dice)
			if this_dice > best_dice:
				best_dxdy = (dx,dy)
				best_dice = this_dice
				print(best_dice)

	# plt.imshow(resgrid)
	# plt.colorbar()
	# plt.savefig('resgrid.png')

	print('best offset =', best_dxdy)
	
	vpm_noshift[2:] = np.roll(np.roll(vpm_noshift_chunk, best_dxdy[0], 1), best_dxdy[1], 2)
	# lk_input_corrected[2:] = np.roll(np.roll(lk_input_corrected[2:], best_dxdy[0], 1), best_dxdy[1], 2)



	#get relative offsets:
	# for i in range(2,len(vpm)):

	# 	gx, gy = com(lk_input[i])
	# 	px, py = com(vpm[i])

	# 	tdx, tdy = 0, 0
	# 	if not np.isnan(px+gx):
	# 		tdx = gx-px
	# 		tdy = gy-py

	# 	gx, gy = com(lk_gt[i])
	# 	px, py = com(vpm_noshift[i])

	# 	dx, dy = 0, 0
	# 	if not np.isnan(px+gx):
	# 		dx = gx-px
	# 		dy = gy-py

	# 	ddx = np.round(tdx - dx).astype('int')
	# 	ddy = np.round(tdy - dy).astype('int')

	# 	vpm_noshift[i] = np.roll(np.roll(vpm_noshift[i], ddx, 1), ddy, 0)



	#get the actual output from the network, i.e. the differentiable approximation to vpm
	temp = 0 #not currently used..
	pred_k = li_model([mean_arr_batch[:1], eli.ones_input, temp])[0].cpu().numpy()
	pred_k = pred_k[:,:1]
	pred_k = np.moveaxis(np.squeeze(pred_k), 2, 0)

	#calculate the dice scores:
	gt = lk_input
	if dice_for == 'approx': 
		pred = pred_k
	else:
		pred = vpm
	gt_present = (np.sum(gt,(1,2), keepdims = True) > 1)*1.
	pred = pred * gt_present
	dice_i = np.sum(gt*pred)*2 / (np.sum(gt) + np.sum(pred))


	#we have to work out the global shit required to align the mesh without slice shifts to the input without slice shifts:
	# print( com(vpm[0]) )
	# print( com(vpm_noshift[0]) )


	#make the image:
	gt_slices_shifted = np.concatenate(lk_input, axis=1)
	gt_slices_shifted_corrected = np.concatenate(lk_input_corrected, axis=1)
	gt_slices = np.concatenate(lk_gt, axis=1)
	pred_slices_shifted = np.concatenate(vpm, axis=1)
	pred_slices = np.concatenate(vpm_noshift, axis=1)

	row_1 = np.tile(gt_slices[...,None],(1,1,3)) #original (uncorrupted) segmentations
	row_2 = np.tile(gt_slices_shifted[...,None],(1,1,3)) #input segmentations (corrupted with slice shifts)
	row_3 = np.tile(pred_slices[...,None],(1,1,3))
	row_4 = np.tile(gt_slices_shifted_corrected[...,None],(1,1,3))

	# error_im_1 = makeErrorImage(gt_slices_shifted, pred_slices_shifted)
	# error_im_2 = makeErrorImage(gt_slices, pred_slices)
	# error_im_3 = makeErrorImage(gt_slices, gt_slices_shifted_corrected)

	error_im = np.concatenate([ row_1, row_2, row_3, row_4], axis=0)

	error_im = gridpad(error_im, (128,128), 10, True)

	# error_im = np.concatenate([ error_im_1, error_im_2, error_im_3], axis=0)

	return dice_i, error_im


	# pred_k_slices = np.concatenate(pred_k, axis=1)
	# error_im_2 = makeErrorImage(gt_slices, pred_k_slices)
	# error_im_3 = makeErrorImage(pred_slices, pred_k_slices)
	# error_im = np.concatenate([ error_im_1, error_im_2, error_im_3], axis=0)

	# if add_cenrtral_grid:
	# 	for i in range(error_im.shape[0]//sz):
	# 		error_im[i*sz+sz//2] = 100
	# 	for i in range(error_im.shape[1]//sz):
	# 		error_im[:,i*sz+sz//2] = 100

	# return dice_i, error_im



def evaluateOnLearnedInputExample(exa, eli, sz, slice_positions, li_model, mean_arr_batch, mode='full', dice_for='approx', add_cenrtral_grid=True):

	'''
	ones_input = torch.Tensor(np.ones((1,1))).to(device)
	num_modes = learned_inputs.num_modes

	#get current predictions (as numpy arrays):
	modes_output, volume_shift, x_shifts, y_shifts, volume_rotations = learned_inputs(ones_input)
	modes_output = modes_output.cpu().numpy()
	volume_shift = volume_shift.cpu().numpy()[0]
	volume_rotations = volume_rotations.cpu().numpy()[0,0]
	x_shifts = x_shifts.cpu().numpy()
	y_shifts = y_shifts.cpu().numpy()

	#create a mesh from the predicted modes:
	normed_modes = np.concatenate([modes_output[0], np.zeros((25-num_modes,))])
	rescaled_modes = normed_modes*(mode_bounds[:,1] - mode_bounds[:,0])/2 + mode_means
	gt_cp = ampsToPoints(rescaled_modes)
	mesh_1.points = gt_cp
	'''

	mesh_1, volume_shift, x_shifts, y_shifts, volume_rotations = eli(just_mesh=False)

	sc = sz/2
	global_shift = -volume_shift[0,::-1]*sc
	slice_shifts = [(-x_shifts[0,i,0]*sc, -y_shifts[0,i,0]*sc, 0) for i in range(y_shifts.shape[1])]

	mesh_1.points += global_shift

	# print(np.mean(mesh_1.points, axis=0))
	
	#voxelize the mesh, and get the required slices:
	# arr, bp_arr = voxelizeUniform(mesh_1, (sz,sz,sz), bp_channel=True)
	arr = voxelizeUniform(mesh_1, (sz,sz,sz), bp_channel=False)
	vpm = getSlicesAtPositions(arr, slice_positions, slice_shifts=slice_shifts)
	# vpm = np.clip(getSlicesAtPositions(arr, slice_positions, global_shift=global_shift, slice_shifts=slice_shifts) + getSlicesAtPositions(arr, slice_positions)*0.5, 0, 1)
	# vpm = np.clip(getSlicesAtPositions(arr, slice_positions, global_shift=global_shift) + getSlicesAtPositions(arr, slice_positions)*0.25, 0, 1)
	# vpm = getSlicesAtPositions(arr, slice_positions, global_shift=global_shift)
	vpm = np.moveaxis(np.squeeze(vpm), 2, 0)

	
	# volume_shift = volume_shift[:,[2,1,0]]
	# mesh_1.points -= volume_shift*sz#( (sz-1)/2 ) #not 100% sure if it should be sz-1 or sz
	# mesh_1.points -= volume_shift[:,::-1]*( (sz-1)/2 ) #not 100% sure if it should be sz-1 or sz
	
	# #voxelize the mesh, and get the required slices:
	# arr = voxelizeUniform(mesh_1, (sz,sz,sz))
	# vpm_shifted = getSlicesAtPositions(arr, slice_positions)
	# vpm_shifted = np.moveaxis(np.squeeze(vpm_shifted), 2, 0)

	# vpm = np.clip(vpm*0.5 + vpm_shifted, 0, 1)
	

	#get the ground truth (target) data:
	lk = exa.cpu().numpy()[0,0]

	# print('---->',lk.shape)


	lk = np.moveaxis(lk, 2, 0)

	lk_corrected = lk + 0

	# for slice_i in range(2,8):
	# 	shift_x, shift_y = -gt_shifts[slice_i-2][0], -gt_shifts[slice_i-2][1]
	# 	shift_x = np.round(shift_x).astype('int')
	# 	shift_y = np.round(shift_y).astype('int')
	# 	lk_corrected[slice_i] = np.roll(np.roll(lk[slice_i], shift_x*0, axis=0), shift_y, axis=1)

	# lk_corrected = np.clip(lk_corrected*0.5 + lk, 0, 1)

	# for slice_i in range(2,8):
	# 	shift_x, shift_y = x_shifts[0,slice_i,0], y_shifts[0,slice_i,0]
	# 	shift_x = np.round(shift_x).astype('int')
	# 	shift_y = np.round(shift_y).astype('int')
	# 	vpm[slice_i] = np.roll(np.roll(vpm[slice_i], -shift_y, axis=0), -shift_x, axis=1)

	#get the actual output from the network, i.e. the differentiable approximation to vpm
	temp = 0
	pred_k = li_model([mean_arr_batch[:1], eli.ones_input, temp])[0].cpu().numpy()
	# print(pred_k.shape)
	pred_k = pred_k[:,:1]
	pred_k = np.moveaxis(np.squeeze(pred_k), 2, 0)

	

	#calculate the dice scores:
	gt = lk
	if dice_for == 'approx':
		pred = pred_k
	else:
		pred = vpm
	gt_present = (np.sum(gt,(1,2), keepdims = True) > 1)*1.


	pred = pred * gt_present
	dice_i = np.sum(gt*pred)*2 / (np.sum(gt) + np.sum(pred))

	#make the (simple or full) image:
	gt_slices = np.concatenate(lk, axis=1)
	gt_slices_corrected = np.concatenate(lk_corrected, axis=1)
	pred_slices = np.concatenate(vpm, axis=1)
	error_im_1 = makeErrorImage(gt_slices_corrected, pred_slices)
	if mode == 'simple':
		return dice_i, error_im_1
	pred_k_slices = np.concatenate(pred_k, axis=1)
	error_im_2 = makeErrorImage(gt_slices, pred_k_slices)
	error_im_3 = makeErrorImage(pred_slices, pred_k_slices)
	error_im = np.concatenate([ error_im_1, error_im_2, error_im_3], axis=0)

	if add_cenrtral_grid:
		for i in range(error_im.shape[0]//sz):
			error_im[i*sz+sz//2] = 100
		for i in range(error_im.shape[1]//sz):
			error_im[:,i*sz+sz//2] = 100

	return dice_i, error_im


def getSlicePositions(mean_arr_slice_thickness, data_slice_thickness, sz, num_slices, add_long_axis=False):

	mean_arr_slice_positions = np.array([mean_arr_slice_thickness*slice_index for slice_index in range(sz)])
	slice_indicies = []
	for slice_index in range(num_slices):
		target_position = slice_index * data_slice_thickness
		i = np.argmin( np.abs(mean_arr_slice_positions-target_position) ) # get index of closest slice in mean_arr
		slice_indicies.insert(0, sz-1-i)

	slice_positions = []
	if add_long_axis:
		slice_positions += [(90,0,0,0,0,0), (0,0,0,0,0,0)]
	for k in slice_indicies:
		slice_positions.append( (0,90,90,2*(k/(sz-1))-1,0,0) )

	return slice_positions
