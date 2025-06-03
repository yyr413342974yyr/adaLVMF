import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from interpolate_spline import interpolate_spline
from scipy.spatial.transform import Rotation as R

from helper_functions import rotation_tensor

from rotanimate import rotanimate

import sys

mycolors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:gray', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',  'tab:olive', 'tab:cyan']

def make_not_trainable(m):
	for param in m.parameters():
		param.requires_grad = False

def makeSliceCoordinateSystems(dicom_exam):
	slice_coordinate_systems = []
	for s in dicom_exam:
		in_plane_1 = s.orientation[:3]
		in_plane_2 = s.orientation[3:]
		out_of_plane = np.cross(in_plane_1, in_plane_2)
		for sl in range(s.slices):
			slice_coordinate_systems.append([in_plane_1, in_plane_2, out_of_plane])
	slice_coordinate_systems = np.array(slice_coordinate_systems)
	return slice_coordinate_systems

class SliceExtractor(torch.nn.Module):

	def __init__(self, input_volume_shape, dicom_exam, allow_global_shift_xy=False, allow_global_shift_z=False, allow_slice_shift=False, allow_rotations=False, series_to_exclude=[]):

		super(SliceExtractor, self).__init__()

		self.vol_shape = input_volume_shape
		self.allow_shifts = allow_global_shift_xy or allow_global_shift_z
		self.allow_global_shift_xy = allow_global_shift_xy
		self.allow_global_shift_z = allow_global_shift_z
		self.allow_slice_shifts = allow_slice_shift
		self.allow_rotations = allow_rotations
		self.initial_alignment_rotation = torch.Tensor(np.zeros(3)).to(device)
		# self.initial_mesh_offset = torch.Tensor(np.zeros(3)).to(device)

		slices = []
		for k, s in enumerate(dicom_exam):
			if not s.name in series_to_exclude:
				# print(s.XYZs[0].shape)
				# print(s.XYZs[0])
				# sys.exit()
				slices.extend(s.XYZs)
		self.num_slices = len(slices)
		self.grid = np.concatenate(slices)[None]

		# self.unit_x = torch.Tensor(np.array([[[1,0,0]]])).to(device)
		# self.unit_y = torch.Tensor(np.array([[[0,1,0]]])).to(device)
		# self.unit_z = torch.Tensor(np.array([[[0,0,1]]])).to(device)

		# print(dicom_exam[0].orientation)

		self.coordinate_system = torch.Tensor(makeSliceCoordinateSystems(dicom_exam)).to(device)

		# self.coordinate_system = torch.Tensor(np.array([[[1,0,0],[0,1,0],[0,0,1]]])).to(device)
		# center = np.mean(self.grid,axis=1,keepdims=True)

		center = dicom_exam.center

		self.grid = (self.grid - center)/64
		# self.grid = self.grid[:,:,::-1]+0
		self.grid = torch.Tensor(self.grid).to(device)

		# print('self.grid.shape', self.grid.shape)

	def forward(self, args):
			
		vol, global_offsets, per_slice_offsets, global_rotations = args

		batched_grid = self.grid.repeat(global_offsets.shape[0],1,1)
		# print('batched_grid.shape', batched_grid.shape)

		r_weight = 0
		if self.allow_rotations:
			r_weight = 1
		R = rotation_tensor(
			global_rotations[...,:1]*r_weight + self.initial_alignment_rotation[0],
			global_rotations[...,1:2]*r_weight + self.initial_alignment_rotation[1],
			global_rotations[...,2:]*r_weight + self.initial_alignment_rotation[2],
		1)

		# rotated_unit_x = torch.bmm(self.unit_x, R)
		# rotated_unit_y = torch.bmm(self.unit_y, R)
		# rotated_unit_z = torch.bmm(self.unit_z, R)
		# print(self.coordinate_system.shape, R.shape)
		rotated_coords = torch.bmm(self.coordinate_system, R.repeat(self.coordinate_system.shape[0],1,1))
		# print(rotated_coords)

		batched_grid = torch.bmm(batched_grid, R)

		# print('x', self.unit_x, '->', rotated_unit_x, torch.sum(rotated_unit_x**2))
		# print('y', self.unit_y, '->', rotated_unit_y, torch.sum(rotated_unit_y**2))
		# print('z', self.unit_z, '->', rotated_unit_z, torch.sum(rotated_unit_z**2))
		# print('coords', self.coordinate_system, '->', rotated_coords)


		if self.allow_shifts:
			if not self.allow_global_shift_xy:
				global_offsets[:,:,1] = 0
				global_offsets[:,:,2] = 0
			if not self.allow_global_shift_z:
				global_offsets[:,:,0] = 0
			batched_grid += global_offsets

		if self.allow_slice_shifts:

			# print(batched_grid.shape, per_slice_offsets.shape)
			# print(batched_grid[0,:3], per_slice_offsets[0,:3])
			# print(rotated_coords.shape, per_slice_offsets.shape)

			# per_slice_offsets *= 0
			# per_slice_offsets[:,0,0] = 1
			# per_slice_offsets[:,1,1] = 1
			# per_slice_offsets[:,2,2] = 1

			batched_grid = batched_grid.view(1,self.num_slices,-1,3)
			# print(per_slice_offsets.shape, rotated_coords.shape )
			# print(rotated_coords.shape, per_slice_offsets[0,...,None].shape )
			# print(batched_grid.shape, torch.bmm(rotated_coords, per_slice_offsets[0,...,None])[None,:,None,:,0].shape )
			# batched_grid += per_slice_offsets.unsqueeze(2) 
			# batched_grid += torch.bmm(per_slice_offsets[0,...,None], rotated_coords).unsqueeze(2) 
			# batched_grid += torch.bmm(rotated_coords, per_slice_offsets[0,...,None]).unsqueeze(2) 
			# batched_grid += torch.bmm(rotated_coords, per_slice_offsets[0,...,None])[None,:,None,:,0] 

			# print(per_slice_offsets[0,:,None].shape, rotated_coords.shape, torch.bmm(per_slice_offsets[0,:,None], rotated_coords).shape, batched_grid.shape)
			# sys.exit()

			batched_grid += torch.bmm(per_slice_offsets[0,:,None], rotated_coords)[None,:]
			batched_grid = batched_grid.view(1,-1,3)
	
		batched_grid = batched_grid.view(-1,self.num_slices,self.vol_shape[0],self.vol_shape[1],3)
		batched_grid = batched_grid.permute(0,2,3,1,4)
		res = torch.nn.functional.grid_sample(vol, batched_grid, align_corners=True, mode='bilinear')

		return res

class GivenPointSliceSamplingSplineWarpSSM(torch.nn.Module):

	def __init__(self, input_volume_shape, control_points, dicom_exam, allow_global_shift_xy=False, allow_global_shift_z=False, allow_slice_shift=False, allow_rotations=False, series_to_exclude=[]):

		super(GivenPointSliceSamplingSplineWarpSSM, self).__init__()

		verbose = False

		self.vol_shape = input_volume_shape
		self.control_points = torch.Tensor(control_points).to(device)

		self.allow_shifts = allow_global_shift_xy or allow_global_shift_z
		self.allow_global_shift_xy = allow_global_shift_xy
		self.allow_global_shift_z = allow_global_shift_z
		self.allow_slice_shifts = allow_slice_shift
		self.allow_rotations = allow_rotations
		self.de_sax_normal = torch.Tensor(dicom_exam.sax_normal).to(device)

		self.initial_alignment_rotation = torch.Tensor(np.zeros(3)).to(device)
		# self.initial_mesh_offset = torch.Tensor(np.zeros(3)).to(device)

		self.sz = self.vol_shape[0] # assume it is the same shape in all directions

		slices = []
		for k, s in enumerate(dicom_exam):
			if not s.name in series_to_exclude:
				slices.extend(s.XYZs)
		self.num_slices = len(slices)

		self.grid = np.concatenate(slices)[None] # now has shape (1, num_slices x sz^2, 3)
		
		# center = np.mean(self.grid,axis=1,keepdims=True)
		center = dicom_exam.center
		self.grid = (self.grid - center)/64

		# self.grid = self.grid[:,:,::-1]+0

		if verbose: 
			
			#make an rotating gif showing the slice positions:
			from matplotlib import pyplot as plt
			from mpl_toolkits.mplot3d import Axes3D
			fig = plt.figure()
			ax = Axes3D(fig)
			for k, s in enumerate(dicom_exam):
				for sl in s.XYZs:
					ptsx = [sl[0,0]/64, sl[127,0]/64, sl[-1,0]/64, sl[-128,0]/64, sl[0,0]/64]
					ptsy = [sl[0,1]/64, sl[127,1]/64, sl[-1,1]/64, sl[-128,1]/64, sl[0,1]/64]
					ptsz = [sl[0,2]/64, sl[127,2]/64, sl[-1,2]/64, sl[-128,2]/64, sl[0,2]/64]
					ax.plot(ptsx, ptsy, ptsz, color=mycolors[k%len(mycolors)])
			angles = np.linspace(0,360,51)[:-1]
			rotanimate(ax, angles, 'testerfig.gif', delay=10) 
			
			#print some details:
			print('%d slices in total' % (self.num_slices,))
			print(center, dicom_exam.center, '(should be equal)')
			print('X range = ', self.grid[...,0].min(), self.grid[...,0].max(), 'Y range = ', self.grid[...,1].min(), self.grid[...,1].max(), 'Z range = ', self.grid[...,2].min(), self.grid[...,2].max())
			print('self.grid.shape = ', self.grid.shape)
			print('allow_shifts = ', self.allow_shifts)
			print('allow_global_shift_xy = ', self.allow_global_shift_xy)
			print('allow_global_shift_z = ', self.allow_global_shift_z)
			print('allow_slice_shifts = ', self.allow_slice_shifts)
			print('allow_rotations = ', self.allow_rotations)


		self.grid = torch.Tensor(self.grid).to(device)

		# self.coordinate_system = torch.Tensor(np.array([[[1,0,0],[0,1,0],[0,0,1]]])).to(device)
		self.coordinate_system = torch.Tensor(makeSliceCoordinateSystems(dicom_exam)).to(device)


	def forward(self, args):
			
		warped_control_points, vol, global_offsets, per_slice_offsets, global_rotations, temp = args
		#duplicated grid and control_points for each input in the batch:
		batched_grid = self.grid.repeat(warped_control_points.shape[0],1,1) #shape = (bs, num_slices * sz^2, 3)
		batched_control_points = self.control_points.repeat(warped_control_points.shape[0],1,1)
       
        
		# warped_control_points = batched_control_points

		### perform the global and per-slice transforms:

		r_weight = 0
		if self.allow_rotations:
			r_weight = 1
		R = rotation_tensor(
			global_rotations[...,:1]*r_weight + self.initial_alignment_rotation[0],
			global_rotations[...,1:2]*r_weight + self.initial_alignment_rotation[1],
			global_rotations[...,2:]*r_weight + self.initial_alignment_rotation[2],
		1)
		batched_grid = torch.bmm(batched_grid, R)

		# rotated_coords = torch.bmm(self.coordinate_system, R)
		rotated_coords = torch.bmm(self.coordinate_system, R.repeat(self.coordinate_system.shape[0],1,1))


		if self.allow_shifts:
			#add the global shift to the whole grid
			if not self.allow_global_shift_xy:
				#zero-out in plane shift
				global_offsets[:,:,1] = 0
				global_offsets[:,:,2] = 0
			if not self.allow_global_shift_z:
				#zero-out in plane shift
				global_offsets[:,:,0] = 0
			batched_grid += global_offsets

		if self.allow_slice_shifts:
			#add the per-slice offsets to each grid slice
			batched_grid = batched_grid.view(1,self.num_slices,-1,3)
			# batched_grid += per_slice_offsets.unsqueeze(2) 
			# batched_grid += torch.bmm(per_slice_offsets, rotated_coords).unsqueeze(2)
			# batched_grid += torch.bmm(rotated_coords, per_slice_offsets[0,...,None])[None,:,None,:,0] 
			batched_grid += torch.bmm(per_slice_offsets[0,:,None], rotated_coords)[None,:]
			batched_grid = batched_grid.view(1,-1,3)

            
		# 原来的代码  
		interpolated_sample_locations = interpolate_spline(train_points=warped_control_points.flip(-1)*2-1,
														   train_values=batched_control_points.flip(-1)*2-1,
														   query_points=batched_grid,
														   order=1).view(-1,self.num_slices,self.vol_shape[0],self.vol_shape[1],3)
        
        
        
		interpolated_sample_locations = interpolated_sample_locations.permute(0,2,3,1,4) #why do we have this strange ordering?

		#sample the mean volume at the calculated sample locations:
		res = torch.nn.functional.grid_sample(vol, interpolated_sample_locations, align_corners=True, mode='bilinear')

		return res




class PCADecoder(torch.nn.Module):
	'''
	This module maps modes to mesh, but has to also take care of normalizations and offsets.
	Takes as inputs normalized modes, de-normalizes them, then uses the POD model to produce a mesh.
	It then scales and shifts the resulting mesh, and retruns it.
	'''
	
	def __init__(self, num_modes, num_points, mode_bounds, mode_means, offset, trainable=False, scale=128,):
		super(PCADecoder, self).__init__()
		self.mode_spans = torch.Tensor((mode_bounds[:,1] - mode_bounds[:,0])/2).to(device)
		self.mode_means = torch.Tensor(mode_means).to(device)
		self.offset = torch.Tensor(offset).to(device)
		self.scale = scale
		self.num_points = num_points

		self.fc1 = torch.nn.Linear(num_modes, 3*num_points) # <-- these weights get set to the POD matrix
		
	def forward(self, x, axis_parameters):
		batch_size = x.shape[0]
		x = x * self.mode_spans + self.mode_means
		mesh_points = self.fc1(x)
		mesh_points = mesh_points.view(batch_size, 3, self.num_points)
		mesh_points = torch.transpose(mesh_points, 1, 2)
		mesh_points = (mesh_points + self.offset) / self.scale
		# print("看看mesh_points的shape: ", mesh_points.shape)

		# yyr 在这里乘这个meshpoints 
		# 把这句话注释了，就不使用axis_parameters
		mesh_points = torch.mul(mesh_points, axis_parameters.unsqueeze(1).unsqueeze(1)).squeeze(0)  

		
		# print("看看axis_parameters: ", axis_parameters)
		# =========================================================================

		return mesh_points



# yyr axis_parameters_Decoder
class axis_parameters_Decoder(torch.nn.Module):
	def __init__(self, PHI, PHI3, num_modes):
		super(axis_parameters_Decoder, self).__init__()
		self.PHI = PHI
		self.PHI3 = PHI3
		self.num_modes = num_modes
		
	def forward(self, axis_parameters):
		axis_parameters_numpy = axis_parameters.detach().cpu().numpy()

		# print("看看axis_parameters：", axis_parameters_numpy)

		self.PHI3 = np.reshape(np.array(self.PHI3), (self.PHI3.shape[0]//3, 3, self.num_modes), order='F')
		self.PHI3[:,0,:] = self.PHI3[:,0,:] * axis_parameters_numpy[0][0]
		self.PHI3[:,1,:] = self.PHI3[:,1,:] * axis_parameters_numpy[0][1]
		self.PHI3[:,2,:] = self.PHI3[:,2,:] * axis_parameters_numpy[0][2]
		self.PHI3 = np.reshape(self.PHI3, (-1, self.num_modes), order="F")

		self.PHI = np.reshape(np.array(self.PHI), (self.PHI.shape[0]//3, 3, self.num_modes), order='F')
		self.PHI[:,0,:] = self.PHI[:,0,:] * axis_parameters_numpy[0][0]
		self.PHI[:,1,:] = self.PHI[:,1,:] * axis_parameters_numpy[0][1]
		self.PHI[:,2,:] = self.PHI[:,2,:] * axis_parameters_numpy[0][2]
		self.PHI = np.reshape(self.PHI, (-1, self.num_modes), order="F")

		return self.PHI, self.PHI3


class learnableInputs(torch.nn.Module):
	'''
	Creates a network that maps an input of a single 1 to the inputs required for the ParameterisedPlaneSliceSamplingSplineWarpSSM
	does this with learnable weights so essentially you can use it to 'learn the inputs'
	(there may well be a better way to do this in pytorch...)
	'''

	def __init__(self, num_modes=12, num_slices=10):
		super(learnableInputs, self).__init__()
		self.modes_output_layer = torch.nn.Linear(1, num_modes)
		self.volume_shift_layer = torch.nn.Linear(1, 3)
		self.x_shift_layer = torch.nn.Linear(1, num_slices)
		self.y_shift_layer = torch.nn.Linear(1, num_slices)
		self.volume_rotations_layer = torch.nn.Linear(1, 3)
		self.num_slices = num_slices
		self.num_modes = num_modes

		# yyr 增加axis_parameters_layer
		self.axis_parameters_layer = torch.nn.Linear(1, 3)
		self.axis_parameters_layer.weight.data.fill_(1.)
		self.axis_parameters_layer.bias.data.fill_(0.)


	def forward(self, x):
		# print('x', x)
		batch_size = x.shape[0]
		modes_output = self.modes_output_layer(x)
		volume_shift = self.volume_shift_layer(x).view(batch_size, 1, 3)
		x_shifts = self.x_shift_layer(x).view(batch_size, self.num_slices, 1)
		y_shifts = self.y_shift_layer(x).view(batch_size, self.num_slices, 1)
		volume_rotations = self.volume_rotations_layer(x).view(batch_size, 1, 3)

		# yyr 增加axis_parameters,试试relu层
		axis_parameters = self.axis_parameters_layer(x)
		# axis_parameters = torch.relu(axis_parameters)
		

		return (modes_output, volume_shift, x_shifts, y_shifts, volume_rotations, axis_parameters)


class SimpleCNN(torch.nn.Module):
	'''
	A standard neural network used to map input volumes to mesh modes and offsets etc
	This is trained for the fixed sized input amortized version of the mesh predictor
	(currently expects single channel inputs of size 128x128x8, i.e. 8 128x128 slices)
	'''
	
	def __init__(self, num_modes=12, num_slices=10):

		super(SimpleCNN, self).__init__()

		self.convolutional_layers = []

		self.conv1 = torch.nn.Conv3d(1, 8, kernel_size=(3,3,1), stride=(2,2,1), padding=0)
		self.conv2 = torch.nn.Conv3d(8, 8, kernel_size=(3,3,1), stride=(2,2,1), padding=0)
		self.conv3 = torch.nn.Conv3d(8, 8, kernel_size=(3,3,1), stride=(2,2,1), padding=0)
		self.conv4 = torch.nn.Conv3d(8, 8, kernel_size=(3,3,1), stride=1, padding=0)

		im_size = 13*13*8
		
		self.fc1 = torch.nn.Linear(8*im_size, 56)
		self.fc2 = torch.nn.Linear(56, 28)
		self.fc3 = torch.nn.Linear(28, 28)

		self.modes_output_layer = torch.nn.Linear(28, num_modes)
		self.volume_shift_layer = torch.nn.Linear(28, 3)
		self.x_shift_layer = torch.nn.Linear(28, num_slices)
		self.y_shift_layer = torch.nn.Linear(28, num_slices)
		self.volume_rotations_layer = torch.nn.Linear(28, 3)

		self.num_slices = num_slices
		
	def forward(self, x):
		
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		# print(x.shape)
		
		batch_size = x.shape[0]
		x = x.view(batch_size,-1)
		
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		
		modes_output = self.modes_output_layer(x)
		volume_shift = self.volume_shift_layer(x).view(batch_size, 1, 3)
		x_shifts = self.x_shift_layer(x).view(batch_size, self.num_slices, 1)
		y_shifts = self.y_shift_layer(x).view(batch_size, self.num_slices, 1)

		global_rotations = self.volume_rotations_layer(x).view(batch_size, 1, 3)

		#no shift on long axis
		# volume_shift[:,:,0] = -0.3
		# volume_shift[:,:,1] = -0.3
		# volume_shift[:,:,2] = 0

		# print(volume_shift)

		return (modes_output, volume_shift, x_shifts, y_shifts, global_rotations)


class TestModel(torch.nn.Module):
	'''
	Used for testing the warpiung and slicing. 
	Takes as input mesh modes and parapmeters for the warping and slicing and then just applies them and returns the resulting slice images
	'''
	
	def __init__(self, pcaD, warp_and_slice_model):
		super(TestModel, self).__init__()
		self.pcaD = pcaD
		self.warp_and_slice_model = warp_and_slice_model
		
	def forward(self, args):
		voxelized_mean_mesh, modes_output, global_shift, slice_shifts, global_rotations = args
		predicted_cp = self.pcaD(modes_output)
		predicted_slices = self.warp_and_slice_model([predicted_cp, voxelized_mean_mesh, global_shift, slice_shifts, global_rotations])

		return predicted_slices

class FullPPModel(torch.nn.Module):
	'''
	Stiches together the parameter predicting neural network (SimpleCNN), the PCA decoder network (PCADecoder), and the rendering and slicing
	module (ParameterisedPlaneSliceSamplingSplineWarpSSM) to form a network that takes a stack of slices as input and returns a stack of slices 
	resulting from the mesh model as output. Can then be trained to try and reconstruct the input data.
	'''
	
	def __init__(self, mp_model, pcaD, warp_and_slice_model):

		super(FullPPModel, self).__init__()

		self.mp_model = mp_model
		self.pcaD = pcaD
		self.warp_and_slice_model = warp_and_slice_model
		
	def forward(self, args):

		voxelized_mean_mesh, input_vol = args

		modes_output, global_shift, x_shifts, y_shifts, global_rotations = self.mp_model(input_vol)
		slice_shifts = torch.cat([y_shifts*0, y_shifts, x_shifts], dim=-1)

		predicted_cp = self.pcaD(modes_output)
		predicted_slices = self.warp_and_slice_model([predicted_cp, voxelized_mean_mesh, global_shift, slice_shifts, global_rotations])

		return predicted_slices

class LearnableInputPPModel(torch.nn.Module):
	'''
	Similar to FullPPModel but uses "lernable input parameters" rather than a parameter predicting neural network.
	Can be optimised on a single input to perform gradient descent to find the best parameters for that particular input.
	'''
	
	def __init__(self, learned_inputs, pcaD, warp_and_slice_model):

		super(LearnableInputPPModel, self).__init__()

		self.learned_inputs = learned_inputs
		self.pcaD = pcaD
		self.warp_and_slice_model = warp_and_slice_model
		self.num_modes = learned_inputs.num_modes

		self.slice_shifts_mask = 1

	def setSliceShiftMask(self, dicom_exam):

		mask = []
		for s in dicom_exam:
			if s.view == 'SAX':
				mask.extend( [1 for slc in range(s.slices)] )
			else:
				mask.extend( [0 for slc in range(s.slices)] )
		mask = np.array(mask)[None,:,None]

		self.slice_shifts_mask = torch.Tensor(mask).to(device)

		
	def forward(self, args):

		if len(args) == 2:
			voxelized_mean_mesh, ones_input = args
			temp = 0
		else:
			voxelized_mean_mesh, ones_input, temp = args

		# yyr 多了axis_parameters
		modes_output, volume_shift, x_shifts, y_shifts, global_rotations, axis_parameters = self.learned_inputs(ones_input)
		# print("看看axis_parameter：", axis_parameters)

		x_shifts = x_shifts * self.slice_shifts_mask
		y_shifts = y_shifts * self.slice_shifts_mask

		slice_shifts = torch.cat([x_shifts, y_shifts, y_shifts*0], dim=-1)



		predicted_cp = self.pcaD(modes_output, axis_parameters)
		predicted_slices = self.warp_and_slice_model([predicted_cp, voxelized_mean_mesh, volume_shift, slice_shifts, global_rotations, temp])
             
		# yyr 增加了axis_parameters
		return predicted_slices, modes_output, volume_shift, global_rotations, predicted_cp, slice_shifts, axis_parameters


# def makeFullPPModel(sz, num_modes, starting_cp, slice_positions, mode_bounds, mode_means, PHI3, 
# 	allow_global_shift_xy=True, 
# 	allow_global_shift_z=True, 
# 	allow_slice_shift=False
# 	):
# 	'''
# 	Builds and initializes all the models and then return them
# 	'''

# 	warp_and_slice_model = ParameterisedPlaneSliceSamplingSplineWarpSSM((sz,sz,sz), starting_cp[None], slice_positions, 
# 		allow_global_shift_xy=allow_global_shift_xy,
# 		allow_global_shift_z=allow_global_shift_z,
# 		allow_slice_shift=allow_slice_shift,
# 		)
# 	warp_and_slice_model.to(device)

# 	num_points = int(PHI3.shape[0]/3)
# 	num_slices = len(slice_positions)

# 	mp_model = SimpleCNN(num_modes=num_modes, num_slices=num_slices)
# 	with torch.no_grad(): #set the initial predictions to be 0:
# 		for m in [mp_model.modes_output_layer, mp_model.volume_shift_layer, mp_model.x_shift_layer, mp_model.y_shift_layer, mp_model.volume_rotations_layer]:
# 			m.weight.fill_(0.)
# 			m.bias.fill_(0.)
# 	mp_model.to(device)

# 	pcaD = PCADecoder(num_modes, num_points, mode_bounds[:num_modes], mode_means[:num_modes])
# 	with torch.no_grad(): #set the weights to the POD projection matrix (and 0 offset), and make not trainable (if required)
# 		pcaD.fc1.weight = nn.Parameter(torch.Tensor(PHI3))
# 		pcaD.fc1.bias.fill_(0.)
# 	pcaD.apply(make_not_trainable)
# 	pcaD.to(device)

# 	full_model = FullPPModel(mp_model, pcaD, warp_and_slice_model)
# 	full_model.to(device)

# 	learned_inputs = learnableInputs(num_modes=num_modes, num_slices=num_slices)
# 	with torch.no_grad(): #set the initial predictions to be 0:
# 		for m in [learned_inputs.modes_output_layer, learned_inputs.volume_shift_layer, learned_inputs.x_shift_layer, learned_inputs.y_shift_layer, learned_inputs.volume_rotations_layer]:
# 			m.weight.fill_(0.)
# 			m.bias.fill_(0.)

# 	li_model = LearnableInputPPModel(learned_inputs, pcaD, warp_and_slice_model)
# 	li_model.to(device)

# 	test_model = TestModel(pcaD, warp_and_slice_model)

# 	return mp_model, pcaD, warp_and_slice_model, full_model, learned_inputs, li_model, test_model


def makeFullPPModelFromDicom(sz, num_modes, starting_cp, dicom_exam, mode_bounds, mode_means, PHI3, offset, allow_global_shift_xy=True, allow_global_shift_z=True, allow_slice_shift=False, allow_rotations=False, series_to_exclude=[]):

	warp_and_slice_model = GivenPointSliceSamplingSplineWarpSSM((sz,sz,sz), starting_cp[None], dicom_exam, 
		allow_global_shift_xy=allow_global_shift_xy,
		allow_global_shift_z=allow_global_shift_z,
		allow_slice_shift=allow_slice_shift,
		allow_rotations=allow_rotations,
		series_to_exclude=series_to_exclude
	)
	warp_and_slice_model.to(device)

	num_points = int(PHI3.shape[0]/3)
	num_slices = warp_and_slice_model.num_slices

	mp_model = SimpleCNN(num_modes=num_modes, num_slices=num_slices)
	with torch.no_grad(): #set the initial predictions to be 0:
		for m in [mp_model.modes_output_layer, mp_model.volume_shift_layer, mp_model.x_shift_layer, mp_model.y_shift_layer, mp_model.volume_rotations_layer]:
			m.weight.fill_(0.)
			m.bias.fill_(0.)
	mp_model.to(device)

	pcaD = PCADecoder(num_modes, num_points, mode_bounds[:num_modes], mode_means[:num_modes], offset=offset)
	with torch.no_grad(): #set the weights to the POD projection matrix (and 0 offset), and make not trainable (if required)
		pcaD.fc1.weight = nn.Parameter(torch.Tensor(PHI3))
		pcaD.fc1.bias.fill_(0.)
	pcaD.apply(make_not_trainable)
	pcaD.to(device)

	full_model = FullPPModel(mp_model, pcaD, warp_and_slice_model)
	full_model.to(device)

	learned_inputs = learnableInputs(num_modes=num_modes, num_slices=num_slices)
	with torch.no_grad(): #set the initial predictions to be 0:
		for m in [learned_inputs.modes_output_layer, learned_inputs.volume_shift_layer, learned_inputs.x_shift_layer, learned_inputs.y_shift_layer, learned_inputs.volume_rotations_layer]:
			m.weight.fill_(0.)
			m.bias.fill_(0.)

	li_model = LearnableInputPPModel(learned_inputs, pcaD, warp_and_slice_model)
	li_model.to(device)

	test_model = TestModel(pcaD, warp_and_slice_model)

	return mp_model, pcaD, warp_and_slice_model, full_model, learned_inputs, li_model, test_model