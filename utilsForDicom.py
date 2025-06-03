import sys
# 2348
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() and "GTX 660" in torch.cuda.get_device_name(0):
	device = 'cpu'

import pydicom
import os
import imageio
import pickle
import ntpath
import meshio
import numpy as np
import pyvista as pv

from tqdm import tqdm
from skimage import measure
from copy import deepcopy
from matplotlib import pyplot as plt
from pydicom.filereader import read_dicomdir

from scipy.ndimage import zoom
from scipy.ndimage.measurements import center_of_mass as com
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.interpolation import shift
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from torchvision import utils


from skimage.morphology import skeletonize

import torch.optim as optim

from model_refactored import SliceExtractor, makeFullPPModelFromDicom, axis_parameters_Decoder
from helper_functions import loadSSMV2, voxelizeUniform, rotation_tensor, meshFittingLoss
from helpers2 import evalLearnedInputs, transformMeshAxes, getRotationMatrix, getSlices, slicewiseDice, mkdir
from textmaster import addLabels
from visualiseDICOM import planeToXYZ, planeToGrid
from DicomToSegDicom import produceSegAtRequiredRes
from shape_correction import simpleShapeCorrection
#from save_paraview_image import saveMeshImage

from meshes2strains import calculateStrains

#from vtisaver import saveVtiFromNumpyArray

'''
todo:

save calculated indecies as a numpy array / as a .txt file
save dicom from nnsegs and meshsegs
check slice shifts are actually slice parallel
correct slice shifts in data
correct mesh mirroring problem (i.e. solve the inverted mesh issue (axies swapped somewhere?))
profile to try and speed things up

process ACDC data
process MMWHS data
process KAGGLE data

possible extensions:
sort longditudinal strain estimate.. (maybe?)
speed up the slice intersection method..
load vti data
use valve direction (if available) for initial mesh orientation
gif of just nn segs over time
make sure smallest versions of each array are saved (i.e. choose a good dytpe)
combine SAX series
method to make uncertainty gif
valve detection / segmentation
predict papiliarry muscles
'''




def estimateThicknessAndRadius(myo):
	'''
	myo should be a binary (myocardium) mask with shape (H,W)
	this function assumes that the myocardium is appoximatley a circle and estimates:
		a) the average myocardium thickness
		b) the average myocardium radius

	#NOTE: this simple estimate may produce very bad estimates if the myo mask has a C shape
	#TODO: make function robust to this issue
	'''

	#if the myocardium mask is empty then return 0 for both thickness and radius:
	if np.sum(myo) == 0:
		return 0,0

	#get center of mass of myocardium (as integers)
	cx, cy = com(myo) 
	cx = int(np.round(cx))
	cy = int(np.round(cy))


	#thickness is the average number of mask pixels in lines extendiong from the center of mass:
	# thickness = (np.sum(myo[cx:cx+1,cy:]) + np.sum(myo[cx:cx+1,:cy]) + np.sum(myo[:cx,cy:cy+1]) + np.sum(myo[cx:,cy:cy+1]))/4
	thickness = (np.sum(myo[cx:cx+1,:]) + np.sum(myo[:,cy:cy+1]) + (np.sum(np.diag(myo)) + np.sum(np.diag(np.rot90(myo)))) * 2**0.5)/8

	# assert thickness == thickness2

	#measure horizontal diameter:
	p = np.where(myo[cx,:]==1)[0]
	if len(p) > 0:
		horizontal_diameter = p[-1]-p[0]
	else:
		horizontal_diameter = 0
	#measure vertical diameter:

	p = np.where(myo[:,cy]==1)[0]
	if len(p) > 0:
		vertical_diameter = p[-1]-p[0]
	else:
		vertical_diameter = 0

	p = np.where(np.diag(myo)==1)[0]
	if len(p) > 0:
		diag1_diam = (p[-1]-p[0])*2**0.5
	else:
		diag1_diam = 0

	p = np.where(np.diag(np.rot90(myo))==1)[0]
	if len(p) > 0:
		diag2_diam = (p[-1]-p[0])*2**0.5
	else:
		diag2_diam = 0

	# print(horizontal_diameter, vertical_diameter, diag1_diam, diag2_diam)

	#combine to estimate radiius:
	radius = (horizontal_diameter + vertical_diameter + diag1_diam + diag2_diam)/8

	return thickness, radius


def makeStrainPlot(fname, myo_thickness_over_time, myo_radius_over_time, myo_length_over_time, myo_volume_over_time, bp_volume_over_time):
	'''
	creates a plot of varius measured indices
	'''

	plt.figure(figsize=(7,8))

	plt.subplot(3,1,1)
	plt.plot(myo_volume_over_time, label='myo muscle volume')
	plt.plot(bp_volume_over_time, label='bloodpool volume')
	plt.tick_params(axis='x', which='both', labelbottom=False)
	plt.ylabel('$\propto mm^3$')
	plt.legend()
	plt.grid(True)

	plt.subplot(3,1,2)
	plt.plot(myo_thickness_over_time, label='myo thickness')
	plt.plot(myo_radius_over_time, label='LV radius')
	plt.tick_params(axis='x', which='both', labelbottom=False)
	plt.ylabel('mm')
	plt.legend()
	plt.grid(True)

	radial_strain = [i/myo_thickness_over_time[0]-1 for i in myo_thickness_over_time]
	circum_strain = [i/myo_radius_over_time[0]-1 for i in myo_radius_over_time]
	long_strain = [i/myo_length_over_time[0]-1 for i in myo_length_over_time]

	rs_max = '-' if len(radial_strain) == 0 else '%.2f' % (max(radial_strain),)
	cs_min = '-' if len(circum_strain) == 0 else '%.2f' % (min(circum_strain),)
	ls_min = '-' if len(long_strain) == 0 else '%.2f' % (min(long_strain),)

	plt.subplot(3,1,3)
	plt.plot(radial_strain, label='radial (max=%s)' % (rs_max,))
	plt.plot(circum_strain, label='circumferential (min=%s)' % (cs_min,))
	plt.plot(long_strain, label='longitudinal (min=%s)' % (ls_min,))
	plt.legend()
	plt.xlabel('time frame')
	plt.ylabel('strain')
	plt.grid(True)

	plt.savefig(fname, dpi=100)
	plt.close()


def path_leaf(path):
	'''
	from: https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
	returns the last "section" of a path, eg:
		path = 'this/folder/here/' returns 'here'
		path = 'this/folder/here.txt' returns 'here.txt'
		should work across different opperating systems
	'''
	head, tail = ntpath.split(path)
	return tail or ntpath.basename(head)


def to3Ch(img):
	'''
	the input img should be array of shape (H,W) or (H,W,3)
	returns an array of shape (H,W,3) with values in the range [0-1]

	in the special case where the input is an array with shape (H,W) containing 
	values in [0,1,2,3] then it is converted to a three channel image by doing:
	0 -> black, 1 -> red, 2 -> green, 3 -> blue

	this is useful for getting an array ready to be saved as an RGB image
	'''

	if len(img.shape) == 2:
		img_vals = np.unique(img)
		if set(img_vals) <= set([0,1,2,3]):
			cimg = np.zeros(img.shape+(3,))
			for i in [1,2,3]:
				cimg[img==i,i-1] = 1 
			return cimg
		else:
			img = img/img.max()
			return np.tile(img[...,None],(1,1,3))

	elif len(img.shape) == 3:
		img = img/img.max()
		return img

	else:
		print('error, input to to3Ch() should have shape (H,W) or (H,W,3), but recieved input with shape:', img.shape)


# def dataArrayFromNifti(full_path):

# 	import nibabel as nib
# 	img = nib.load(full_path)

# 	data = np.transpose(img.get_fdata(), (3,2,0,1))
# 	hdr = img.header

# 	pixel_spacing = [hdr.get('pixdim')[3], hdr.get('pixdim')[1], hdr.get('pixdim')[2]]
# 	image_ids = None
# 	dicom_details = None
# 	slice_locations = [0 for k in range(data.shape[1])]
# 	image_positions = [[0,0,k*pixel_spacing[0]] for k in range(data.shape[1])]
# 	trigger_times = None
# 	is3D = False
# 	multifile = None

# 	return data, pixel_spacing, image_ids, dicom_details, slice_locations, trigger_times, image_positions, is3D, multifile


# def dataArrayFromNumpyArray(full_path):

# 	data = np.load(full_path)[...,0]

# 	pixel_spacing = [10,1,1]
# 	image_ids = None
# 	dicom_details = None
# 	slice_locations = [0 for k in range(data.shape[1])]
# 	image_positions = [[0,0,k*pixel_spacing[0]] for k in range(data.shape[1])]
# 	trigger_times = None
# 	is3D = False
# 	multifile = None

# 	return data, pixel_spacing, image_ids, dicom_details, slice_locations, trigger_times, image_positions, is3D, multifile



class DicomSeries(object):
	'''A class for representing a single dicom series. Is used with DicomExam.'''

	def __init__(self, full_path, id_string=None):

		self.full_path = full_path
		self.series_folder_name = path_leaf(full_path).lower().split('.')[0]
		self.name = self.series_folder_name if id_string is None else id_string

		self.initialiseDetaultValues()

		if full_path.endswith('nii.gz'):
			print('loading series from NIfTI')
			# data, pixel_spacing, image_ids, dicom_details, slice_locations, trigger_times, image_positions, is3D, multifile = dataArrayFromNifti(full_path)
			# self.orientation = [0,1,0,1,0,0]

			self.dataArrayFromNifti(full_path)

		elif full_path.endswith('.par') or full_path.endswith('.rec'):
			print('loading series from par/rec files')
			self.dataArrayFromPARREC(full_path)

		elif full_path.endswith('.npy'):
			print('loading series from .npy file')
			# data, pixel_spacing, image_ids, dicom_details, slice_locations, trigger_times, image_positions, is3D, multifile = dataArrayFromNumpyArray(full_path)
			# self.orientation = [0,1,0,1,0,0]
			# self.name = 'sax_stack'

			self.dataArrayFromNumpyArray(full_path)

		else:

			# self.dataArrayFromDicom(full_path)

			data, pixel_spacing, image_ids, dicom_details, slice_locations, trigger_times, image_positions, is3D, multifile = dataArrayFromDicom(full_path)
			self.orientation = np.array(list(dicom_details['ImageOrientation']))

			self.data = data
			self.pixel_spacing = pixel_spacing
			self.image_ids = image_ids
			self.dicom_details = dicom_details
			self.slice_locations = slice_locations
			self.trigger_times = trigger_times
			self.image_positions = image_positions
			self.is3D = is3D
			self.multifile = multifile

		self.prepped_data = self.data
		# self.original_data_shape = data.shape
		
		# self.view = None
		# self.seg = None
		# self.prepped_seg = None
		# self.XYZs = []

		self.frames = self.data.shape[0]
		self.slices = self.data.shape[1]

		# self.VP_heuristic1 = None
		# self.VP_heuristic2 = None
		# self.slice_above_valveplane = None
		# self.uncertainty = None
		# self.mesh_seg = None
		# self.mesh_seg_uncertainty = None

		# self.tta_n = None
		# self.tta_preds_mean = None
		# self.tta_preds_std = None

		self.guessView()

	def __str__(self):

		details_str = "%s (%s), data shape = %s" % (self.series_folder_name, self.view, str(self.data.shape))
		if self.prepped_data.shape != self.data.shape:
			details_str += ' (resampled to %s)' % (self.prepped_data.shape,)
		if self.seg is not None:
			details_str += ' (has been segmented)'
		return details_str

	def addEmptySlice(self, position='top', spacing_scale=1):
		'''
		adds an empty slice to the series (to the top or bottom). Can be useful for telling the mesh where *not* to g

		spacing_scale is the multiple of the default spacing between slices to use, defaults to 1

		spacing_scale = 0 puts the new slice exactly ontop of the current last slice
		spacing_scale = 1 puts the new slice 1 slice_gap above the current last slice (ie. same spacing as between the other slices)
		'''

		if self.data.shape[1] <= 1:
			print('can currently only add slices to series with atleast 2 slices, skipping')
			return

		#what about self.image_ids?

		if position == 'top':
			self.data = np.concatenate([np.zeros(self.data.shape)[:,:1], self.data], axis=1)
			self.slice_locations = [0] + self.slice_locations
			# self.image_positions.insert(0, list(2*np.array(self.image_positions[0]) - np.array(self.image_positions[1])) )
			offset = np.array(self.image_positions[0]) - np.array(self.image_positions[1])
			self.image_positions.insert(0, list(np.array(self.image_positions[0])+offset*spacing_scale) )

		elif position == 'bottom':
			self.data = np.concatenate([self.data, np.zeros(self.data.shape)[:,:1]], axis=1)	
			self.slice_locations = self.slice_locations + [0]

			offset = np.array(self.image_positions[-1]) - np.array(self.image_positions[-2])
			self.image_positions.append( list(np.array(self.image_positions[-1]) + offset*spacing_scale) )
			# self.image_positions.append( list(2*np.array(self.image_positions[-1]) - np.array(self.image_positions[-2])) )

		print(self.image_positions)
			
		self.prepped_data = self.data
		self.slices += 1
			
	def saveSidewaysView(self, t=0, shifts=None, thickness=12):
		'''
		used for visualising slices down through the short-axis stack, and can be usefult for visualising slice shifts
		'''

		assert shifts is None or shifts.shape == (self.prepped_data.shape[1],2)

		def getPairedView(arr, shifts):

			arr = arr[t] + 0
			num_slices, sz1, sz2 = arr.shape
			view_1_image_stack = []
			view_2_image_stack = []
			
			cx, cy = sz1//2, sz2//2
			if shifts is not None: #apply the slice shifts:
				max_shift = np.ceil(np.max(np.abs(shifts))).astype('int')
				arr0s = np.pad(arr, ((0,0),(max_shift, max_shift),(max_shift,max_shift)), mode='constant', constant_values=0)
				arr = np.pad(arr, ((0,0),(max_shift, max_shift),(max_shift,max_shift)), mode='constant', constant_values=1)
				for k in range(num_slices):
					# arr0s[k] = shift(arr0s[k], (shifts[k,1],shifts[k,0]), cval=1, order=0)
					# arr[k] = shift(arr[k], (shifts[k,1],shifts[k,0]), cval=1, order=0)
					arr0s[k] = shift(arr0s[k], (-shifts[k,0],-shifts[k,1]), cval=1, order=0)
					arr[k] = shift(arr[k], (-shifts[k,0],-shifts[k,1]), cval=1, order=0)

				_, cx, cy = com(arr0s)
				cx = int(np.round(cx))
				cy = int(np.round(cy))

			for k in range(num_slices):

				v1_slice = np.tile( arr[k:k+1,cx,:], (thickness, 1) )
				view_1_image_stack.append( v1_slice )

				v2_slice = np.tile( arr[k:k+1,:,cy], (thickness, 1) )
				view_2_image_stack.append( v2_slice )

			view_1 = np.concatenate(view_1_image_stack)
			view_2 = np.concatenate(view_2_image_stack)
			img = np.concatenate([view_1, np.ones((thickness*num_slices, 10)), view_2], axis=1)
			img = (img*255).astype('uint8')

			return img

		data_img = getPairedView(self.prepped_data, shifts)
		mask_img = getPairedView((self.prepped_seg==2)*1, shifts)
		img = np.concatenate([data_img, data_img[:10]*0+255, mask_img])

		return img

		#, self.prepped_seg.shape )


	def initialiseDetaultValues(self,):
		'''set default values for various series properties'''

		self.image_ids = None
		self.dicom_details = None
		self.trigger_times = None
		self.is3D = False
		self.multifile = None
		self.view = None
		self.seg = None

		self.prepped_seg = None
		self.XYZs = []

		self.VP_heuristic1 = None
		self.VP_heuristic2 = None
		self.slice_above_valveplane = None
		self.uncertainty = None
		self.mesh_seg = None
		self.mesh_seg_uncertainty = None

		self.tta_n = None
		self.tta_preds_mean = None
		self.tta_preds_std = None


	def dataArrayFromNifti(self, full_path):

		import nibabel as nib
		img = nib.load(full_path)
		hdr = img.header

		# yyr
		print("看看数据的shape",img.get_fdata().shape)
        
		# ============================
		'''
		个别数据集shape问题
		'''
		# img_data = np.reshape(img.get_fdata(), (250, 146, 8, 1)) 
		# self.data = np.transpose(img_data, (3,2,0,1))
		# ============================
        
		'''
		ACDC数据集的shape
		'''
		self.data = np.transpose(img.get_fdata(), (3,2,0,1))


		self.pixel_spacing = [hdr.get('pixdim')[3], hdr.get('pixdim')[1], hdr.get('pixdim')[2]]

		# self.image_ids = None
		# self.dicom_details = None
		self.slice_locations = [0 for k in range(self.data.shape[1])]
		# self.image_positions = [[0,0,-k*self.pixel_spacing[0]] for k in range(self.data.shape[1])]
		self.image_positions = [[0,0,k*self.pixel_spacing[0]] for k in range(self.data.shape[1])]
		# self.trigger_times = None
		# self.is3D = False
		# self.multifile = None
		self.orientation = [0,1,0,1,0,0]
		# self.orientation = [1,0,0,1,0,0]
		self.name += 'sax_stack'

	def dataArrayFromPARREC(self, full_path):
		'''loads a series from a nifti file'''

		from parrec.recread import Recread
		from parrec.parread import Parread
		import numpy as np
		from scipy.spatial.transform import Rotation as R
		from nibabel.affines import apply_affine

		recFile = Recread(full_path)
		parFile = Parread(full_path)

		with open(parFile.parfile) as file:
			upper_pars = parFile._read_upper_part(file)
			lower_pars = parFile._read_lower_part(file)

		loadedImg              = recFile.read()
		imgInfo                = recFile.parameter['ImageInformation']
		cardiacPhases          = upper_pars['MaxNumberOfCardiacPhases']
		offset_center          = np.array(upper_pars['OffCentreMidslice'],dtype='float32')
		n_slices               = int(upper_pars['MaxNumberOfSlicesLocations'])
		preparationDirection   = upper_pars['PreparationDirection']
		volumeDimensions       = [imgInfo[0]['ReconResolution'][0],imgInfo[0]['ReconResolution'][1],n_slices]
		pixel_spacing_original = [imgInfo[0]['PixelSpacing'][0],imgInfo[0]['PixelSpacing'][1],imgInfo[0]['SliceThickness']]
		self.pixel_spacing     = [imgInfo[0]['SliceThickness'],imgInfo[0]['PixelSpacing'][0],imgInfo[0]['PixelSpacing'][1]]
		data = loadedImg.reshape(loadedImg.shape[0],loadedImg.shape[1],n_slices,cardiacPhases)
		self.data = np.array(np.transpose(data, (3,2,0,1)),dtype='float64')

		ap_rot, fh_rot, rl_rot = np.array(upper_pars['AngulationMidslice'],dtype='float32')

		# Permutation matrices to adapt to the Philips coordinate system
		# PSL to RAS affine   (from nibabel package)
		PSL_TO_RAS = np.array([	[0, 0, -1, 0],  # L -> R
								[-1, 0, 0, 0],  # P -> A
								[0, 1, 0, 0],   # S -> S
								[0, 0, 0, 1]])

		ACQ_TO_PSL = dict(	transverse=np.array([[0, 1, 0, 0], # P
												[0, 0, 1, 0],  # S
												[1, 0, 0, 0],  # L
												[0, 0, 0, 1]]),
							sagittal=np.diag([1, -1, -1, 1]),
							coronal=np.array([	[0, 0, 1, 0],  # P
												[0, -1, 0, 0], # S
												[1, 0, 0, 0],  # L
												[0, 0, 0, 1]])
						)

		Mx,My,Mz  = np.zeros((4,4)),np.zeros((4,4)),np.zeros((4,4))
		Mx[:3,:3] = R.from_euler('x',ap_rot,degrees=True).as_matrix()
		Mx[3,3]   = 1.0
		My[:3,:3] = R.from_euler('y',fh_rot,degrees=True).as_matrix()
		My[3,3]   = 1.0
		Mz[:3,:3] = R.from_euler('z',rl_rot,degrees=True).as_matrix()
		Mz[3,3]   = 1.0
		rot = np.dot(Mz,np.dot(Mx, My))

		to_center        = np.zeros((4,4))
		to_center[:3,:3] = np.eye(3)
		to_center[:3,3]  = -(np.array(volumeDimensions)-1) / 2.
		zoomer = np.diag(np.array(pixel_spacing_original+[1]))

		if preparationDirection == 'RL':
			patientDirection = 'coronal' 
		elif preparationDirection == 'FH':
			patientDirection = 'transverse'
		elif preparationDirection == 'AP':
			patientDirection = 'sagittal'

		permute_to_psl    = ACQ_TO_PSL.get(patientDirection)

		psl_aff = np.dot(rot, np.dot(permute_to_psl, np.dot(zoomer, to_center)))
		psl_aff[:3,3] += offset_center
		self.affine = np.dot(PSL_TO_RAS, psl_aff)
		self.image_positions = [apply_affine(self.affine, [0,0,k]) for k in range(self.data.shape[1])]

		ax1 = apply_affine(self.affine, [0,1,0]) - apply_affine(self.affine, [0,0,0])
		ax1 = ax1 / np.sum(ax1**2)**0.5
		ax2 = apply_affine(self.affine, [1,0,0]) - apply_affine(self.affine, [0,0,0])
		ax2 = ax2 / np.sum(ax2**2)**0.5
		ori = np.concatenate([ax1, ax2])
		self.orientation = ori

		self.slice_locations = [[0,0,k*self.pixel_spacing[0]] for k in range(self.data.shape[1])]#[0 for k in range(self.data.shape[1])]
		# self.image_positions = [[0,0,-k*self.pixel_spacing[0]] for k in range(self.data.shape[1])]
		# I need to compute the coordinates of the centers of the slices for each slice of the image
		self.image_positions = []
		for sliceN in range(self.data.shape[1]):
			dmmy = apply_affine(self.affine, [0,0,sliceN])
			self.image_positions.append([dmmy[0],dmmy[1],dmmy[2]])

	def dataArrayFromNumpyArray(self, full_path):

		self.data = np.load(full_path)[...,0]
		self.pixel_spacing = [10,1,1]
		# if 'mmwhs' in full_path:
		# 	print('is mmwhs')
		# 	self.pixel_spacing = [-10,1,1]
			# self.data = self.data[:,::-1]
		# self.image_ids = None
		# self.dicom_details = None
		self.slice_locations = [0 for k in range(self.data.shape[1])]
		self.image_positions = [[0,0,k*self.pixel_spacing[0]] for k in range(self.data.shape[1])]
		# self.trigger_times = None
		# self.is3D = False
		# self.multifile = None
		self.orientation = [0,1,0,1,0,0]
		self.name += 'sax_stack'


	def getXYZ(self, t=0, slice_index=0):
		'''gets the pixel co-ordinates in "real space", returns a (x,y,z) value for each pixel'''

		return planeToXYZ(self.data.shape[2:], self.image_positions[slice_index], self.orientation, self.slice_locations[slice_index], self.pixel_spacing)

	def guessView(self):
		'''sets the series' view property to one of: [SAX, 2CH, 3CH, 4CH, unknown]'''

		#try to guess the view from series name:
		if 'sax' in self.series_folder_name:
			self.view = 'SAX'
		elif '2ch' in self.series_folder_name:
			self.view = '2CH'
		elif '3ch' in self.series_folder_name:
			self.view = '3CH'
		elif '4ch' in self.series_folder_name:
			self.view = '4CH'
		elif 'sa' in self.series_folder_name:
			self.view = 'SAX'

		#if the data is from npy of nifti then assume SAX:
		elif 'nii.gz' in self.series_folder_name:
			self.view = 'SAX'
		elif '.npy' in self.series_folder_name:
			self.view = 'SAX'

		#otherwise, look at the number of slices:
		elif self.data.shape[1] > 3:
			print('guessing %s is SAX because it has more than 3 sices' % (self.series_folder_name,))
			self.view = 'SAX'

		#otherwise we don't know:
		else:
			self.view = 'unknown'

		return self.view

def loadDicomExam(base_dir, output_folder='autodl-tmp/outputs', id_string=None):
	'''loads a saved DicomExam (pass the same params as when creating the dicom exam)'''
	id_string = path_leaf(base_dir).split('.')[0] if id_string is None else id_string
	fname = os.path.join(output_folder, id_string+'_outputs', 'DicomExam.pickle')
	file_to_read = open(fname, "rb")
	de = pickle.load(file_to_read)
	file_to_read.close()
	return de

class DicomExam(object):
	'''
	Dicom exam object which contains a number of DicomSeries objects
	'''

	def __init__(self, base_dir, output_folder='autodl-tmp/outputs', id_string=None):

		print('\ncreating DicomExam from: %s' % (base_dir,))

		self.time_frames = None
		self.id_string = path_leaf(base_dir).split('.')[0] if id_string is None else id_string
		self.base_dir = base_dir
		self.series = []
		self.series_names = []
		self.output_folder = output_folder
		self.sax_slice_interesections = None
		self.series_to_exclude = []
		self.fitted_meshes = {}
		self.setFolderNames()

		if base_dir.endswith('nii.gz'):
			print('loading Exam from NIfTI (one series, assume it is a SAX stack)')
			self.series.append( DicomSeries(base_dir, self.id_string) )
			self.series_names.append( 'sax_stack' )
			self.num_series = 1

		elif base_dir.endswith('.npy'):
			print('loading Exam from numpy array (one series, assume it is a SAX stack)')
			self.series.append( DicomSeries(base_dir, self.id_string) )
			self.series_names.append( 'sax_stack' )
			self.num_series = 1

		else:
			ordered_series = sorted_nicely(os.listdir(base_dir))
			ordered_series = [x for x in ordered_series if x[0] != '.']
			ordered_series = [x for x in ordered_series if x[0] != '.']
			ordered_series = [x for x in ordered_series if x[:2] != 'XX']
			ordered_series = [x for x in ordered_series if x[-3:] != 'rec']
			ordered_series = [x for x in ordered_series if x != 'Output']

			for series_dir in ordered_series:
				full_path = os.path.join(base_dir, series_dir)
				print('loading series from /%s' %(series_dir,) )
				if not 'rv' in series_dir.lower():

					ds = DicomSeries(full_path)

					# yyr
					ds.view = 'SAX'

					if np.prod(ds.data.shape) > 1:
						self.series.append( ds )
						self.series_names.append(series_dir)
			self.num_series = len(self.series)

	def addSeries(self, dicom_series, series_name):

		print('adding series...')

		# print(len(self), self[0].data.shape)

		self.series.append( dicom_series )
		self.series_names.append( series_name )
		self.num_series += 1

		# print(len(self), self[0].data.shape, self[1].data.shape)

		
	def resetMeshFitting(self,):
		#get rid of previous meshes that were fitted
		self.fitted_meshes = {}

	def summary(self,):
		'''print a short summary of the dicom exam'''
		print('summary:')
		print('\tsource directory: %s' % (self.base_dir,))
		print('\tnumber of series: %d' % (self.num_series,))
		print('\tseries details:')
		for s in self.series:
			print('\t%s' % (str(s),))

	def getSegmentationArray(self,series_name):
		'''returns the (original image sized) segmentations masks for a given series'''
		for s in self.series:
			if s.name == series_name:
				return s.seg.astype('uint8')
		print('no series found with name: %s, available series are:' % (series_name,))
		for s in self.series:
			print(s.name)

	def setFolderNames(self):
		'''create names for various folders which will be used when saving outputs'''

		prefix = self.output_folder
		self.folder = {}
		self.folder['base'] = os.path.join(prefix, self.id_string+'_outputs')
		self.folder['debug'] = os.path.join(prefix, self.id_string+'_outputs', 'debug')
		self.folder['plots'] = os.path.join(prefix, self.id_string+'_outputs', 'plots')
		self.folder['meshes'] = os.path.join(prefix, self.id_string+'_outputs', 'meshes')
		self.folder['gifs'] = os.path.join(prefix, self.id_string+'_outputs', 'gifs')
		self.folder['strain_meshes'] = os.path.join(prefix, self.id_string+'_outputs', 'strain_meshes')
		self.folder['image_predictions'] = os.path.join(prefix, self.id_string+'_outputs', 'images', 'predictions')
		self.folder['initial_segs'] = os.path.join(prefix, self.id_string+'_outputs', 'images', 'initial_nn_seg_images')
		self.folder['mesh_segs'] = os.path.join(prefix, self.id_string+'_outputs', 'images', 'mesh_seg_images')
		self.folder['3d_visualisations'] = os.path.join(prefix, self.id_string+'_outputs', '3d_visualisations')
		self.folder['image_space_meshes'] = os.path.join(prefix, self.id_string+'_outputs', 'image_space_meshes')
		self.folder['initial_seg_dicom'] = os.path.join(prefix, self.id_string+'_outputs', 'segmentation_dicom', 'initial_nn_seg')
		self.folder['initial_seg_uncertainty'] = os.path.join(prefix, self.id_string+'_outputs', 'images', 'initial_nn_seg_uncertainty')
		self.folder['mesh_seg_uncertainty'] = os.path.join(prefix, self.id_string+'_outputs', 'images', 'mesh_seg_uncertainty')
		self.folder['cache'] = os.path.join(prefix, self.id_string+'_outputs', 'cache')
		self.folder['data'] = os.path.join(prefix, self.id_string+'_outputs', 'data')

	def changeOutputFolder(self, new_folder):

		self.output_folder = new_folder
		self.setFolderNames()


	def standardiseTimeframes(self, resample_to='fewest'):
		'''
		make sure all series have the same number of time frames, and resample them if they don't.
		'''

		print('standardising number of time frames across series by resampling..')

		if len(self.series) == 1:
			print('only one series, so no resampling required')
			self.time_frames = self.series[0].frames
			return

		time_frames_seen = []
		for s in self.series:
			time_frames_seen.append(s.frames)
		time_frames_seen = np.unique(time_frames_seen)

		if len(time_frames_seen) == 1:
			self.time_frames = time_frames_seen[0]
			print('all series already have %d time frame(s), so no resampling required' % (time_frames_seen[0]))
			return

		if 1 in time_frames_seen:
			print('some series only have a single time frame, these will be ignored')
			time_frames_seen = [t for t in time_frames_seen if t != 1]

		if resample_to == 'fewest':
			#downsample to smallest time resolution:
			target_slices = np.min(time_frames_seen)
		else: 
			#otherwise upsample to highest temporal resolution:
			target_slices = np.max(time_frames_seen)

		print('resampling all series to %d time frames' % (target_slices,))
		for s in self.series:

			#skip series with only 1 time frame (upsampling them doesn't really make any sense..)
			if s.frames == 1:
				continue

			if s.frames != target_slices:

				s.prepped_data = zoom(s.data, (target_slices/s.data.shape[0],1,1,1), order=1)

				# is_sax = (s.view in ['SAX', 'unknown'])
				# dat, seg, c1, c2 = produceSegAtRequiredRes(resampled_data, s.pixel_spacing, is_sax, use_tta)
				# sz = 128
				# c1 = np.clip(c1-sz//2, 0, seg.shape[2]-sz) 
				# c2 = np.clip(c2-sz//2, 0, seg.shape[3]-sz) 
				# s.prepped_seg_resampled = np.transpose(seg[:,:,c1:c1+sz,c2:c2+sz], (0,1,3,2))
				# s.prepped_data_resampled = np.transpose(dat[:,:,c1:c1+sz,c2:c2+sz], (0,1,3,2))
				# s.resampled = True

		self.time_frames = target_slices

	def __str__(self,):
		s = ''
		for i in range(len(self.series_names)):
			s += '%s %s (%s)\n' % (self.series_names[i], str(self.series[i].data.shape), str(self.series[i].prepped_data.shape))
		return s

	def __getitem__(self, i):
		return self.series[i]

	def __len__(self):
		return self.num_series

	def combineSeries(self,):
		#TODO: finish this method
		#check if any seperate series are actually the same series (e.g. different slices), and if so combine them
		groups = []
		for i in range(self.num_series):
			i_grouped = False
			for j, g in enumerate(groups):
				if np.array_equal(self.series[i].orientation, self.series[g[0]].orientation):
					groups[j].append(i)
					i_grouped = True
					break
			if not i_grouped:
				groups.append([i])

		for g in groups:
			if len(g) > 1:
				print('should combine', g)

	def calculateSegmentationUncertainty(self, masks_to_use='network'):


		if masks_to_use == 'network':
			if self[0].tta_preds_mean is None: 
				print('you need to run segmentation using TTA=True before you can use calculateNetworkSegmentationUncertainty(masks_to_use="network")')
				print('skipping')
				return
			folder = self.folder['initial_seg_uncertainty']
		elif masks_to_use == 'mesh':
			if self[0].mesh_seg is None:
				print('you need to run fitMesh() before you can use calculateNetworkSegmentationUncertainty(masks_to_use="mesh")')
				print('skipping')
				return
			folder = self.folder['mesh_seg_uncertainty']

		mkdir(folder)

		for s in self:

			if masks_to_use == 'network':
				ap_mean = s.tta_preds_mean #np.mean(s.all_preds, axis=0)
				ap_std = s.tta_preds_std #np.std(s.all_preds, axis=0)

				#produce values in [1/n,1] for each image, where n is the number of seperate segmentation predicted for each image whenusing TTA:
				n = s.tta_n#all_preds.shape[0]
				uncert = 1-np.sum(ap_mean,axis=(2,3))/(np.sum(ap_mean>0,axis=(2,3))+0.000001)
				uncert = (uncert - 1/n) * n/(n-1)  #map from [1/n,1] to [0,1]

				s.uncertainty = uncert #save the uncertainty array for the series 2D array of shape (time-frames, slices)

				for t in range(self.time_frames):
					for sl in range(s.slices):
						if np.sum(s.data[t,sl]) == 0:
							s.uncertainty[t,sl] = 0

			elif masks_to_use == 'mesh':
				#todo: fix this bit
				ap_mean = s.mesh_seg
				ap_std = s.mesh_seg_std
				uncert = 1-np.sum(ap_mean,axis=(2,3))/(np.sum(ap_mean>0,axis=(2,3))+0.000001)
				#note, uncert not normalised here as n may be different for different time frames (but probably isn't?)..

			std_img = np.concatenate(np.concatenate(ap_std,axis=2))
			std_img = ((std_img/0.5)*255).astype('uint8')
			fname = os.path.join(folder, 'std_img_%s.png' % (s.name,))
			imageio.imwrite(fname, std_img)

			mean_img = np.concatenate(np.concatenate(ap_mean,axis=2))
			mean_img = (mean_img*255).astype('uint8')
			fname = os.path.join(folder, 'mean_img_%s.png' % (s.name,))
			imageio.imwrite(fname, mean_img)

			fname = os.path.join(folder, 'per_image_uncertainty_%s.png' % (s.name,))
			plt.imshow(np.mean(uncert, axis=-1))
			plt.colorbar()
			plt.savefig(fname)
			plt.close()

	def makeSliceShiftCorrectedData(self,t=0):

		import pyvista as pv

		for i,s in enumerate(self):
			s.corrected_data = s.data[t] + 0
			shifts = np.concatenate([self.fitted_meshes[t]['x_shifts'][0][:,None], self.fitted_meshes[t]['y_shifts'][0][:,None]], axis=1) * 64 / s.pixel_spacing[1:]
			for k in range(s.corrected_data.shape[0]):
				s.corrected_data[k] = shift(s.corrected_data[k], (shifts[k,1],-shifts[k,0]), cval=0, order=0)
				grid = planeToGrid(s.corrected_data[k].shape, s.image_positions[k], s.orientation, s.pixel_spacing[1:])
				grid.cell_arrays["values"] = s.corrected_data[k].T.flatten(order="F")
				grid.save('series_%d_slice_%d.vts' % (i,k))


	def visualiseLearnedSliceOffsets(self,t=0,suffix=''):

		if self.num_series > 1:
			print("visualiseLearnedSliceOffsets() is currently only implemented for exams with one series, skipping")

		shifts = np.concatenate([-self.fitted_meshes[t]['x_shifts'][0][:,None], -self.fitted_meshes[t]['y_shifts'][0][:,None]], axis=1)
		shifts *= 64 

		arrow_image = imageio.imread('images_for_figures/arrow.png')[...,0]

		for s in self:
			if s.view == 'SAX':
				initial_alignment = s.saveSidewaysView(t, shifts=None)
				corrected_alignment = s.saveSidewaysView(t, shifts=shifts)

		diff = initial_alignment.shape[0] - arrow_image.shape[0]

		arrow_image = np.pad(arrow_image, ((diff//2, diff-diff//2),(10,10)), mode='constant', constant_values=255)

		print(initial_alignment.shape, arrow_image.shape, corrected_alignment.shape)

		img = np.concatenate([initial_alignment,arrow_image,corrected_alignment], axis=1)

		imageio.imwrite('shift_corrections/slice_shift_correction%s.png' % (suffix,), img)


	def setSegmentations(self, masks, sz=128):
		'''
		method to set the  segmentation masks (eg. if they are already known) 
		can be used instead of .segment() when masks are available

		masks should be a list of arrays, one for each series, with the array size matching 
		'''

		for i, s in enumerate(self):

			m = masks[i]

			if s.prepped_data.shape != m.shape: #check the masks are the correct shape
				print('mask shape',m.shape, 'does not match prepped_data shape', s.prepped_data.shape, 'for series %d' % (i,))
				print('aborting')
				sys.exit()

			data = zoom(s.prepped_data,(1, 1, s.pixel_spacing[1], s.pixel_spacing[2]), order=1)

			# normalize intensities:
			data = data - data.min()
			data = np.clip(data, 0, np.percentile(data, 99.5))
			data = data / data.max()

			m = zoom(m,(1, 1, s.pixel_spacing[1], s.pixel_spacing[2]), order=0)

			c1, c2 = com(np.mean(m.reshape((-1,m.shape[2],m.shape[3])), axis=0))
			c1, c2 = int(np.round(c1)), int(np.round(c2))

			c1 = np.clip(c1-sz//2, 0, m.shape[3]-sz) 
			c2 = np.clip(c2-sz//2, 0, m.shape[2]-sz)

			# c1 = np.clip(c1-sz//2, 0, m.shape[2]-sz) 
			# c2 = np.clip(c2-sz//2, 0, m.shape[3]-sz)

			# print(m.shape, c1, c2, sz)

			s.prepped_seg = np.transpose(m[:,:,c1:c1+sz,c2:c2+sz], (0,1,3,2)).astype('int')
			s.prepped_data = np.transpose(data[:,:,c1:c1+sz,c2:c2+sz], (0,1,3,2))

			s.c1, s.c2 = c1,c2

			for slice_index in range(s.slices):

				X,Y,Z = planeToXYZ(m.shape[2:], s.image_positions[slice_index], s.orientation, s.slice_locations[slice_index], [1,1])
				# print(X.shape,Y.shape,Z.shape)
				X, Y, Z = X[c2:c2+sz,c1:c1+sz], Y[c2:c2+sz,c1:c1+sz], Z[c2:c2+sz,c1:c1+sz]
				# print(X.shape,Y.shape,Z.shape)
				s.XYZs.append( np.concatenate([X.reshape((sz**2,1)), Y.reshape((sz**2,1)), Z.reshape((sz**2,1))], axis=1) )
				# s.centers.append( np.mean(s.XYZs[-1], axis=0) )

		self.estimateLandmarks()


	def segment(self, use_tta=False, sz=128):
		#segments each series

		for s in self:
			print('segmenting %s (%d images)%s' % (s.name, np.prod(s.prepped_data.shape[:2]), ' using TTA' if use_tta else ''))

			is_sax = (s.view in ['SAX', 'unknown'])

			dat, seg, c1, c2, all_preds = produceSegAtRequiredRes(s.prepped_data, s.pixel_spacing, is_sax, use_tta=use_tta)
			s.seg = zoom(seg,(1, 1, 1/s.pixel_spacing[1], 1/s.pixel_spacing[2]), order=0)


			if all_preds is not None:
				#save the mean and stand deviation of the TTA predictions:
				s.tta_preds_mean = np.mean(all_preds, axis=0)
				s.tta_preds_std = np.std(all_preds, axis=0)
				s.tta_n = all_preds.shape[0] #and the number of TTA predictions that were made

			print(c1,c2, seg.shape)
			# c1 = np.clip(c1-sz//2, 0, seg.shape[3]-sz) 
			# c2 = np.clip(c2-sz//2, 0, seg.shape[2]-sz)
			#todo- replace this quick fix:
			c1 = np.clip(c1-sz//2, 0, min(seg.shape[3]-sz,seg.shape[2]-sz))
			c2 = np.clip(c2-sz//2, 0, min(seg.shape[3]-sz,seg.shape[2]-sz))
			print(c1,c2, seg.shape)

			s.c1,s.c2 = c1,c2 #save the center point

			for slice_index in range(s.slices):

				X,Y,Z = planeToXYZ(seg.shape[2:], s.image_positions[slice_index], s.orientation, s.slice_locations[slice_index], [1,1])
				# print(X.shape,Y.shape,Z.shape)
				X, Y, Z = X[c2:c2+sz,c1:c1+sz], Y[c2:c2+sz,c1:c1+sz], Z[c2:c2+sz,c1:c1+sz]
				# print(X.shape,Y.shape,Z.shape)
				s.XYZs.append( np.concatenate([X.reshape((sz**2,1)), Y.reshape((sz**2,1)), Z.reshape((sz**2,1))], axis=1) )
				# s.centers.append( np.mean(s.XYZs[-1], axis=0) )



			s.prepped_seg = np.transpose(seg[:,:,c1:c1+sz,c2:c2+sz], (0,1,3,2))
			s.prepped_data = np.transpose(dat[:,:,c1:c1+sz,c2:c2+sz], (0,1,3,2))

			# s.prepped_seg = np.transpose(seg[:,:,c2:c2+sz,c1:c1+sz], (0,1,3,2))
			# s.prepped_data = np.transpose(dat[:,:,c2:c2+sz,c1:c1+sz], (0,1,3,2))

			if is_sax:
				s.prepped_seg = simpleShapeCorrection(s.prepped_seg)

		self.estimateLandmarks()

	def saveSegDicom(self, segmentations_from):

		if segmentations_from not in ['network', 'mesh']:
			print('segmentations_from must be either "network" or "mesh" (currently recieved %s)' % (segmentations_from,) )
			return
		
		if self[0].seg is None:
			print("no segmentations to save (run .segment() first to produce masks)")
			return

		mkdir(self.folder['initial_seg_dicom'])

		for i in range(self.num_series):

			s = self.series[i]
			s_name = self.series_names[i]

			if s.seg.shape != s.data.shape:
				print('data was resampled before segmenting, and so segmentation masks are a differtent shape/resolution from the orginal data:')
				print('original data shape = ', s.data.shape)
				print('segmentation masks shape = ', s.seg.shape)
				print('currently cant save a dicom if this is true')
				print('if you want a segmentation dicom, load the data, call .segment() then .saveSegDicom() without calling .standardiseTimeframes()')
				return

			lstFilesDCM = getSortedFilenames(s.full_path)

			if s.multifile:

				series_output_folder = os.path.join(self.folder['initial_seg_dicom'], s_name)
				mkdir(series_output_folder)

				print(np.unique(np.round(s.seg)))
				for j in range(s.image_ids.shape[0]):
					for k in range(s.image_ids.shape[1]):

						fname = lstFilesDCM[int(s.image_ids[j,k])]
						f = pydicom.read_file(fname)
						file = path_leaf(fname)# fname.split('/')[-1]
						# print('saving to %s' % (series_output_folder+'/'+file))
						f.PixelData = np.round(s.seg[j,k]).astype(f.pixel_array.dtype).tostring()
						f.save_as( os.path.join(series_output_folder,file) )

			else:

				fname = lstFilesDCM[0]
				# print('loading', fname)

				f = pydicom.read_file(fname)
				file = path_leaf(fname)
				# print('saving to %s' % (output_folder+'/'+file))
				f.PixelData = np.round(s.seg[i,j]).astype(f.pixel_array.dtype).tostring()
				f.save_as(self.output_folder + '/' + file)


	def showImagesVista(self):

		time_frame = 0

		plotter = pv.Plotter()
		for s in self: #for easch series:
			intensities = s.data

			if 'sax' in s.name and s.name != 'sax_30':
				continue
			
			for slice_index in [s.slices//2]: #for each slice:
				grid = planeToGrid(intensities.shape[2:], s.image_positions[slice_index], s.orientation, s.pixel_spacing)
				plotter.add_mesh(grid, scalars=intensities[time_frame,slice_index].T.flatten(order="F"), cmap='gray', opacity=1)

		plotter.show(auto_close=False)
		cpos = plotter.camera_position
		plotter.close()

		plotter = pv.Plotter()
		plotter.camera_position = cpos
		for s in self: #for easch series:
			intensities = (s.seg==2)*2

			if 'sax' in s.name and s.name != 'sax_30':
				continue
			
			for slice_index in [s.slices//2]: #for each slice:
				grid = planeToGrid(intensities.shape[2:], s.image_positions[slice_index], s.orientation, s.pixel_spacing)
				plotter.add_mesh(grid, scalars=intensities[time_frame,slice_index].T.flatten(order="F"), cmap='gray', opacity=0.5)

		plotter.show(auto_close=False)
		cpos = plotter.camera_position
		plotter.close()


	def addValvePlaneLabels(self, s, im, imsz):
		'''
		adds markings to an image produced by saveImages to indicate the slices believed to be above the valve plane
		'''

		for i in range(self.time_frames):
			for j in range(s.slices):

				if s.slice_above_valveplane is not None:
					if s.slice_above_valveplane[i,j]:
						im[int(imsz*(j+0.8)):imsz*(j+1),imsz*i:imsz*(i+1)] = im[int(imsz*(j+0.8)):imsz*(j+1),imsz*i:imsz*(i+1)]/2 + [0.5,0.5,0]

				if s.VP_heuristic1 is not None:
					if s.VP_heuristic1[i,j] == 2:
						im[imsz*j+3:imsz*j+25,imsz*i+3:imsz*i+25] = [1,1,0]

				if s.VP_heuristic2 is not None:
					if not s.VP_heuristic2[i,j]:
						im[imsz*j+3+28:imsz*j+25+28,imsz*i+3:imsz*i+25] = [1,1,0]

				if self.sax_slice_interesections is not None:

					if self.sax_slice_interesections[j,i] < 0.25:
						im[imsz*j+3+28*2:imsz*j+25+28*2,imsz*i+3:imsz*i+25] = [1,1,0]

		return im

	def _addLandmarkPoints(self,img,s):
		'''
		Takes the grid of images produced by saveImages and adds:
		'''

		landmark_coords = np.array([self.vpc+self.center, self.rv_center+self.center, self.base_center])
		# print(landmark_coords)

		for j in range(len(s.XYZs)):

			coords = s.XYZs[j] 
			pts = cdist(landmark_coords, coords)

			pts_vpc = np.reshape(pts[0], s.prepped_seg[0,0].shape)
			pts_rv_center = np.reshape(pts[1], s.prepped_seg[0,0].shape)
			pts_base_center = np.reshape(pts[2], s.prepped_seg[0,0].shape)
			
			pts_vpc = np.unravel_index(pts_vpc.argmin(), pts_vpc.shape)
			pts_rv_center = np.unravel_index(pts_rv_center.argmin(), pts_rv_center.shape)
			pts_base_center = np.unravel_index(pts_base_center.argmin(), pts_base_center.shape)
			
			x,y = j*s.prepped_data.shape[2]+pts_rv_center[0], pts_rv_center[1]
			img[x-5:x+5,y-5:y+5] = (1,0,0)

			x,y = j*s.prepped_data.shape[2]+pts_vpc[0], pts_vpc[1]
			img[x-5:x+5,y-5:y+5] = (0,0,1)

			x,y = j*s.prepped_data.shape[2]+pts_base_center[0], pts_base_center[1]
			img[x-5:x+5,y-5:y+5] = (0,1,0)

		if self.valve_center is not None:
			landmark_coords = np.array([self.valve_center+self.center])
			for j in range(len(s.XYZs)):
				pts = cdist(landmark_coords, coords)
				pts_valve_center = np.reshape(pts[0], s.prepped_seg[0,0].shape)
				pts_valve_center = np.unravel_index(pts_valve_center.argmin(), pts_valve_center.shape)
				x,y = j*s.prepped_data.shape[2]+pts_valve_center[0], pts_valve_center[1]
				img[x-5:x+5,y-5:y+5] = (0,1,1)

		return img

	def _addUncertaintyLabels(self, s, img, imsz, y_offset=0):

		if s.uncertainty is None:
			return img

		for i in range(s.uncertainty.shape[0]):
			uncerts = np.mean(s.uncertainty[i], axis=-1)
			img = addLabels(img, 40+imsz*i, y_offset+imsz-20, 0, imsz, ["%.2f" % (u,) for u in uncerts] ) #add the uncertainties of the segmentation predictions to the image

		return img


	def saveImages(self, downsample_factor=1, show_landmarks=True, subfolder=None, prefix='', show_uncertainty=True, use_mesh_images=False, show_vale_plane=True, overlay=False):

		ds = downsample_factor


		output_folder = self.folder['mesh_segs'] if use_mesh_images else self.folder['initial_segs']

		if subfolder is not None:
			output_folder = os.path.join(output_folder, subfolder)
		mkdir(output_folder)

		for s_ind, s in enumerate(self):#for each series:

			seg_data = s.mesh_seg if use_mesh_images else s.prepped_seg

			img = np.concatenate(np.concatenate(s.prepped_data[:,:,::ds,::ds], axis=2))
			img = to3Ch(img)

			if show_landmarks:
				self._addLandmarkPoints(img,s)

			if seg_data is not None:
				lab = np.concatenate(np.concatenate(seg_data[:,:,::ds,::ds], axis=2))
				lab = to3Ch(lab)
                
				# yyr 改颜色测试  
				image_uint8 = np.uint8(lab * 255)
				imageio.imwrite('COLOR-test.png',image_uint8)            

				imsz = s.prepped_data.shape[2]//ds

				if show_vale_plane and s.view == 'SAX':
					img = self.addValvePlaneLabels(s, img, imsz)
					lab = self.addValvePlaneLabels(s, lab, imsz)
				
				img = np.concatenate([img,lab], axis=0)
				img = (img*255).astype('uint8')

				
				if overlay:
					img1,img2 = img[:img.shape[0]//2], img[img.shape[0]//2:]              
                    
					img = np.clip(img1+img2*0.3, 0, 255)
					img = img.astype('uint8')
                      
                        
					# 颜色
					imageio.imwrite( 'COLOR.png', img2)
                    

				if show_uncertainty:
					img = self._addUncertaintyLabels(s, img, imsz, y_offset=(img.shape[0]//2)*(1-overlay))
                    
			imageio.imwrite( os.path.join(output_folder, prefix, s.series_folder_name+'.png'), img)
            
            

	
		
	def calculateSAXvsLAXintersections(self):
		'''
		This function:
			1. pruces a gif showing the detected myocardium over time (in all series/slices) + the slice intersections
			2. computes self.sax_slice_interesections which is useful for approximating the position of the valve plane
		note: this method is currently quite slow, could probably be made much faster
		'''

		mkdir(self.folder['cache'])
		# unique_name = '-'.join(self.series_names)
		unique_name = 'data_cache'
		cache_file = os.path.join(self.folder['cache'], unique_name)

		if os.path.isfile(cache_file+'.npy'):
			print('loading cached sax_slice_interesections')
			self.sax_slice_interesections = np.load(cache_file+'.npy', allow_pickle=True)

			if self.sax_slice_interesections != ():

				return

		print('calculating plane intersections (used for estimating valve plane position):')
		print('(currently a very slow brute force method, should be sped up..)')

		num_sax_slices, num_slices = 0, 0
		first_slice_offset=[]#used to handle the situation where we have multiple SAX series
		for s in self: #for each series..
			if s.view == 'SAX':
				first_slice_offset.append(num_sax_slices)
				num_sax_slices += s.slices
			num_slices += s.slices

		#calculate the total number of intersections we need to check (used for progress bar)
		total_intersections_to_check = num_slices**2
		pbar = tqdm(total = total_intersections_to_check)

		#sax_slice_interesections keeps track of how many pixels on myocardium in LAX slices 
		#each SAX slice intersects (for each time frame). We can use this later for trying 
		#heuristically guessing which SAX slices are above the valve plane
		sax_slice_interesections = np.zeros((num_sax_slices, self.time_frames))

		all_preds = []
		for i1, s1 in enumerate(self): #for each series..

			segs1 = s1.prepped_seg

			s1_normal = np.cross(s1.orientation[3:],s1.orientation[:3])

			for j1 in range(segs1.shape[1]): #for each slice..

				plane_s1_j1 = np.append(s1_normal, -np.dot(s1_normal, s1.XYZs[j1][0]))

				main_coords = s1.XYZs[j1] #get the coordinates of all pixels in the slice
				for_lines = np.zeros(segs1[0,j1].shape) #make a blank image (off the rewuired size) where we will draw the intersection lines

				current_sax_id=-1
				for i2, s2 in enumerate(self): #for each series..

					if s2.view=='SAX':
						current_sax_id+=1

					s2_normal = np.cross(s2.orientation[3:],s2.orientation[:3])

					segs2 = s2.prepped_seg

					if i2 == i1: #don't need to check a series against itself
						pbar.update(segs2.shape[1])
						continue

					# print(np.dot(s1_normal, s2_normal), s1_normal, s2_normal)
					if np.abs(np.dot(s1_normal, s2_normal)) > 0.95: #don't need to check a series against itself
						pbar.update(segs2.shape[1])
						continue

					for j2 in range(segs2.shape[1]): #for each slice..

						plane_s2_j2 = np.append(s2_normal, -np.dot(s2_normal, s2.XYZs[j2][0]))

						#get the coordinates of all pixels in the slice
						coords = s2.XYZs[j2] 
						#get all points that are close to points in the other slice
						pts = np.sum(cdist(main_coords, coords) <= 0.75, axis=1) 
						#draw these points on line image:
						pts = (pts > 0).reshape(segs1[0,j1].shape)
						for_lines[np.where(pts)] = 1

						#for intersection between SAX and non-sax slices, add the results to the sax_slice_interesections array:
						if s2.view == 'SAX' and s1.view != 'SAX':
							dj = first_slice_offset[current_sax_id] #offset handles situation where we have multiple SAX series
							for k in range(segs1.shape[0]):
								s1_k_myo = segs1[k,j1]==2
								sax_slice_interesections[j2+dj, k] += np.sum(s1_k_myo[np.where(pts)])

						pbar.update(1)
				
				slice_over_time_with_lines = ((np.clip((segs1[:,j1]==2)*1 + for_lines[None]*1,0,1))*255).astype('uint8')

				all_preds.append(slice_over_time_with_lines)

				#make the debug folder (if it doesnt already exist) to save the gif in:
				output_folder = self.folder['debug']
				mkdir(output_folder)
				fname = os.path.join(output_folder,'saveIntersectionImages.gif')
				imageio.mimsave(fname, np.concatenate(all_preds, axis=2)) #save gif

			#normalise sax_slice_interesections (i.e. make all values lie in [0,1])

			if np.sum(sax_slice_interesections) == 0:
				self.sax_slice_interesections = None
			else:
				self.sax_slice_interesections = sax_slice_interesections / np.max(sax_slice_interesections)


		pbar.close()
		print('calculateSAXvsLAXintersections() completed')

		if self.sax_slice_interesections is None:
			np.save(cache_file+'.npy', self.sax_slice_interesections)
			print('sax_slice_interesections cached')



	def predictAorticValvePosition(self):

		valve_xyzs = []
		for s in self:
			if s.view == 'SAX':

				if s.VP_heuristic2 is None or s.VP_heuristic2 is None:
					continue

				slices_to_use = (1-s.VP_heuristic2) * (1-s.slice_above_valveplane) #indicates slices with a c shape that are still within the LV

				sanitycheck_images = []
				centers = []
				approx_valve_masks = []
				for t in range(self.time_frames):
					sanitycheck_images.append([])
					for j in range(s.slices):

						if slices_to_use[t,j] and s.distance_from_center[j]  > 0:
							myo = s.prepped_seg[t,j]==2
							bp = s.prepped_seg[t,j]==3

							aprox_valve_mask = binary_dilation(bp) * (1-bp) * (1-myo)

							sanitycheck_images[-1].append(aprox_valve_mask)

							# approx_valve_masks.append( aprox_valve_mask )

							xyz = s.XYZs[j].reshape((128,128,3))[aprox_valve_mask==1]
							print( np.mean(xyz, axis=0), s.distance_from_center[j] ) 
							valve_xyzs.append( np.mean(xyz, axis=0) )
							# centers.append( com(aprox_valve_mask) )

						else:
							sanitycheck_images[-1].append(np.zeros((128,128)))

				sanitycheck_images = np.array(sanitycheck_images)
				# print(sanitycheck_images.shape)
				# imageio.imwrite('sanitycheck_%s.png' % (s.name,), 255*np.mean(sanitycheck_images.reshape((-1,128,128)),axis=0))

		if len(valve_xyzs) > 0:
			self.valve_center = np.median(np.array(valve_xyzs), axis=0)
		else:
			self.valve_center = None

		print('self.valve_center =', self.valve_center)

	def estimateLandmarks(self, series_to_use='all', center_shift=None, init_mode=1):
		'''
		this function tries to find various landmark points/directions in the image data, using the SAX series.
		specifically, it identifies:

			self.vpc (center of the most basal valve plane)
			self.sax_normal (normal vector of the short axis slice)
			self.rv_center center of mass of the right ventricle
			self.rv_direction vector pointing from vpc ro rv_center, projected onto the sax plane

		series_to_use can be a list of (SAX) series names which will be used, otherwise all SAX series are used
		'''

		#cacluate the centers of all the slices (in the DICOMS frame of ref):

		# print('###')
		# print('CALLED WITH init_mode=', init_mode)
		# print('###')

		
		if init_mode == 0:
			all_slices = []
			for s in self:
				all_slices.extend(s.XYZs)
			grid = np.concatenate(all_slices)
			self.center = np.mean(grid, axis=0)

		elif init_mode == 1:
			all_slices = []
			for s in self:
				if s.view == 'SAX':
					if s.slice_above_valveplane is None:
						all_slices.extend(s.XYZs)
					else:
						for k in range(len(s.XYZs)):
							if not s.slice_above_valveplane[0,k]:
								all_slices.append(s.XYZs[k])
				else:
					all_slices.extend(s.XYZs)
			grid = np.concatenate(all_slices)
			self.center = np.mean(grid, axis=0)


		# if center_shift is not None:
		# 	self.center -= center_shift


		self.vpc, self.sax_normal, self.rv_center = [], [], [] #make an array and then take average to handel the case of multiple SAX series
		for s in self:

			if s.view == 'SAX':

				if series_to_use is not 'all' and s.name not in series_to_use:
					continue

				### calculate the valve plane center (relative to self.center)
				#if we haven't yet estimated the valve-plane position, just use the first slice in the SAX series
				if s.slice_above_valveplane is None:
					self.vpc.append( np.mean(s.XYZs[0], axis=0) - self.center )
				else: #otherwise, use the center of first (most basal) slice in the LV:
					for k in range(len(s.XYZs)):
						if not s.slice_above_valveplane[0,k] and not np.sum(s.data[0,k])==0:
							self.vpc.append( np.mean(s.XYZs[k], axis=0) - self.center )
							break

				if len(self.vpc) == 0: #catch the (strange) case where no slice seems to be in the LV
					self.vpc.append( np.mean(s.XYZs[0], axis=0) - self.center )

				#short-axis normal:
				Xxyz = s.orientation[3:]
				Yxyz = s.orientation[:3]
				self.sax_normal.append( np.cross(Yxyz,Xxyz) )

				#get center of RV (relative to self.center):
				rv_xyzs = []
				for j in range(s.prepped_seg.shape[0]):
					
					RV = (s.prepped_seg[j] == 1)
					
					if np.sum(RV) == 0:
						continue
					
					for i in range(len(RV)):
						# if s.uncertainty is not None and np.mean(s.uncertainty[j,i],axis=-1) < 0.25:
						if s.uncertainty is not None and np.mean(s.uncertainty[j,i],axis=-1) > 0.75:
							# print(j, 'uncertain')
							continue
						if s.slice_above_valveplane is not None and s.slice_above_valveplane[j,i]:
							# print(j, 'slice_above_valveplane')
							continue

						# print(RV[i].shape)
						rv_xyzs.append(s.XYZs[i].reshape((128,128,3))[RV[i]==1])

				if len(rv_xyzs) > 0:
					rv_xyzs = np.concatenate(rv_xyzs,axis=0)
					rv_center = np.mean(rv_xyzs,axis=0) - self.center
					self.rv_center.append( rv_center )

		if len(self.vpc) > 0:
			self.vpc = np.mean(self.vpc, axis=0)
			self.sax_normal = np.mean(self.sax_normal, axis=0)
			self.rv_center = np.mean(self.rv_center, axis=0)
		else:
			print('warning: no SAX slices found for calculating landamrks in estimateLandmarks()')
			self.vpc, self.sax_normal, self.rv_center, self.rv_direction = None, None, None, None
			return

		#calculate the center of the base as a point in 3D space
		max_dist_from_center = 0
		self.base_center = None
		for s in self:
			if s.view == 'SAX':

				s.distance_from_center = []
				for k in range(len(s.XYZs)): #for each slice:

					x = np.mean(s.XYZs[k],axis=0) #center of slice
					u = self.center #center of volume
					n = self.sax_normal / np.linalg.norm(self.sax_normal, 2) #make sure normal is length 1

					#project the center of the slice onto the sax_normal line, passing up throuygh the volume center:
					intersection_point = u + n*np.dot(x - u, n)

					#calculate the ditance of this projected point fromn the center (positive--> more toward base, negative--> more toward apex)
					dist_fom_center = np.mean((intersection_point - self.center)/self.sax_normal)#all three dims should be the same, so we just take the mean
					s.distance_from_center.append( dist_fom_center )

					if dist_fom_center > max_dist_from_center:
						self.base_center = x
						max_dist_from_center = dist_fom_center

		if self.base_center is None:
			for s in self:
				if s.view == 'SAX':
					self.base_center = np.mean(s.XYZs[0],axis=0)

		if center_shift is None:
			self.estimateLandmarks(series_to_use=series_to_use, center_shift=self.sax_normal*(58-max_dist_from_center), init_mode=init_mode)

		rv_direction = self.rv_center / np.linalg.norm(self.rv_center)
		rv_direction_projected_on_sax_plane = rv_direction - np.dot(rv_direction, self.sax_normal) * self.sax_normal
		self.rv_direction = rv_direction_projected_on_sax_plane / np.linalg.norm(rv_direction_projected_on_sax_plane)

		self.predictAorticValvePosition()
		if self.valve_center is not None:
			self.valve_center = self.valve_center - self.center
			aortic_valve_direction = self.valve_center / np.linalg.norm(self.valve_center)
			aortic_valve_direction_projected_on_sax_plane = aortic_valve_direction - np.dot(aortic_valve_direction, self.sax_normal) * self.sax_normal
			self.aortic_valve_direction = aortic_valve_direction_projected_on_sax_plane / np.linalg.norm(aortic_valve_direction_projected_on_sax_plane)

		print('landmarks estimated = ', self.vpc, self.sax_normal, self.rv_center, self.valve_center)


	def voxaliseMeshAndCalculateStrains(self,):

		mesh_offset = np.array([70,70,118]) #this offset just shifts the mesh into the possitive axis quadrant for voxalizing

		slices = []

		rad, thi, lng, mvol, bpvol = [],[],[],[],[]

		for time_frame in range(self.time_frames):
			msh = self.fitted_meshes[time_frame]['mesh'][0]

			myo, bp = voxelizeUniform(msh, (128,128,128), bp_channel=True, offset=mesh_offset)
			myo = myo*1 #convert from bool to int

			#calculate which sax slices contain myocardium:
			in_slice = (np.sum(myo, axis=(0,1))>0)*1
			p = np.where(in_slice==1)[0]

			#centeral sax slice is just in the center of all slices that contain myocardium:
			cz = (p[0]+p[-1])//2

			#the two lax slices should go through the center of the most apical point:
			cx,cy=com(myo[:,:,p[0]])
			cx = np.round(cx).astype('int')
			cy = np.round(cy).astype('int')

			#to measure longditudinal length, we first find a 1 pixel thick line running throught the center of the myocardium in each LAX view:
			skel1 = skeletonize(myo[cx,:,:]*1)
			skel2 = skeletonize(myo[:,cy,:]*1)

			#the naive length estimate is then just the number of pixels in this line: (we avaegar over the two views)
			naive_length = (np.sum(skel1) + np.sum(skel2))/2

			#because the myocardium has a "U" shape, we calculate some correction terms:
			#these correction terms aim to account for the fact that some pixels are diagonally adjacent (i.e. have a distance of √2 between them, rather than 1)
			#we approximate how often this occurs by measuring the width, and then add √2-1 to the naive length estimate (i.e. the pixelo count) for each of these pixels
			cf1 = np.where(np.sum(skel1, axis=1)>=1)[0]
			cf1 = (cf1[-1]-cf1[0])*(2**0.5-1)
			cf2 = np.where(np.sum(skel2, axis=1)>=1)[0]
			cf2 = (cf2[-1]-cf2[0])*(2**0.5-1)
			myo_length = naive_length + (cf1 + +cf2)/2

			slices.append( np.concatenate([myo[cx,:,:], skel1, myo[:,cy,:], skel2, myo[:,:,cz]])*255 )

			#average estimates from three central slices (8mm apart) for thickness and radius estimates:
			myo_thickness_1, myo_radius_1 = estimateThicknessAndRadius(myo[:,:,cz])
			myo_thickness_2, myo_radius_2 = estimateThicknessAndRadius(myo[:,:,cz-8])
			myo_thickness_3, myo_radius_3 = estimateThicknessAndRadius(myo[:,:,cz+8])
			myo_thickness = (myo_thickness_1 + myo_thickness_2 + myo_thickness_3)/3
			myo_radius = (myo_radius_1 + myo_radius_2 + myo_radius_3)/3

			print("%d, %.2f, %.2f, %.2f, %.2f, %.2f" % (time_frame, myo_thickness, myo_radius, myo_length, np.sum(bp)/1000, np.sum(myo)/1000) )

			thi.append(myo_thickness)
			rad.append(myo_radius)
			lng.append(myo_length)
			mvol.append(np.sum(myo)/1000)
			bpvol.append(np.sum(bp)/1000)

			imageio.imwrite( os.path.join(self.folder['debug'], 'slices_over_cycle.png'), np.concatenate(slices, axis=1).astype('uint8') )

		fname = os.path.join(self.folder['plots'], 'incidies_from_voxalised_mesh.png')
		makeStrainPlot(fname, thi, rad, lng, mvol, bpvol)

		mkdir(self.folder['data'])
		np.save(os.path.join(self.folder['debug'], 'indecies.npy'), np.array([thi, rad, lng, mvol, bpvol]))
		

	def proxyCardiacIndecies(self, mask_to_use='network'):
		'''
		computes various cardiac indecie estimates using voxelized masks:
		if mask_to_use == 'network':
			use the segmentation masks from the neural network, 
		if mask_to_use == 'mesh':
			use the segmentation masks extracted from the mesh that was fitted to the data (i.e by voxelizing and slicing the mesh)

		specifically, this function estimate:
			RV blood pool volume over time
			LV blood pool volume over time
			LV myocardium volume over time
			LV myocardium thickness, circumfrance and length over time
			LV global strains over time (radial, circumferential, longditudinal)
			ejection fraction
			stroke volume

		the estimates are made by "pixel counting" to measure various thicknesses, lengths, areas etc

		NOTE: to compute the global and local strains using the mesh itself (rather than the resulting voxel masks) you need to use TODO
		'''

		if self.time_frames == 1:
			print('proxyCardiacIndecies() currently only works for exams with more than 1 time frame')
			return

		myo_thickness_over_time = []
		myo_radius_over_time = []

		myo_volumes_over_time = []
		myo_lengths_over_time = []
		bp_volumes_over_time = []
		rvbp_volume_over_time = []

		for s in self: #for each series..

			if s.name in self.series_to_exclude: #skip over series we didn't to use for the mesh fitting
				continue

			if mask_to_use == 'mesh':
				masks = np.round(s.mesh_seg+0)
				masks = np.sum(masks*[[[[[1,2,3]]]]], axis=-1)
			else:
				masks = s.prepped_seg
			print("s  :"+s.view)

			if s.view == 'SAX':
				myo_thickness_over_time.append([])
				myo_radius_over_time.append([])
				rvbp_volume_over_time.append([])
				j = s.slices // 2 #central slice
				for t in range(self.time_frames):

					dj_options = [-1,0,1] if masks.shape[1] >= 3 else [0] #middle three if there are three or more slice, otherwise jus take the first slice

					thicknesses, radiuses, rv_volume = [], [], []
					for dj in dj_options:

						myo = (masks[t,j+dj]==2)*1
						thickness, radius = estimateThicknessAndRadius(myo)
						thicknesses.append(thickness)
						radiuses.append(radius)

						rv = (masks[t,j]==1)*1
						rv_volume.append(rv)

					myo_thickness_over_time[-1].append( np.mean(thicknesses) )
					myo_radius_over_time[-1].append( np.mean(radiuses) )
					rvbp_volume_over_time[-1].append( np.mean(rv_volume) )

			else:
				myo_volumes_over_time.append([])
				bp_volumes_over_time.append([])
				myo_lengths_over_time.append([])
				for t in range(self.time_frames):
					j = s.slices // 2

					myo = (masks[t,j]==2)*1
					bp = (masks[t,j]==3)*1

					vol = np.sum(myo)
					if vol != 0:
						myo_volumes_over_time[-1].append(vol)
					else:
						myo_volumes_over_time[-1].append(np.nan)

					vol = np.sum(bp)
					if vol != 0:
						bp_volumes_over_time[-1].append(vol)
					else:
						bp_volumes_over_time[-1].append(np.nan)

					moy_len = np.sum(skeletonize(myo))
					if moy_len != 0:
						myo_lengths_over_time[-1].append(moy_len)
					else:
						myo_lengths_over_time[-1].append(np.nan)

		if len(myo_thickness_over_time) > 0:
			myo_thickness_over_time = np.nanmean(np.array(myo_thickness_over_time), axis=0)
		if len(myo_radius_over_time) > 0:
			myo_radius_over_time = np.nanmean(np.array(myo_radius_over_time), axis=0)
		if len(rvbp_volume_over_time) > 0:
			rvbp_volume_over_time = np.nanmean(np.array(rvbp_volume_over_time), axis=0)
		if len(myo_volumes_over_time) > 0:
			myo_volumes_over_time = np.nanmean(np.array(myo_volumes_over_time), axis=0)
		if len(bp_volumes_over_time) > 0:
			bp_volumes_over_time = np.nanmean(np.array(bp_volumes_over_time), axis=0)
		if len(myo_lengths_over_time) > 0:
			myo_lengths_over_time = np.nanmean(np.array(myo_lengths_over_time), axis=0)

		plt.plot(rvbp_volume_over_time)
		plt.savefig('rv_volume.png')
		plt.close()
		
		output_folder = self.folder['plots']
		mkdir(output_folder)

		file_name = os.path.join(output_folder,'proxy_strains_from_%s_predictions.png' % (mask_to_use,))

		makeStrainPlot(file_name, myo_thickness_over_time, myo_radius_over_time, myo_lengths_over_time, myo_volumes_over_time, bp_volumes_over_time)

		return myo_thickness_over_time, myo_radius_over_time, myo_lengths_over_time, myo_volumes_over_time, bp_volumes_over_time


	def estimateValvePlanePosition(self, use_lax_slices_if_available=True):
		'''
		The segmentation network may make myocardium predictions that go above the valve plane 
		(for example, by segmenting the atrium wall), and this can cause LV mesh fitting issues.

		So, in function we try to infer which SAX slices / fames that are above the valve plane, 
		so that we can ignore their masks during LV mesh fitting.

		#we do this using three heuristics:
			a) comparing SAX slices with LAX slices, and removing myocardium that doesn't appear in any LAX slices
			b) detecting where the LV myocardium transitions from a closed circle to a C shape, and guessing that
			this is just before the top of the LV
			c) detecting where the RV splits in two (according to the segmentation predictions), and guessing that
			this is just before the top of the LV

			note that method (a) requires at least 1 LAX view, but methods (b) and (c) only use SAX slices

		#taken together the heursitics should provide a reasoable estinate of the valve plane
		'''

		if use_lax_slices_if_available:
			self.calculateSAXvsLAXintersections() #calculates intersections between SAX and LAX imaging planes (this is also used to help estimate valve plane)

		#heuristic (c) - the second slice after the RV "splits" (when moving from the apex towards the base) is the top LV slice:
		for s in self:
			if s.view == 'SAX':
				too_high = np.zeros((self.time_frames,s.slices))
				for t in range(self.time_frames):
					for j in range(s.slices):
						c = len(np.unique(measure.label(s.prepped_seg[t,j]==1, background=0)))
						if c !=2:
							too_high[t,j] = 1
							if j > 2 and too_high[t,j-1] == 0 and too_high[t,j-2] == 0:
								too_high[t,j] = 0
							elif j > 0	 and too_high[t,j-1] == 1:
								too_high[t,j-1] = 2
				s.VP_heuristic1 = too_high
		
		#heuristic (b) - the second C-shaped slice (when moving from the apex towards the base) is the top LV slice:
		for s in self:
			if s.view == 'SAX':
				s.VP_heuristic2 = np.zeros((self.time_frames,s.slices))
				for t in range(self.time_frames):
					for j in range(s.slices):
						myo = s.prepped_seg[t,j]==2
						bp = s.prepped_seg[t,j]==3
						bpbd = binary_dilation(bp) * (1-bp)
						s.VP_heuristic2[t,j] = (np.sum(myo*bpbd)+0.00001) / (np.sum(bpbd)+0.00001)
				s.VP_heuristic2 = s.VP_heuristic2>0.98 #do this rather than ==1 to allow for the odd missing pixel etc


		#combine heuristics to single guess:
		first_slice_offset = 0
		for s in self:
			if s.view == 'SAX':
				s.slice_above_valveplane = np.zeros((self.time_frames,s.slices))

				if self.sax_slice_interesections is not None: #if sax_slice_interesections is available, use it:
					for t in range(self.time_frames):
						for j in range(s.slices):
							s.slice_above_valveplane[t,j] = (self.sax_slice_interesections[first_slice_offset+j,t] < 0.25)
				else:
					for t in range(self.time_frames):
						for j in range(s.slices):
							s.slice_above_valveplane[t,j] = (s.VP_heuristic1[t,j] == 2)

				first_slice_offset += s.slices


		#heuristic (d) - bounding box size / pixel count:

		'''
		for s in self:
			if s.view == 'SAX':
				s.VP_heuristic3 = np.zeros((self.time_frames,s.slices))
				for t in range(self.time_frames):
					for j in range(s.slices):
						if s.resampled:
							myo = s.prepped_seg_resampled[t,j]==2
						else:
							myo = s.prepped_seg[t,j]==2

						x, y = np.nonzero(myo)
						myo_area = np.pi * (((x.max()-x.min()) + (y.max()-y.min()))/4)**2

						s.VP_heuristic3[t,j] = np.sum(myo) / myo_area

				# s.VP_heuristic3 = s.VP_heuristic3>0.98 #do this rather than ==1 to allow for the odd missing pixel etc

				s.VP_heuristic3 /= np.std(s.VP_heuristic3)
				s.VP_heuristic3 -= np.mean(s.VP_heuristic3)

				from scipy.ndimage.filters import gaussian_filter

				s.VP_heuristic3 = gaussian_filter(s.VP_heuristic3, sigma=1)

				plt.imshow(s.VP_heuristic3)
				plt.colorbar()
				plt.savefig('VP_heuristic3.png')
		'''


						


	def save(self):
		mkdir(self.folder['base'])
		fname = os.path.join(self.folder['base'], 'DicomExam.pickle')
		file_to_save = open(fname, "wb")
		pickle.dump(self, file_to_save)
		file_to_save.close()

	# def load(self,file_to_load):
	# 	file_to_read = open(file_to_load, "rb")
	# 	self = pickle.load(file_to_read)
	# 	file_to_read.close()

	def visualiseSlicesIn3D(self, mesh=None, time_frame=0, interactive=False, mesh_axes=None, t=0, return_image=False, folder=''):

		mkdir(self.folder['3d_visualisations'])

		pv.set_plot_theme("document")

		for k in range(1):

			# plotter = pv.Plotter()#off_screen=True)
			if not interactive:
				plotter = pv.Plotter(off_screen=True)
			else:
				plotter = pv.Plotter()
			slice_obs = []
			for s in self: #for easch series:

				intensities = [s.data, s.seg][k]

				# for slice_index in [s.slices//2]: #central slice:
				for slice_index in [0, s.slices-1]: #central slice:
				# for slice_index in range(1):#s.slices): #for each slice:
					grid = planeToGrid(intensities.shape[2:], s.image_positions[slice_index], s.orientation, s.pixel_spacing)

					#make color map:
					blue = np.array([0.1,0.55,0.88,1])
					black = np.array([0, 0, 0, 1])
					mapping = np.linspace(0, 1, 256)
					newcolors = np.empty((256, 4))
					newcolors[mapping >= 0.5] = blue
					newcolors[mapping < 1] = black
					from matplotlib.colors import ListedColormap
					my_colormap = ListedColormap(newcolors)

					# plotter.add_mesh(grid, scalars=1*(intensities[time_frame,slice_index].T.flatten(order="F")==2), opacity=0.5, cmap='gray')#[0,0,0,1],[0.1,0.55,0.88,1]])#, cmap='gray')
					plotter.add_mesh(grid, scalars=intensities[time_frame,slice_index].T.flatten(order="F"), opacity=0.5, cmap='gray')#[0,0,0,1],[0.1,0.55,0.88,1]])#, cmap='gray')

			if mesh is not None:
				plotter.add_mesh(mesh, show_scalar_bar=False, color='grey')
		
			
			'''
			plotter.add_mesh( pv.Cylinder(center=self.center+self.vpc+self.rv_direction*50, direction=self.rv_direction, radius=5, height=100), color='red' )
			plotter.add_mesh( pv.Cone(center=self.center+self.vpc+self.rv_direction*105, direction=self.rv_direction, radius=10, height=10), color='red' )
			plotter.add_mesh( pv.Cylinder(center=self.center+self.vpc+self.sax_normal*50, direction=self.sax_normal, radius=5, height=100), color='blue' )
			plotter.add_mesh( pv.Cone(center=self.center+self.vpc+self.sax_normal*105, direction=self.sax_normal, radius=10, height=10), color='blue' )

			if mesh_axes is not None:
				mesh_vpc, mesh_sax_normal, mesh_rv_direction = mesh_axes

				plotter.add_mesh( pv.Cylinder(center=self.center+mesh_vpc+mesh_rv_direction*50, direction=mesh_rv_direction, radius=5, height=100), color='pink' )
				plotter.add_mesh( pv.Cone(center=self.center+mesh_vpc+mesh_rv_direction*105, direction=mesh_rv_direction, radius=10, height=10), color='pink' )
				plotter.add_mesh( pv.Cylinder(center=self.center+mesh_vpc+mesh_sax_normal*50, direction=mesh_sax_normal, radius=5, height=100), color='green' )
				plotter.add_mesh( pv.Cone(center=self.center+mesh_vpc+mesh_sax_normal*105, direction=mesh_sax_normal, radius=10, height=10), color='green' )
			'''

			# plotter.add_mesh( pv.Cylinder(center=self.center+self.vpc+self.sax_normal*50, direction=self.sax_normal, radius=5, height=100), color='blue' )
			# plotter.add_mesh( pv.Cone(center=self.center+self.vpc+self.sax_normal*105, direction=self.sax_normal, radius=10, height=10), color='blue' )

			if not interactive:

				if True:
					img = plotter.show(screenshot=True)[1]
					if return_image:
						return img
					if k == 0:
						imageio.imwrite(os.path.join(folder,'3dslicevis_%d.png' % (t,)), img)
					else:
						imageio.imwrite(os.path.join(folder,'3dslicevis_mask_%d.png' % (t,)), img)

				else:

					rotate_about = (self.sax_normal + self.rv_direction)/2

					plotter.show(auto_close=False)
					path = plotter.generate_orbital_path(n_points=36, factor=1.9, viewup=rotate_about)
					if mesh is None:
						plotter.open_gif("orbit.gif")
					else:
						plotter.open_gif("orbit_with_mesh.gif")
					plotter.orbit_on_path(path, write_frames=True, viewup=rotate_about)
					plotter.close()
			else:
				plotter.show()



		# plotter = pv.Plotter(off_screen=True)
		# slice_obs = []
		# t = 0
		# for s in self: #for easch series:
		# 	# for slice_index in range(1): #for each slice:
		# 	central_slice = s.slices//2
		# 	for slice_index in [central_slice]: #for each slice:
		# 	# for slice_index in range(s.slices): #for each slice:
		# 		grid = planeToGrid(s.seg.shape[2:], s.image_positions[slice_index], s.orientation, s.pixel_spacing)
		# 		plotter.add_mesh(grid, scalars=s.seg[t,slice_index].T.flatten(order="F"), cmap='gray', opacity=0.5)

		# if mesh is not None:
		# 	print('adding mesh')
		# 	plotter.add_mesh(mesh, show_scalar_bar=False)

		# # plotter.show()
		# img = plotter.show(screenshot=True)[1]
		# print(img.shape)
		# imageio.imwrite('3dslicevis_mask.png', img)


		# X,Y,Z = np.dot(M,pts)[:3]
		# X = X.reshape((img_size[0]+1, img_size[1]+1),order="F").T
		# Y = Y.reshape((img_size[0]+1, img_size[1]+1),order="F").T
		# Z = Z.reshape((img_size[0]+1, img_size[1]+1),order="F").T

		# grid = pv.StructuredGrid(X,Y,Z)
		# plotter = pv.Plotter(off_screen=True)
		# slice_obs = []
		# t = 0
		# for s in self: #for easch series:
		# 	for slice_index in range(s.slices): #for each slice:
		# 		X = s.XYZs[slice_index][...,0].reshape((128,128))
		# 		Y = s.XYZs[slice_index][...,1].reshape((128,128))
		# 		Z = s.XYZs[slice_index][...,2].reshape((128,128))
		# 		grid = planeToGrid(s.data.shape[2:], s.image_positions[slice_index], s.orientation, s.pixel_spacing)
		# 		plotter.add_mesh(grid, scalars=s.data[t,slice_index].T.flatten(order="F"), cmap='gray')
		# img = plotter.show(screenshot=True, window_size=[800,800])[1]
		# print(img.shape)
		# imageio.imwrite('3dslicevis.png', img)

		# s.XYZs

	def _getTensorLabelsAndInputImage(self, time_frame, remove_masks_above_valveplane, is_epi_pts, use_sdf=True):
		'''
		for a given time frame we collect together the labels and images from all series:
			a) assemble the labels as a tensor to be used as the target for mesh fitting
			b) create a composite image (i.e. join them all in a row) from all the corresponding images (used for visulaisation)
		'''

		sax_start, sax_end = 0, 0

		ori_tensor_labels, tensor_labels, input_images, uncertainty = [], [], [], []
		for s in self:

			if s.name in self.series_to_exclude:
				#skip over series we don't want to use for the fitting
				continue

			if s.view == 'SAX':
				sax_start = len(tensor_labels)
				sax_end = sax_start + s.slices
			for i in range(s.slices):
				# yyr 获得原始版的tensor_label,                
				ori_myo = s.prepped_seg[time_frame,i][None,None,...,None]==2
				ori_bp = s.prepped_seg[time_frame,i][None,None,...,None]==3      
                
				# 外模改成2和3
				if is_epi_pts:
					myo = (s.prepped_seg[time_frame,i][None,None,...,None]==2) | (s.prepped_seg[time_frame,i][None,None,...,None]==3)
				else:
					# yyr 内膜改成3，与血池一样                
					myo = s.prepped_seg[time_frame,i][None,None,...,None]==3
                
				bp = s.prepped_seg[time_frame,i][None,None,...,None]==3

            
				if use_sdf:
					from scipy.ndimage.morphology import distance_transform_edt
					myo = myo*1.
					ori_myo = ori_myo*1.

					myo = distance_transform_edt(myo)/10
					bp = bp*0

					ori_myo = distance_transform_edt(ori_myo)/10
					ori_bp = ori_bp*0

				input_image = s.prepped_data[time_frame,i]

				if s.uncertainty is not None:
					uncert = np.mean(s.uncertainty[time_frame,i])
				else:
					uncert = 1

				if s.view == 'SAX':
					if remove_masks_above_valveplane: #should remove masks from too high slices:
						if s.slice_above_valveplane is None:
							print("warning: s.slice_above_valveplane is None (did you call .estimateValvePlanePosition() before .meshFit()?)")
						if s.slice_above_valveplane[time_frame,i]:
							myo = myo*0
							bp = bp*0
                            
							ori_myo = ori_myo*0
							ori_bp = ori_bp*0
                            
							if not np.sum(s.data[time_frame,i]) == 0: #if it isnt an empty slice
								uncert = 1

				tensor_labels.append(np.concatenate([myo,bp], axis=1))
				ori_tensor_labels.append(np.concatenate([ori_myo,ori_bp], axis=1))
                
				input_images.append(input_image)
				uncertainty.append( uncert )
		input_image = np.tile(np.concatenate(input_images, axis=1)[...,None], (1,1,3))
		tensor_labels = np.concatenate(tensor_labels, axis=-1)
		tensor_labels = torch.Tensor( tensor_labels ).to(device)
        
		ori_tensor_labels = np.concatenate(ori_tensor_labels, axis=-1)
		ori_tensor_labels = torch.Tensor( ori_tensor_labels ).to(device)        
        
		uncertainty = torch.Tensor( np.array(uncertainty) ).to(device)

		return ori_tensor_labels, tensor_labels, input_image, sax_start, sax_end, uncertainty

	def getMeshInImageSpace(self, msh, mesh_offset, li_model, mean_arr_batch, ones_input, warp_and_slice_model, mesh_axes):
		# yyr 多了一个_
		_, _, vol_shifts_out, rot_out, _, _, _ = li_model([mean_arr_batch[:1], ones_input, 0])
		vol_shifts_out = vol_shifts_out.detach().cpu().numpy()[0]
		myR = rotation_tensor(
			rot_out[...,:1] + warp_and_slice_model.initial_alignment_rotation[0],
			rot_out[...,1:2] + warp_and_slice_model.initial_alignment_rotation[1],
			rot_out[...,2:] + warp_and_slice_model.initial_alignment_rotation[2],
		1).detach().cpu().numpy()[0]

		mesh_axes_transformed = transformMeshAxes(mesh_axes, vol_shifts_out, myR)

		msh.points = ((msh.points + mesh_offset)/128)*2 - 1
		msh.points = np.stack([msh.points[:,2],msh.points[:,1],msh.points[:,0]], axis=1)
		msh.points = np.dot(msh.points - vol_shifts_out, myR.T) 
		msh.points = msh.points*64 + self.center

		return msh

	def createEvaluationImage(self, input_image, mesh_render, outputs, tensor_labels, bp_weight, uncert, label_image_rows=True):

		#move tensors onto the cpu, and reshape them into the expected stack of slices (num, W, H, channels)
		mcolor = np.transpose(mesh_render.cpu().numpy()[0], (3,1,2,0))>0 #the result from actually doing the mesh rendering
		pred = np.transpose(outputs.detach().cpu().numpy()[0], (3,1,2,0))>0 #the result from the differential approximation to mesh rendering
		target = np.transpose(tensor_labels.detach().cpu().numpy()[0], (3,1,2,0))>0 #the target masks (ie the segmentation masks predicted by the neural network)

		#calculate the per-slice dice:
		slice_dice, has_target = slicewiseDice(pred[...,0:1], target[...,0:1])

		#make a combined image for saving: 
		if pred.shape[-1] == 2:
			pcolor = np.concatenate([pred[...,0:1],pred[...,0:1]*0,pred[...,0:1]*0], axis=-1)
			tcolor = np.concatenate([target[...,0:1]*0.1,target[...,0:1]*0.55,target[...,0:1]*0.88], axis=-1)
			mcolor = np.concatenate([mcolor[...,0:1],mcolor[...,0:1]*0,mcolor[...,1:2]*bp_weight], axis=-1)
		else:
			pcolor = pred[...,0]
			tcolor = target[...,0]
			mcolor = mcolor[...,0]

		#make intersection image which is white where the masks agree
		intersection_image = np.tile(np.round(pred[...,0:1])*target[...,0:1],(1,1,3))
		intersection_image = intersection_image + pcolor*(1-intersection_image) + tcolor*(1-intersection_image)
		# intersection_image[...,1] = (intersection_image[...,0] + intersection_image[...,2]) > 1.1

		img = np.concatenate([
			input_image,
			np.concatenate(tcolor, axis=1),
			np.concatenate(pcolor, axis=1),
			np.concatenate(intersection_image, axis=1),
			# np.concatenate(mcolor, axis=1),
		])*255
		img = img.astype('uint8')
		img = addLabels(img, 40, 128*4-20, 128, 0, ["%.2f" % (sdval,) for sdval in slice_dice] ) #add the dice vals to the image
		if uncert is not 1:
			img = addLabels(img, 40, 128*3-20, 128, 0, ["%.2f" % (u,) for u in uncert] ) #add the uncertainties of the segmentation predictions to the image


		if label_image_rows: #label image rows if required
			img = np.concatenate([np.zeros((img.shape[0],175,3))+255, img], axis=1).astype('uint8')
			img = addLabels(img, 10, 58, 0, 128, ["a) input slices", "b) network\n   preditions", "c) rendered and\n   sliced mesh", "d) intersection\n   of (b) & (c)"], blackbox=False, font_color=(0,0,0) )

		return img

	def resetModel(self, learned_inputs, eli, use_bp_channel, sz, mesh_offset, ones_input, pcaD, warp_and_slice_model, random_starting_mesh = True):

		print('reseting model')

		#initialize / reset the model to predict 0s (= the mean mesh):
		with torch.no_grad():

			for m in [learned_inputs.modes_output_layer, learned_inputs.volume_shift_layer, learned_inputs.x_shift_layer, learned_inputs.y_shift_layer, learned_inputs.volume_rotations_layer]:
				m.weight.fill_(0.)
				m.bias.fill_(0.)

			if random_starting_mesh:
				#initialize the network to start with a random mesh (rather than the mean mesh):
				for i in range(len(learned_inputs.modes_output_layer.bias)):
					learned_inputs.modes_output_layer.bias[i] = np.random.random()*2-1 #random number in [-1,1]

			msh = eli()

			if use_bp_channel:
				mean_arr, mean_bp_arr = voxelizeUniform(msh, (sz,sz,sz), is_epi_pts=is_epi_pts, bp_channel=use_bp_channel, offset=mesh_offset)
				mean_arr_batch = torch.Tensor(np.concatenate([mean_arr[None,None], mean_bp_arr[None,None]], axis=1)).to(device)
			else:
				mean_arr = voxelizeUniform(msh, (sz,sz,sz), is_epi_pts=is_epi_pts, bp_channel=use_bp_channel, offset=mesh_offset)
				mean_arr_batch = torch.Tensor(np.tile(mean_arr[None,None],(1,1,1,1,1))).to(device)

			modes_output, global_offsets, x_shifts, y_shifts, global_rotations = learned_inputs(ones_input)
			predicted_cp = pcaD(modes_output)
			warp_and_slice_model.control_points = predicted_cp

	def prepMeshMasks(self):
		'''
		combine the masks obtained from the fitted meshes into numpy arrays, ready for visualisation or saving

		specifically, for eash series s this method creates:

		s.mesh_seg = the mean segmentation mask produced from the fitted mesh(es)
		s.mesh_seg_std = the standard deviation over the segmentation mask produced from the fitted mesh(es)

		both mesh_seg and mesh_seg_std have shape = (timesteps, slices, width, height)
		'''

		num_chans = 2

		slice_positions = [0]		
		for s in self:

			if s.name in self.series_to_exclude:
				#skip over series we didn't to use for the mesh fitting
				continue

			slice_positions.append(s.slices+slice_positions[-1])
			sind,eind = slice_positions[-2],slice_positions[-1]
			sz = s.prepped_seg.shape[2]

			seg_means, seg_stds = [], []
			for t in range(self.time_frames):

				num_fitted_meshes = 0 if t not in self.fitted_meshes else len(self.fitted_meshes[t]['rendered_and_sliced'])

				if num_fitted_meshes == 0:
					seg_means.append( np.zeros( (1,s.slices,sz,sz,num_chans) ) )
					seg_stds.append( np.zeros( (1,s.slices,sz,sz,num_chans) ) )
				else:
					if num_fitted_meshes == 1:
						# print(self.fitted_meshes[t]['rendered_and_sliced'][0].shape)
						seg_means.append( self.fitted_meshes[t]['rendered_and_sliced'][0][sind:eind][None] )
						seg_stds.append( np.zeros( (1,s.slices,sz,sz,num_chans) ) )
					elif num_fitted_meshes > 1:
						seg_means.append( np.mean([self.fitted_meshes[t]['rendered_and_sliced'][k][sind:eind][None] for k in range(num_fitted_meshes)], axis=0) )
						seg_stds.append( np.std([self.fitted_meshes[t]['rendered_and_sliced'][k][sind:eind][None] for k in range(num_fitted_meshes)], axis=0) )

			s.mesh_seg = np.concatenate(seg_means, axis=0)
			s.mesh_seg_std = np.concatenate(seg_stds, axis=0)

			if s.mesh_seg.shape[-1] == 2:
				s.mesh_seg = np.concatenate([s.mesh_seg[...,:1]*0, s.mesh_seg], axis=-1)
				s.mesh_seg_std = np.concatenate([s.mesh_seg_std[...,:1]*0, s.mesh_seg_std], axis=-1)
			elif s.mesh_seg.shape[-1] == 1:
				s.mesh_seg = np.concatenate([s.mesh_seg*0, s.mesh_seg, s.mesh_seg*0], axis=-1)
				s.mesh_seg_std = np.concatenate([s.mesh_seg_std*0, s.mesh_seg_std, s.mesh_seg_std*0], axis=-1)







	def calculateStrainsFromMeshes(self, mesh_model_dir='ShapeModel'):

		calculateStrains(mesh_model_dir, self.folder['meshes'], self.folder['strain_meshes'], self.folder['plots'], range(self.time_frames))


	def getPoses(self,):

		poses = []
		for time_frame in self.fitted_meshes:
			pose = np.concatenate([
				self.fitted_meshes[time_frame]['modes'][0],
				self.fitted_meshes[time_frame]['global_offset'][0],
				self.fitted_meshes[time_frame]['global_rotation'][0],
			])
			poses.append(pose)
		poses = np.array(poses)

		return poses


	def embedPath(self):

		poses = self.getPoses()

		poses -= np.mean(poses, axis=0)
		poses /= np.std(poses, axis=0)

		from sklearn.decomposition import PCA

		pca = PCA(n_components=2)
		pca.fit(poses.T)
		X,Y = pca.components_

		plt.plot(X,Y)
		plt.savefig('embedpath.png')

	def addMeshImage(self, img, msh):

		meshio.write("tmp.vtk", msh)
		#saveMeshImage("tmp.vtk", button=2, filename='tmp_mesh_image.png', cut_on_axis=True)
		mesh_image = imageio.imread('tmp_mesh_image.png')

		mesh_image = np.concatenate([np.zeros(mesh_image[:56].shape)+255, mesh_image, np.zeros(mesh_image[:56].shape)+255])
		img = np.concatenate([img, mesh_image], axis=1).astype('uint8')
		return img


	def fitMesh(self, 

			time_frames_to_fit = 'all', #can be 'all', 'all_loop' or a list of time frames
			burn_in_length = 0, #set to 0 for no burn-in
			num_fits = 1,

			lr = 0.003, #learning rate 
			is_epi_pts = True, # 先拟合外模                
			num_modes = 25, #number of modes to use in shape model (must be in 1 to 25)

			
			training_steps = 150, #when starting from a mesh fit to the previous time frame (which should already be close)

			#what kind of mesh movement should be allowed:
			allow_global_shift_xy = True,
			allow_global_shift_z = True,
			allow_slice_shift = True,
			allow_rotations = True,

			#should the fitting ignore the predicted masks which are suspected to be above the valve plane?
			remove_masks_above_valveplane = True,

			#weights of different loss function components:
			mode_loss_weight = 0.05, #how strongly to penalise large mode values
			global_shift_penalty_weigth = 0.3, #how strongly to penalise global shifting of the mesh
			slice_shift_penalty_weigth = 10, #how strongly to penalise slice shifts
			rotation_penalty_weigth = 1, #how strongly to penalise global mesh rotation

			#set frequency for printing / saving results:
			steps_between_progress_update=50, #how many gradient descent steps to take between priniting status updates
			steps_between_fig_saves=50, #how many gradient descent steps to take between saving figures

			#slice_weighting should be one of: 'none', 'uncertainty' or 'uncertainty_threshold'
			#if slice_weighting is 'none':
			#	 --> all slices are have a weight of 1 when calculating the dice
			#if slice_weighting is 'uncertainty':
			#	 --> all slices are weighted by 1-uncertainty when calculating the dice (i.e. more certain segmentations have higher weight)
			#if slice_weighting is 'uncertainty_threshold':
			#	 --> slices have a weight of 1 if their uncertainty is lower than uncertainty_threshold, otherwise they have 0 weight 
			slice_weighting = 'uncertainty',
			# 	这里改了
			# 	slice_weighting='none',
			uncertainty_threshold = 0.5,

			#provide series names here to exlude some series from use during the mesh fitting:
			series_to_exclude = [], 

			#these you probably don't need to change:
			# yyr 本来是50               
			cp_frequency = 50, #larger numbers (e.g. 100,150) should make the differentiable rendering method faster but less accurate 
			mesh_model_dir = 'ShapeModel', #folder containing the shape model

			add_modes_label = False,

			train_mode = 'until_no_progress', #or 'normal'

			save_training_progress_gif = False, #save a gif of the training progress for each timestep

			random_starting_mesh = False,

			show_progress=True,

			disable_lax_shift=False,

			finalise_with_sdf = False,

			save_inital_mesh_orbit = True,
			):
		
		'''
		This function fits the mesh model to the (segmentation masks predicted from the) data. 
		You must have called .segment() before you call this (or .setSementation())
		'''
		print("看看cp: ", cp_frequency)

		#save the series to exclude to the DicomExam object, as they are required in various methods (also, make lower case as we make all series names lower case)
		self.series_to_exclude = [s_to_e.lower() for s_to_e in series_to_exclude]

		#here we make sure slice_weighting='none' if we don't have uncertainty values (as the other slice_weighting modes require uncertainties)
		if self[0].uncertainty is None and slice_weighting != 'none':
			print('no uncertainty values available, so cannot use slice_weighting="%s"' % (slice_weighting,))
			print('defaulting to slice_weighting="none"')
			slice_weighting = 'none'

		#set some params:
		use_bp_channel = True
		sz = 128

		use_sdf = False

		#make some folders for saving outputs:
		for f in ['debug', 'gifs', 'image_predictions', 'meshes', 'plots', 'image_space_meshes', 'strain_meshes']:
			mkdir(self.folder[f])


		# yyr 测试拟合外模是否合适            
		print("拟合的是外膜吗: ", is_epi_pts)
        
		#load the shape (mesh) model:
		'''
        PHI:            (14412, 25)
        PHI3:           (72, 25)     72 = 24 * 3，24个控制点
        mode_bounds：   (25, 2)
        mode_means：    (1, 25)
        '''        
		mesh_1, starting_cp, PHI3, PHI, mode_bounds, mode_means, mesh_offset, exterior_points_index, mesh_axes = loadSSMV2(num_modes, cp_frequency=cp_frequency, model_dir=mesh_model_dir, is_epi_pts=is_epi_pts)
		# print("看看PHI3的shape：", PHI3.shape)

		# yyr 增加axis_parameters_Decoder
		# axis_PDcoder = axis_parameters_Decoder(PHI, PHI3, num_modes)
		# axis_PDcoder.to(device)

		# print("看看mode_bounds：", mode_bounds)
		# print("看看mode_means：", mode_means)

        
		#create a (voxelized) mean array and create fixed tensors for use during training:
		if use_bp_channel:
			mean_arr, mean_bp_arr = voxelizeUniform(mesh_1, (sz,sz,sz), is_epi_pts=is_epi_pts, bp_channel=use_bp_channel, offset=mesh_offset)
			mean_arr = mean_arr.astype('float')
			mean_arr_batch = torch.Tensor(np.concatenate([mean_arr[None,None], mean_bp_arr[None,None]], axis=1)).to(device)
		else:
			mean_arr = voxelizeUniform(mesh_1, (sz,sz,sz), is_epi_pts=is_epi_pts, bp_channel=use_bp_channel, offset=mesh_offset)
			mean_arr = mean_arr.astype('float')
			mean_arr_batch = torch.Tensor(np.tile(mean_arr[None,None],(1,1,1,1,1))).to(device)		
		ones_input = torch.Tensor(np.ones((1,1))).to(device)


		#make the pytorch models:
		se = SliceExtractor((sz,sz,sz), self,
			allow_global_shift_xy=allow_global_shift_xy, 
			allow_global_shift_z=allow_global_shift_z, 
			allow_slice_shift=allow_slice_shift,
			allow_rotations=allow_rotations,
			series_to_exclude=series_to_exclude)

		mp_model, pcaD, warp_and_slice_model, full_model, learned_inputs, li_model, test_model = makeFullPPModelFromDicom(sz, 
			num_modes, starting_cp, self, mode_bounds, mode_means, PHI3, mesh_offset,
			allow_global_shift_xy=allow_global_shift_xy, 
			allow_global_shift_z=allow_global_shift_z, 
			allow_slice_shift=allow_slice_shift,
			allow_rotations=allow_rotations,
			series_to_exclude=series_to_exclude)


		if disable_lax_shift:
			li_model.setSliceShiftMask(self)

		eli = evalLearnedInputs(learned_inputs, mode_bounds, mode_means, mesh_1, PHI)

		if random_starting_mesh:

			#initialize the network to start with a random mesh (rather than the mean mesh):
			bias_n = len(learned_inputs.modes_output_layer.bias)
			learned_inputs.modes_output_layer.bias = torch.nn.Parameter(torch.Tensor(np.random.random((bias_n,))*2-1).to(device))

		


		### here we calulate the inital mesh alignment (and add it to the model): ###

		initial_mesh_vpc, initial_mesh_sax_normal, initial_mesh_rv_direction = transformMeshAxes(mesh_axes, 0, np.eye(3))

		### calculate an initial rotation, which roughly aligns the mesh model with the heart seen in the dicom data:
		rotM = getRotationMatrix(initial_mesh_sax_normal, self.sax_normal) # a) calculate rotation matrix that aligns the SAX normals
		new_mesh_rv_direction = np.dot(rotM, initial_mesh_rv_direction) # b) get the update rv direction (i.e. after applying rotM)
		
		valve_direction = self.rv_direction if self.valve_center is None else self.aortic_valve_direction #if we cant find the valve just point towards RV
			
		rotM2 = getRotationMatrix(new_mesh_rv_direction, valve_direction) # c) calculate the rotation matrix align the rv directions
		rotM = np.dot(rotM2, rotM) #d) combine rotations
		euler_rot = np.array(Rotation.from_matrix(rotM).as_euler('xyz')) #e) convert to euler angles (used by the model)

		#set the calculated rotations (as euler angles) as the initial rotation used in the model:
		with torch.no_grad():
			warp_and_slice_model.initial_alignment_rotation += torch.Tensor(euler_rot).to(device)
			se.initial_alignment_rotation += torch.Tensor(euler_rot).to(device)

			# #test
			# warp_and_slice_model.initial_mesh_offset -= torch.Tensor(self.vpc/64).to(device)
			# se.initial_mesh_offset -= torch.Tensor(self.vpc/64).to(device)
		
	
		save_timesteps_gif = False #save a gif showing the final results over all frames
		save_final_fit_image = False #save an image of the final prediction
		save_mesh = True #save the predicted mesh
		save_mesh_in_image_space = True

		all_dice_results = [] #stores the highest dice result seen during each fit
		best_predictions = [] #stores the prediction corresponding to the highest dice
		best_meshes = []
		best_outputs = []
		all_target_label = []
		current_target_label = []


		if time_frames_to_fit == 'all':
			tf_to_fit = list(range(-burn_in_length,0))+list(range(self.time_frames))
		elif time_frames_to_fit == 'all_loop':
			tf_to_fit = list(range(-burn_in_length,0))+list(range(self.time_frames))*5
		else:
			tf_to_fit = time_frames_to_fit



		for rep in range(num_fits):
			ts_gif_frames = []

			if rep != 0:

				self.resetModel(learned_inputs, eli, use_bp_channel, sz, mesh_offset, ones_input, pcaD, warp_and_slice_model)

			for time_frame in tf_to_fit:

				burn_in = time_frame < 0 #flag indicating whether this is the burn-in phase

				if burn_in:
					print('fitting time-frame %d (burn in period)' % (time_frame,))
				else:
					print('fitting time-frame %d' % (time_frame,))

				# train_steps = initial_training_steps if time_frame == 0 else training_steps
				train_steps = training_steps

				#extract labels (used as target for fitting) and image (used foir visualsation) for the given time frame:
				ori_tensor_labels, tensor_labels, input_image, sax_start, sax_end, uncert = self._getTensorLabelsAndInputImage(time_frame, remove_masks_above_valveplane, is_epi_pts=is_epi_pts, use_sdf=use_sdf)


				if save_inital_mesh_orbit:
					msh = eli()
					imspace_msh = self.getMeshInImageSpace(deepcopy(msh), mesh_offset, li_model, mean_arr_batch, ones_input, warp_and_slice_model, mesh_axes)
					#self.visualiseSlicesIn3D(mesh=imspace_msh, folder=self.folder['debug'])
					save_inital_mesh_orbit = False

				with torch.no_grad():

					if slice_weighting == 'none':
						slice_weights = 1
					elif slice_weighting == 'uncertainty':
						slice_weights = 1-uncert #higher weight for more certain slices
						slice_weights = slice_weights/torch.mean(slice_weights)
					elif slice_weighting == 'uncertainty_threshold':
						slice_weights = uncert<uncertainty_threshold #higher weight for more certain slices
					else:
						print('unknown slice_weighting: %s (should be "none", "uncertainty" or "uncertainty_threshold")' % (slice_weighting,))
						print('defaulting to slice_weighting = "none"')
						slice_weights = 1

				opt_method = optim.Adam

				optimizer = opt_method(li_model.parameters(), lr=lr) #(re)initialise the optimiser
				
				tp_gif_frames = [] #create an array for images that will be used for the training progress (tp) gif
				fitting_3D_gif_frames = [] #create an array for images that will be used for the training progress (tp) gif

				myo_weight, bp_weight = 1, 1

				ts3 = train_steps/3

				i = 0
				dice_history = []
				prediction_history = []

				def should_continue_training(i, dice_history):

					if train_mode == 'until_no_progress':
						if len(dice_history) < 2 or (len(dice_history) - np.argmax(dice_history)) < 4 or i < train_steps:
							return True
						else:
							return False
					else:
						return i < train_steps

				axis_parameters_list = [] 
				best_dice = 0
				while should_continue_training(i, dice_history):
					optimizer.zero_grad()
					# yyr 增加axis_parameters
					outputs, modes_out, global_shifts_out, rot_out, predicted_cp, slice_shifts_out, axis_parameters = li_model([mean_arr_batch[:1], ones_input, 0])
					axis_parameters_list.append(axis_parameters)


					if i > 2*ts3:
						bp_weight = 0
					elif i > ts3:
						bp_weight = (2*ts3-i)/ts3
					else:
						bp_weight = 1

					# yyr 增加一个axis_parameters_loss
					dice_loss, modes_loss, global_shift_loss, rotation_loss, slice_shift_loss, axis_parameters_loss = meshFittingLoss(outputs, modes_out, global_shifts_out, slice_shifts_out, rot_out, tensor_labels, axis_parameters,
						mode_loss_weight, global_shift_penalty_weigth, slice_shift_penalty_weigth, rotation_penalty_weigth, myo_weight, bp_weight, slice_weights, use_sdf)

					# loss = dice_loss + modes_loss + global_shift_loss + rotation_loss + slice_shift_loss 
					loss = dice_loss + modes_loss + global_shift_loss + rotation_loss + slice_shift_loss + axis_parameters_loss

					# if dice_loss < 0.5:
					# 	# print('smaller!')
					# 	loss = loss + torch.mean(outputs)*12

					loss.backward()
					optimizer.step()

					# print("看看梯度",axis_parameters.grad)

					with torch.no_grad():

						# print('torch.mean(outputs) =', torch.mean(outputs))

						#every so many steps we save some things:
						if i % steps_between_fig_saves == 0: 

							msh = eli() #get the current mesh from the model

							#voxelize the mesh and extract the slices (applying learned rotations and offsets):
							mesh_render = getSlices(se, msh, sz, use_bp_channel, mesh_offset, learned_inputs, ones_input, is_epi_pts=is_epi_pts)

							update_starting_mesh = True
							if update_starting_mesh:
								### update the starting mesh 
								### (doing this sometimes makes the approximate prediction more accurate, by avoiding approximating large deformations)

								# a) render (voxelize) the currently predicted mesh:
								if use_bp_channel:
									mean_arr, mean_bp_arr = voxelizeUniform(msh, (sz,sz,sz), is_epi_pts=is_epi_pts, bp_channel=use_bp_channel, offset=mesh_offset)

									if finalise_with_sdf and len(dice_history) > 1 and (dice_history[-1]-dice_history[-2])<0.007 and use_sdf == False:
										use_sdf = True
										print('### switch to finetuning with SDF')
										ori_tensor_labels, tensor_labels, input_image, sax_start, sax_end, uncert = self._getTensorLabelsAndInputImage(time_frame, remove_masks_above_valveplane, is_epi_pts=is_epi_pts, use_sdf=use_sdf)
										# optimizer = opt_method(li_model.parameters(), lr=lr/4) #(re)initialise the optimiser

									if use_sdf:
										mean_arr = mean_arr*1.
										from scipy.ndimage.morphology import distance_transform_edt
										for ma_i in range(128):
											mean_arr[:,:,ma_i] = distance_transform_edt(mean_arr[:,:,ma_i])/10
											mean_bp_arr[:,:,ma_i] = mean_bp_arr[:,:,ma_i]*0

									mean_arr_batch = torch.Tensor(np.concatenate([mean_arr[None,None], mean_bp_arr[None,None]], axis=1)).to(device)
									
								else:
									mean_arr = voxelizeUniform(msh, (sz,sz,sz), is_epi_pts=is_epi_pts, bp_channel=use_bp_channel, offset=mesh_offset)
									mean_arr_batch = torch.Tensor(np.tile(mean_arr[None,None],(1,1,1,1,1))).to(device)

								# b) set the currently predicted control points as the initial control points: 	
								modes_output, global_offsets, x_shifts, y_shifts, global_rotations, axis_parameters = learned_inputs(ones_input)
								predicted_cp = pcaD(modes_output, axis_parameters)
								warp_and_slice_model.control_points = predicted_cp

								optimizer = opt_method(li_model.parameters(), lr=lr)

					

						if i % steps_between_progress_update == 0: #every so many steps we print the current progress:

							#calculate the per-slice dice:
							mcolor = np.transpose(mesh_render.detach().cpu().numpy()[0], (3,1,2,0))
							pred = np.transpose(outputs.detach().cpu().numpy()[0], (3,1,2,0))>0
                            
                            
# 							print("看看pred.shape：", pred.shape)  

# 							# yyr
							from torchvision import utils  
							# utils.save_image(tensor_labels[0,0,:,:,3], 'tensor_labels.png')  
							# utils.save_image(outputs[0,0,:,:,3], 'outputs.png')  
# 							print("看看outputs：", outputs[0,0,:,:,3][64][64])                            
                            
							if use_sdf:
								ori_tensor_labels, tensor_labels_real, _, _, _, _ = self._getTensorLabelsAndInputImage(time_frame, remove_masks_above_valveplane, is_epi_pts=is_epi_pts, use_sdf=False)
								target = np.transpose(tensor_labels_real.detach().cpu().numpy()[0], (3,1,2,0))
							else:
								target = np.transpose(tensor_labels.detach().cpu().numpy()[0], (3,1,2,0))
							slice_dice, has_target = slicewiseDice(pred[...,0:1], target[...,0:1])

							current_dice = np.sum(slice_dice)/np.sum(has_target)
							latest_loss = loss.item()
                        
  
                            
							# 只要有更好的dice，就储存当前的mesh, outputs   
							# 在这里应该对eli的shift进行均值操作          
							if current_dice > best_dice:
								best_dice = current_dice
								best_mesh = eli()                                
								time_frame_best_outputs = outputs
								time_frame_target_label = ori_tensor_labels
								mytarget = tensor_labels                         
                                
								print("看看最好的slice_dice：", slice_dice)  
								# print("看看最好的dice：", current_dice) 
                                
                            
							if show_progress:
								print("%d/%d: loss = %.3f, dice = %.3f, aproximation error=%.3f, size=%.3f" % (i, train_steps, latest_loss, current_dice, np.sum( (mcolor-pred)**2), torch.mean(outputs) ) )
								print("loss breakdown: 1-dice=%.3f, modes=%.3f, global shifts=%.3f, slice shifts=%.3f, rotation=%.3f" % (dice_loss.item(), modes_loss.item(), global_shift_loss.item(), slice_shift_loss.item(), rotation_loss.item()))	

							dice_history.append( current_dice )
							prediction_history.append( pred )


					i+=1

				# yyr 储存最好的mesh, outputs
				best_meshes.append(best_mesh)                
				best_outputs.append(time_frame_best_outputs)
				all_target_label.append(time_frame_target_label)
				current_target_label.append(mytarget) 

				# yyr ======================================================================
				# # 提取每个张量中的X轴数值  
				# x_values_list = [t[0, 0] for t in axis_parameters_list]  
				# # 求最大值、最小值和平均值  
				# max_x_value = torch.max(torch.stack(x_values_list))  
				# min_x_value = torch.min(torch.stack(x_values_list))  
				# mean_x_value = torch.mean(torch.stack(x_values_list).to(torch.float))  
  
				# print("x轴的最大值:", max_x_value.item())  
				# print("x轴的最小值:", min_x_value.item())  
				# print("x轴的平均值:", mean_x_value.item()) 


				# # 提取每个张量中的X轴数值  
				# y_values_list = [t[0, 1] for t in axis_parameters_list]  
				# # 求最大值、最小值和平均值  
				# max_y_value = torch.max(torch.stack(y_values_list))  
				# min_y_value = torch.min(torch.stack(y_values_list))  
				# mean_y_value = torch.mean(torch.stack(y_values_list).to(torch.float))  
  
				# print("y轴的最大值:", max_y_value.item())  
				# print("y轴的最小值:", min_y_value.item())  
				# print("y轴的平均值:", mean_y_value.item()) 


				# z_values_list = [t[0, 2] for t in axis_parameters_list]  
				# # 求最大值、最小值和平均值  
				# max_z_value = torch.max(torch.stack(z_values_list))  
				# min_z_value = torch.min(torch.stack(z_values_list))  
				# mean_z_value = torch.mean(torch.stack(z_values_list).to(torch.float))  
  
				# print("z轴的最大值:", max_z_value.item())  
				# print("z轴的最小值:", min_z_value.item())  
				# print("z轴的平均值:", mean_z_value.item()) 
				# ====================================================================                
                    
				print('finished in %d steps' % (i,))

				all_dice_results.append( np.max(dice_history) )
				best_predictions.append( prediction_history[np.argmax(dice_history)] )

				if burn_in:
					continue

				### training completed for the current time_frame, so save some results

				with torch.no_grad():
					#get the final mesh from the model (and the rendered and sliced results)
					msh, modes = eli(just_mesh=False)
                    
					# print("msh的point：", msh.points.shape)   
					# print("msh的label：", msh.point_data["labels"].shape) 
                
					mesh_render = getSlices(se, msh, sz, use_bp_channel, mesh_offset, learned_inputs, ones_input, is_epi_pts=is_epi_pts)
					#get the final modes, offsets and rotations:
					modes_output, global_offsets, x_shifts, y_shifts, global_rotations, _ = learned_inputs(ones_input)
					#get the mesh in image space:
					imspace_msh = self.getMeshInImageSpace(deepcopy(msh), mesh_offset, li_model, mean_arr_batch, ones_input, warp_and_slice_model, mesh_axes)
					#calculate the dice loss between targets (neural net seg predictions) and predictions (rendered and sliced mesh):
					pred = np.transpose(outputs.detach().cpu().numpy()[0], (3,1,2,0))
					target = np.transpose(tensor_labels.detach().cpu().numpy()[0], (3,1,2,0))
					slice_dice, has_target = slicewiseDice(pred[...,0:1], target[...,0:1])
					mean_dice = np.sum(slice_dice*has_target)/np.sum(has_target)

				#create dictionary entry for this timeframe (if required):
				if time_frame not in self.fitted_meshes:
					self.fitted_meshes[time_frame] = {}
				elif time_frames_to_fit == 'all_loop':

					mode_inter_frame_distances, offset_inter_frame_distances, rotation_inter_frame_distances = [], [], []
					for t_f in range(self.time_frames):

						this_frames_modes = self.fitted_meshes[t_f]['modes'][0]
						next_frames_modes = self.fitted_meshes[(t_f+1)%self.time_frames]['modes'][0]
						mode_inter_frame_distances.append( np.sum((next_frames_modes-this_frames_modes)**2)**0.5 )

						this_frames_offset = self.fitted_meshes[t_f]['global_offset'][0]
						next_frames_offset = self.fitted_meshes[(t_f+1)%self.time_frames]['global_offset'][0]
						offset_inter_frame_distances.append( np.sum((next_frames_offset-this_frames_offset)**2)**0.5 )

						this_frames_rotation = self.fitted_meshes[t_f]['global_rotation'][0]
						next_frames_rotation = self.fitted_meshes[(t_f+1)%self.time_frames]['global_rotation'][0]
						rotation_inter_frame_distances.append( np.sum((next_frames_rotation-this_frames_rotation)**2)**0.5 )

					mean_mode_ifd = np.mean(mode_inter_frame_distances) 
					this_frame_self_mode_distance = np.sum((modes[0]-self.fitted_meshes[time_frame]['modes'][0])**2)**0.5
					similar_modes = this_frame_self_mode_distance <= mean_mode_ifd/2
						
					mean_offset_ifd = np.mean(offset_inter_frame_distances) 
					this_frame_self_offset_distance = np.sum((global_offsets.cpu().numpy()[0,0]-self.fitted_meshes[time_frame]['global_offset'][0])**2)**0.5
					similar_offsets = this_frame_self_offset_distance <= mean_offset_ifd/2
						
					mean_rotation_ifd = np.mean(rotation_inter_frame_distances) 
					this_frame_self_rotation_distance = np.sum((global_rotations.cpu().numpy()[0,0]-self.fitted_meshes[time_frame]['global_rotation'][0])**2)**0.5
					similar_rotations = this_frame_self_rotation_distance <= mean_rotation_ifd/2

					print("%d, mode: %.3f, %.3f" % (time_frame, this_frame_self_mode_distance, mean_mode_ifd) )
					print("%d, offset: %.3f, %.3f" % (time_frame, this_frame_self_offset_distance, mean_offset_ifd) )
					print("%d, rotation: %.3f, %.3f" % (time_frame, this_frame_self_rotation_distance, mean_rotation_ifd) )

					if similar_modes and similar_offsets and similar_rotations:
						break

					self.fitted_meshes[time_frame] = {}
					if save_timesteps_gif:
						ts_gif_frames = ts_gif_frames[1:]

				#add results to the fitted_meshes dictionary:
				#(note, each key points to a list, rather than a single result, as we may fit the mesh multiple times to the same timeframe, and we want to keep all results)

				self.fitted_meshes[time_frame].setdefault('mesh',[]).append( deepcopy(msh) )
				self.fitted_meshes[time_frame].setdefault('imspace_mesh',[]).append( imspace_msh )
				self.fitted_meshes[time_frame].setdefault('modes',[]).append( modes[0] )
				self.fitted_meshes[time_frame].setdefault('global_offset',[]).append( global_offsets.cpu().numpy()[0,0] )
				self.fitted_meshes[time_frame].setdefault('x_shifts',[]).append( x_shifts.cpu().numpy()[0,:,0] )
				self.fitted_meshes[time_frame].setdefault('y_shifts',[]).append( y_shifts.cpu().numpy()[0,:,0] )
				self.fitted_meshes[time_frame].setdefault('global_rotation',[]).append( global_rotations.cpu().numpy()[0,0] )
				self.fitted_meshes[time_frame].setdefault('dice',[]).append( mean_dice )
				self.fitted_meshes[time_frame].setdefault('rendered_and_sliced',[]).append( np.transpose(mesh_render.cpu().numpy()[0], (3,1,2,0)) )

				rep_string = '_rep=%d' % (rep,) if num_fits > 1 else '' #if we are doing multiple tis append a rep counter
				modes_string = '_modes=%d' % (num_modes,) if add_modes_label else '' #if we are doing multiple tis append a rep counter
				

				
				# if save_mesh:
				# 	meshio.write(os.path.join(self.folder['meshes'],"mesh_t=%d%s.vtk" % (time_frame,rep_string)), msh)

				if save_mesh_in_image_space: #save the mesh (alined to image space)
					meshio.write( os.path.join(self.folder['image_space_meshes'],"image_space_mesh_t=%d%s.vtk" % (time_frame,rep_string)), imspace_msh)
                    


			if save_timesteps_gif:
				imageio.mimsave(os.path.join(self.folder['gifs'],'timesteps_rep=%d.gif' % (rep,)), ts_gif_frames)
                    
          
		
		self.prepMeshMasks() #combine the masks obtained from the fitted meshes into numpy arrays, ready for visualisation or saving
        
		return all_dice_results, best_predictions, best_meshes, best_outputs, all_target_label, current_target_label
    
	def mergeMeshes(self, best_epi_meshes, best_endo_meshes):
		'''
        yyr 合并mesh，有内膜点的坐标，外模点的坐标，将相应坐标的点替换掉
        '''  
		# 获得内膜下标
		endo_pts = np.load(os.path.join('ShapeModel/Boundary_nodes/ENDO_points.npy'))
        
		final_meshes = []        
		len_meshes = len(best_epi_meshes)
		len_MeshPoints = len(best_epi_meshes[0].points)        
        
		# 把外模mesh的内膜点替换掉
		for i in range(len_meshes):    # 遍历每个mesh
			for j in range(len_MeshPoints):             # 替换endo-pts
				if j in endo_pts:                    
					best_epi_meshes[i].points[j] = best_endo_meshes[i].points[j]
				else:
					best_epi_meshes[i].points[j] = ( best_epi_meshes[i].points[j] + best_endo_meshes[i].points[j] ) / 2.0                 

			final_meshes.append(best_epi_meshes[i])
            
		return final_meshes  
    
    
    
	def getFinalOutputs(self, best_epi_outputs, best_endo_outputs):
		'''
        yyr 通过相减生成outputs
        '''   
		len_outputs = len(best_epi_outputs)  
        
		final_outputs = []
		for i in range(len_outputs): 
			best_epi_outputs[i] = torch.where(best_epi_outputs[i] != 0, torch.tensor(1), best_epi_outputs[i]) 
			best_endo_outputs[i] = torch.where(best_endo_outputs[i] != 0, torch.tensor(1), best_endo_outputs[i]) 
			new_outputs = abs(torch.sub(best_epi_outputs[i], best_endo_outputs[i]))
			final_outputs.append(new_outputs)
        
        
			c = torch.sum(best_endo_outputs[i])
			# print("============================================/n", c)
            
		return final_outputs  
    
    
        
	def getFinalDice(self, final_outputs, all_target_label, fitted_timeframe):
		'''
        yyr 计算最终的dice
        '''

		for i in range(len(final_outputs)):  
			utils.save_image(all_target_label[i][0,0,:,:,5], 'current_tensor_labels.png')
			utils.save_image(final_outputs[i][0,0,:,:,5], 'current_final_outputs.png')
            
            
			target = np.transpose(all_target_label[i].detach().cpu().numpy()[0], (3,1,2,0))
			pred = np.transpose(final_outputs[i].detach().cpu().numpy()[0], (3,1,2,0))>0
            
			slice_dice, has_target = slicewiseDice(pred[...,0:1], target[...,0:1])
            
			current_dice = np.sum(slice_dice)/np.sum(has_target)
           
			# print("=====================================================")
			# print("第%d个时期的slice_dice：" % fitted_timeframe, slice_dice)        
			# print(current_dice)
			# print("=====================================================")
			return current_dice        
            
            
	def test_getFinalDice(self, best_epi_outputs, best_endo_outputs, epi_target, endo_target):
		'''
        yyr 计算最终的dice
        '''
		len_outputs = len(best_epi_outputs)  
        
		final_outputs , final_targets = [], []
		for i in range(len_outputs):         
			new_outputs = best_epi_outputs[i] - best_endo_outputs[i]
			final_targets.append(epi_target[i] - endo_target[i])            
			final_outputs.append(new_outputs)  
		

            
   
		target = np.transpose(final_targets[0].detach().cpu().numpy()[0], (3,1,2,0))
		pred = np.transpose(final_outputs[0].detach().cpu().numpy()[0], (3,1,2,0))>0
		slice_dice, has_target = slicewiseDice(pred[...,0:1], target[...,0:1])
		current_dice = np.sum(slice_dice)/np.sum(has_target)
		print("看看保存数据的slice_dice：", slice_dice)
		# print("看看has_target：", np.sum(has_target))            
		print(current_dice)
        
	def saveMeshes(self, best_meshes, fitted_timeframe):
		'''
        yyr 保存mesh到vtk文件中
        '''
		for i in range(len(best_meshes)):         
			meshio.write(os.path.join(self.folder['meshes'],"Merge-mesh_t=%d.vtk" % fitted_timeframe), best_meshes[i])    
            
            
            
	def saveEpiEndoMeshes(self, best_meshes, isEpi):
		'''
        yyr 分别保存内外模的mesh文件
        '''
		if isEpi:
			path = 'Epi-meshes/'
		else:
			path = 'Endo-meshes/'            
		for i in range(len(best_meshes)):         
			meshio.write(os.path.join(path,"mesh_t=%d.vtk" % i), best_meshes[i], file_format="vtk")      
    


def nameCheck(file_name):
	return ( (file_name[0] != '.') and ('.gif' not in file_name))

def getSortedFilenames(dicom_path):

	lstFilesDCM = []
	for dirName, subdirList, fileList in os.walk(dicom_path):
		for filename in fileList:
			if nameCheck(filename):
				lstFilesDCM.append(os.path.join(dirName,filename))
	lstFilesDCM = sorted(lstFilesDCM)

	return lstFilesDCM


'''
def makeSegDicom(dicom_path, seg_data, output_folder):

	mkdir(output_folder)

	_,_,image_ids,_,_,_,_,_,multifile = dataArrayFromDicom(dicom_path)

	lstFilesDCM = getSortedFilenames(dicom_path)

	if multifile:

		assert seg_data.shape[:2] == image_ids.shape

		seg_data = np.round(seg_data)

		print(seg_data.dtype, np.unique(seg_data))
		# sys.exit()
		for i in range(image_ids.shape[0]):
			for j in range(image_ids.shape[1]):
				fname = lstFilesDCM[int(image_ids[i,j])]
				print('loading', fname)
				f = pydicom.read_file(fname)
				file = fname.split('/')[-1]

				print('saving to %s' % (output_folder+'/'+file))

				f.PixelData = seg_data[i,j].astype(f.pixel_array.dtype).tostring()

				print(f.pixel_array.dtype, f.pixel_array.shape)
				print(seg_data.dtype, seg_data.shape)
				f.save_as(output_folder+'/'+file)

		print(dicom_path, output_folder)

	else:

		fname = lstFilesDCM[0]
		f = pydicom.read_file(fname)
		file = fname.split('/')[-1]

		seg_data = np.round(seg_data)

		print('saving to %s' % (output_folder+'/'+file))

		print(np.unique(seg_data[0].astype(f.pixel_array.dtype)))

		f.PixelData = seg_data[0].astype(f.pixel_array.dtype).tostring()

		# print(f.pixel_array.dtype, f.pixel_array.shape)
		# print(seg_data.dtype, seg_data.shape)
		f.save_as(output_folder+'/'+file)

		# print('makeSegDicom not yet implemented for multifile = False')
'''


				

def dataArrayFromDicom(PathDicom, multifile='unknown'):

	if multifile == 'unknown':

		lstFilesDCM = getSortedFilenames(PathDicom)

		if len(lstFilesDCM) >= 3:
			multifile = True
			# print('guessing dicom format as multi-file')

		else:
			multifile = False
			# print('guessing dicom format as single-file')
			# return dataArrayFromDicomSingleFile(PathDicom)
		
	if multifile == True:
		return dataArrayFromDicomFolder(PathDicom)
		# try:
		# 	return dataArrayFromDicomFolder(PathDicom)
		# except:
		# 	print( pydicom.dcmread(PathDicom) )
	elif multifile == False:
		print('reading single file DICOMs is not yet well implemented')
		# return dataArrayFromDicomFolder(PathDicom)
		return dataArrayFromDicomSingleFile(PathDicom)
	else:
		print('error, "multifile" should be True or False, got:', multifile)
		return None


def dataArrayFromDicomSingleFile(PathDicom):

	print(PathDicom)

	lstFilesDCM = getSortedFilenames(PathDicom)

	if len(lstFilesDCM) == 0:
		f = pydicom.read_file(PathDicom)
	else:
		f = pydicom.read_file(lstFilesDCM[0])

	print(f)
	sys.exit()

	# print(f.dir('rows'))
	# # print(f.keys())
	# print(f.NumberOfFrames)
	# print(f.Rows)
	# print(f.Columns)

	# for k, el in enumerate(f.elements()):
	# 	if len(str(el)) > 200:

	# 		print( k, el.tag, len(el.value), el.VR)


	dicom_dir_details = {
		'SliceLocation':f.get('SliceLocation', '?'),
		'InstanceNumber':f.get('InstanceNumber', '?'),
		'ImagePosition':f.get('ImagePositionPatient', '?'),
		'ImageOrientation':f.get('ImageOrientationPatient', '?'),
		'PatientPosition':f.get('PatientPosition', '?'),
		'PixelSpacing':f.get('PixelSpacing', '?'),
	}



	try:
		imgdata = f.pixel_array
	except:
		imgdata = np.zeros((1,1,1))

	if np.prod(imgdata.shape) > 1:
		print(imgdata.shape, dicom_dir_details)


	is3D = True
	multifile = False

	return imgdata[None], (0.45,0.45,0.45), None, dicom_dir_details, None, None, None, is3D, multifile


def dataArrayFromDicomFolder(PathDicom):

	lstFilesDCM = getSortedFilenames(PathDicom)
				
	# Get ref file (find one that has pixel spacing)
	for i, f in enumerate(lstFilesDCM):
		try:
			RefDs = pydicom.read_file(lstFilesDCM[i], force=True)
			RefDs.PixelSpacing != None
			break
		except:
			pass

	dicom_dir_details = {
		'SliceLocation':RefDs.get('SliceLocation', '?'),
		'InstanceNumber':RefDs.get('InstanceNumber', '?'),
		# 'ImageSize':RefDs.pixel_array.shape,
		'ImagePosition':RefDs.get('ImagePositionPatient', '?'),
		'ImageOrientation':RefDs.get('ImageOrientationPatient', '?'),
		'PatientPosition':RefDs.get('PatientPosition', '?'),
		'PixelSpacing':RefDs.get('PixelSpacing', '?'),
	}

	# Load spacing values (in mm)
	ConstPixelSpacing = (float(RefDs.SliceThickness), float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]))

	# loop through all the DICOM files and build lists of spatial and temporal positions:
	slice_locations, trigger_times = [], []
	for filenameDCM in lstFilesDCM:
		ds = pydicom.read_file(filenameDCM, force=True)
		location, t_time = ds.get('SliceLocation', '?'), ds.get('TriggerTime', '?')
		if location != '?' and t_time != '?':
			slice_locations.append( location )
			trigger_times.append( t_time )

	slice_locations= list(set(slice_locations))
	slice_locations.sort()
	trigger_times= list(set(trigger_times))
	trigger_times.sort()

	# work out data size from unique spatial and temporal positions:
	data = np.zeros( (len(trigger_times), len(slice_locations), int(RefDs.Rows), int(RefDs.Columns)), dtype=RefDs.pixel_array.dtype)
	placment = np.zeros((len(trigger_times), len(slice_locations)))
	image_ids = np.zeros((len(trigger_times), len(slice_locations)))

	# load images into 4D data array:
	image_positions = [None for i in range(len(slice_locations))]
	for i, filenameDCM in enumerate(lstFilesDCM):
		ds = pydicom.read_file(filenameDCM, force=True)
		location, t_time = ds.get('SliceLocation', '?'), ds.get('TriggerTime', '?')
		if location != '?' and t_time != '?':
			z = slice_locations.index(location)
			t = trigger_times.index(t_time)
			if ds.pixel_array.shape == data[t,z].shape:
				data[t,z] = ds.pixel_array
			placment[t,z] = 1
			image_ids[t,z] = i
			image_positions[z] = ds.get('ImagePositionPatient', '?')


	#squash to solve small variations in trigger_time:
	i = 0
	while i < data.shape[0]-1:
		if np.max((placment[i]>0)*1 + (placment[i+1]>0)*1) <= 1:
			data = np.concatenate([data[:i], data[i+1:i+2]+data[i:i+1], data[i+2:]], axis=0)
			placment = np.concatenate([placment[:i], placment[i+1:i+2]+placment[i:i+1], placment[i+2:]], axis=0)
			image_ids = np.concatenate([image_ids[:i], image_ids[i+1:i+2]+image_ids[i:i+1], image_ids[i+2:]], axis=0)
		else:
			i += 1

	#todo: check that this is actually false:
	is3D = False

	multifile = True

	return data, ConstPixelSpacing, image_ids, dicom_dir_details, slice_locations, trigger_times, image_positions, is3D, multifile


import re 

def sorted_nicely( l ): 
	""" Sort the given iterable in the way that humans expect.""" 
	convert = lambda text: int(text) if text.isdigit() else text 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)
