import pydicom
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import imageio

from utilsForDicom import *

import pyvista as pv

# def dataArrayFromDicom(PathDicom, multifile='unknown'):

# 	if multifile == 'unknown':

# 		lstFilesDCM = []
# 		for dirName, subdirList, fileList in os.walk(PathDicom):
# 			for filename in fileList:
# 				lstFilesDCM.append(os.path.join(dirName,filename))

# 		if len(lstFilesDCM) >= 3:
# 			multifile = True
# 			print('guessing dicom format as multi-file')

# 		else:
# 			multifile = False
# 			print('guessing dicom format as single-file')
		
# 	if multifile == True:
# 		return dataArrayFromDicomFolder(PathDicom)
# 	elif multifile == False:
# 		print('reading single file DICOMs is not yet implemented')
# 		return dataArrayFromDicomSingleFile(PathDicom)
# 	else:
# 		print('error, "multifile" should be True or False, got:', multifile)
# 		return None


# def dataArrayFromDicomSingleFile(PathDicom):

# 	lstFilesDCM = []
# 	for dirName, subdirList, fileList in os.walk(PathDicom):
# 		for filename in fileList:
# 			# if ".dcm" in filename.lower():  # check whether the file's DICOM
# 			if filename[:2] in ['im', 'IM']:
# 				lstFilesDCM.append(os.path.join(dirName,filename))

# 	f = pydicom.read_file(lstFilesDCM[0])

# 	for j, de in enumerate(f):
# 		if j == 218:
# 			t = de.value
# 		if j == 219:
# 			z = de.value

# 	str_f = str(f)
# 	res = str_f[str_f.index('Pixel Spacing'):].split('\n')[0].split('[')[-1][:-1]
# 	res = res.replace("'","").split(',')
# 	res = [float(x) for x in res]

# 	data = f.pixel_array.reshape((z,t,f.pixel_array.shape[-2],f.pixel_array.shape[-1]))
# 	data = np.moveaxis(data,0,1)

# 	return data, [0, res[0], res[1]], None



# def dataArrayFromDicomFolder(PathDicom):

# 	lstFilesDCM = []
# 	for dirName, subdirList, fileList in os.walk(PathDicom):
# 		for filename in fileList:
# 			#should check whether the file is a DICOM
# 			lstFilesDCM.append(os.path.join(dirName,filename))
# 	lstFilesDCM = sorted(lstFilesDCM)
				
# 	# Get ref file
# 	RefDs = pydicom.read_file(lstFilesDCM[0], force=True)

# 	dicom_dir_details = {
# 		'SliceLocation':RefDs.get('SliceLocation', '?'),
# 		'InstanceNumber':RefDs.InstanceNumber,
# 		'ImageSize':RefDs.pixel_array.shape,
# 		'ImagePosition':RefDs.get('ImagePositionPatient', '?'),
# 		'ImageOrientation':RefDs.get('ImageOrientationPatient', '?'),
# 		'PatientPosition':RefDs.get('PatientPosition', '?'),
# 		'PixelSpacing':RefDs.get('PixelSpacing', '?'),
# 	}

# 	# Load spacing values (in mm)
# 	ConstPixelSpacing = (float(RefDs.SliceThickness), float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]))

# 	# loop through all the DICOM files and build lists of spatial and temporal positions:
# 	slice_locations, trigger_times = [], []
# 	for filenameDCM in lstFilesDCM:
# 		ds = pydicom.read_file(filenameDCM)
# 		location, t_time = ds.get('SliceLocation', '?'), ds.get('TriggerTime', '?')
# 		if location != '?' and t_time != '?':
# 			slice_locations.append( location )
# 			trigger_times.append( t_time )

# 	slice_locations= list(set(slice_locations))
# 	slice_locations.sort()
# 	trigger_times= list(set(trigger_times))
# 	trigger_times.sort()

# 	# work out data size from unique spatial and temporal positions:
# 	data = np.zeros( (len(trigger_times), len(slice_locations), int(RefDs.Rows), int(RefDs.Columns)), dtype=RefDs.pixel_array.dtype)
# 	placment = np.zeros((len(trigger_times), len(slice_locations)))
# 	image_ids = np.zeros((len(trigger_times), len(slice_locations)))

# 	# load images into 4D data array:
# 	image_positions = [None for i in range(len(slice_locations))]
# 	for i, filenameDCM in enumerate(lstFilesDCM):
# 		ds = pydicom.read_file(filenameDCM)
# 		location, t_time = ds.get('SliceLocation', '?'), ds.get('TriggerTime', '?')
# 		if location != '?' and t_time != '?':
# 			z = slice_locations.index(location)
# 			t = trigger_times.index(t_time)
# 			data[t,z] = ds.pixel_array
# 			placment[t,z] = 1
# 			image_ids[t,z] = i
# 			image_positions[z] = ds.get('ImagePositionPatient', '?')


# 	#squash to solve small variations in trigger_time:
# 	i = 0
# 	while i < data.shape[0]-1:
# 		if np.max((placment[i]>0)*1 + (placment[i+1]>0)*1) <= 1:
# 			data = np.concatenate([data[:i], data[i+1:i+2]+data[i:i+1], data[i+2:]], axis=0)
# 			placment = np.concatenate([placment[:i], placment[i+1:i+2]+placment[i:i+1], placment[i+2:]], axis=0)
# 			image_ids = np.concatenate([image_ids[:i], image_ids[i+1:i+2]+image_ids[i:i+1], image_ids[i+2:]], axis=0)
# 		else:
# 			i += 1

# 	return data, ConstPixelSpacing, image_ids, dicom_dir_details, slice_locations, trigger_times, image_positions


# import re 

# def sorted_nicely( l ): 
# 	""" Sort the given iterable in the way that humans expect.""" 
# 	convert = lambda text: int(text) if text.isdigit() else text 
# 	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
# 	return sorted(l, key = alphanum_key)

def planeToGrid(img_size, position=np.array([0,0,0]), orientation=[np.array([1,0,0,1,0,0])], pixel_spacing=[1,1]):
	'''creates a pyvista StructuredGrid to match a given DICOM slice'''

	Xxyz = orientation[3:]
	Yxyz = orientation[:3]

	Sx,Sy,Sz = position

	Xx,Xy,Xz = Xxyz
	Yx,Yy,Yz = Yxyz
	Di,Dj = pixel_spacing[-2:]

	M = np.array([
		[Xx*Di, Yx*Dj, 0, Sx],
		[Xy*Di, Yy*Dj, 0, Sy],
		[Xz*Di, Yz*Dj, 0, Sz],
		[0,     0,     0, 1],
	])

	xv, yv = np.meshgrid( np.linspace(0, img_size[0], img_size[0]+1), np.linspace(0, img_size[1], img_size[1]+1) )
	pts = np.concatenate([xv.reshape((1,-1)), yv.reshape((1,-1)), xv.reshape((1,-1))*0, xv.reshape((1,-1))*0+1], axis=0)

	X,Y,Z = np.dot(M,pts)[:3]
	X = X.reshape((img_size[0]+1, img_size[1]+1),order="F").T
	Y = Y.reshape((img_size[0]+1, img_size[1]+1),order="F").T
	Z = Z.reshape((img_size[0]+1, img_size[1]+1),order="F").T

	grid = pv.StructuredGrid(X,Y,Z)

	return grid

def planeToXYZ(img_size, position=np.array([0,0,0]), orientation=[np.array([1,0,0,1,0,0])], slice_location=0, pixel_spacing=[1,1]):

	Xxyz = orientation[3:]
	Yxyz = orientation[:3]

	if len(pixel_spacing) == 2:
		inter_plane_distance = 1
	else:
		inter_plane_distance = pixel_spacing[0]

	plane_normal = np.cross(Xxyz, Yxyz)

	Sx,Sy,Sz = position

	Xx,Xy,Xz = Xxyz
	Yx,Yy,Yz = Yxyz
	Di,Dj = pixel_spacing[-2:]

	M = np.array([
		[Xx*Di, Yx*Dj, 0, Sx],
		[Xy*Di, Yy*Dj, 0, Sy],
		[Xz*Di, Yz*Dj, 0, Sz],
		[0,     0,     0, 1],
	])

	# print(M)

	xv, yv = np.meshgrid( np.linspace(0, img_size[1], img_size[1]), np.linspace(0, img_size[0], img_size[0]) )

	pts = np.concatenate([xv.reshape((1,-1)), yv.reshape((1,-1)), xv.reshape((1,-1))*0, xv.reshape((1,-1))*0+1], axis=0)

	X,Y,Z = np.dot(M,pts)[:3]

	X = X.reshape(img_size)
	Y = Y.reshape(img_size)
	Z = Z.reshape(img_size)

	# print(X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max())

	return X, Y, Z

def planeToXYZ_original(img_size, position=np.array([0,0,0]), orientation=[np.array([1,0,0,1,0,0])], slice_location=0, pixel_spacing=[1,1]):

	slice_location = 0
	# orientation = np.array([1,2,3,4,5,6]).reshape((2,3))
	# print(orientation)
	# sys.exit()

	Xxyz = orientation[3:]
	Yxyz = orientation[:3]

	# print(img_size, position, Xxyz, Yxyz, pixel_spacing)
	# sys.exit()

	mode = 'new'

	if not (isinstance(slice_location, int) or isinstance(slice_location, float)):
		print('setting slice location to 0 as a non-numerical slice location was given:',slice_location)
		slice_location = 0


	if mode == 'new':

		# print(pixel_spacing)

		if len(pixel_spacing) == 2:
			inter_plane_distance = 1
		else:
			inter_plane_distance = pixel_spacing[0]

		plane_normal = np.cross(Xxyz, Yxyz)
		# print('plane normal length =', np.sum(plane_normal**2)**0.5)

		# plane_normal = -np.array([0,0,1])

		# print('slice location =', slice_location)
		if isinstance(slice_location, int):
			slice_location = slice_location * inter_plane_distance
		# print('adding', plane_normal * slice_location)
		# print('to', position)

		Sx,Sy,Sz = position + plane_normal * slice_location

		# print('resulting in', Sx,Sy,Sz)

		Xx,Xy,Xz = Xxyz
		Yx,Yy,Yz = Yxyz
		Di,Dj = pixel_spacing[-2:]

		M = np.array([
			[Xx*Di, Yx*Dj, 0, Sx],
			[Xy*Di, Yy*Dj, 0, Sy],
			[Xz*Di, Yz*Dj, 0, Sz],
			[0,     0,     0, 1],
		])

		xv, yv = np.meshgrid( np.linspace(0, img_size[1], img_size[1]), np.linspace(0, img_size[0], img_size[0]) )

		pts = np.concatenate([xv.reshape((1,-1)), yv.reshape((1,-1)), xv.reshape((1,-1))*0, xv.reshape((1,-1))*0+1], axis=0)

		# print(np.dot(M,pts)[3,:5])

		X,Y,Z = np.dot(M,pts)[:3]

		X = X.reshape(img_size)
		Y = Y.reshape(img_size)
		Z = Z.reshape(img_size)

	else:

		v1 = orientation[0]
		v2 = orientation[1]

		v3 = np.cross(v1,v2)
		this_position = position #+ slice_location*v3

		Di,Dj = pixel_spacing[-2:]

		v1 = v1*np.linspace(0, 1, img_size[0])[:,None] * img_size[0] * Di
		v2 = v2*np.linspace(0, 1, img_size[1])[:,None] * img_size[1] * Dj

		X = np.add.outer(v1[:,0], v2[:,0]) + this_position[0]
		Y = np.add.outer(v1[:,1], v2[:,1]) + this_position[1]
		Z = np.add.outer(v1[:,2], v2[:,2]) + this_position[2]

	return X, Y, Z

def get_the_slice(x,y,z, surfacecolor):
	return go.Surface(x=x,y=y,z=z, surfacecolor=surfacecolor, showscale=False, colorscale=[[0,'rgb(0,0,0)'], [1,'rgb(255,255,255)']])#, opacityscale=[[0,0],[1,1]])


if __name__ == '__main__':

	from scipy.ndimage import zoom
	from imageio import imwrite as imsave
	import sys

	import plotly.graph_objects as go
	# import plotly.express as go

	import plotly.io as pio
	pio.renderers.default = "png"


	base_dir = "DICOM/CorFlow10/"
	# base_dir = "P14_T03_con/"

	all_data, all_segs, slice_obs = [], [], []

	ordered_folders = sorted_nicely(os.listdir(base_dir))
	ordered_folders = [x for x in ordered_folders if '.DS' not in x]
	ordered_folders = [x for x in ordered_folders if 'FUNCTION' in x]

	print(ordered_folders)

	for folder in ordered_folders:

		print( '\nprocessing dicom:', os.path.join(base_dir, folder))

		#load DICOM as 4D numpy array with metadata
		data, pixel_spacing, image_ids, dicom_details, slice_locations, trigger_times, image_positions = dataArrayFromDicom(os.path.join(base_dir, folder))
		print('original data shape:', data.shape)

		sys.exit()
		# position = np.array(list(dicom_details['ImagePosition']))
		orientation = np.array(list(dicom_details['ImageOrientation']))
		# img_size = list(dicom_details['ImageSize'])
		# slice_location = dicom_details['SliceLocation']
		
		# normalize image:
		data = data - data.min()
		data = np.clip(data, 0, np.percentile(data, 99.5))
		data = data / data.max()

		t = 4

		print(len(slice_locations), data.shape, list(range(0,data.shape[1], np.max([1,data.shape[1]-1]))))

		if data.shape[1] <= 3:
			slcs = [data.shape[1]//2]
		else:
			slcs = [0,data.shape[1]//2,data.shape[1]-1]
		for slice_index in slcs:

			X, Y, Z = planeToXYZ(data.shape[2:], image_positions[slice_index], orientation, slice_locations[slice_index], pixel_spacing)

			slice_obs.append( get_the_slice(X,Y,Z,data[t,slice_index].T) )

	fig1 = go.Figure(data=slice_obs)
	fig1.show()

	print('still going')

	fig1.update_layout(
		autosize=False,
		width=700,
		height=700,
		margin=dict(
			l=10,
			r=10,
			b=10,
			t=10,
			pad=4
		),
		scene = dict(
			xaxis = dict(
				showbackground=False,
				zerolinecolor="white",
			),
			yaxis = dict(
				showbackground=False,
				zerolinecolor="white"
			),
			zaxis = dict(
				showbackground=True,
				zerolinecolor="white",
				showticklabels=False,
				visible=False
			),
		),
	)


	images = []
	steps = 3
	for i, theta in enumerate(np.linspace(np.pi/4, 2.25*np.pi*steps/(steps+1), steps)):

		print('frame %d of %d' % (i,steps))

		camera = dict(
			up=dict(x=0, y=0, z=1),
			center=dict(x=0, y=0, z=0),
			eye=dict(x=1.77*np.sin(theta), y=1.77*np.cos(theta), z=1.25)
		)
			
		fig1.update_layout(
			scene_camera=camera
		)

		print('writing...')
		fig1.write_image("tmp0.png")

		images.append(imageio.imread('tmp0.png'))
	imageio.mimsave('movie.gif', images)
