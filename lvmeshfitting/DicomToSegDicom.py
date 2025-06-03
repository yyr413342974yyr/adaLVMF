import pydicom
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import imageio
from scipy.ndimage import zoom
from imageio import imwrite as imsave
from shape_correction import correctMask
import sys
from models import makeTernausNet16
from scipy.ndimage.measurements import center_of_mass

import keras.backend as K

# from utilsForDicom import *

from TestTimeAugmentation import predictTTA



def saveImage(fname, arr):

	if arr.shape[0] > 15:
		skip = int(np.ceil(arr.shape[0]/15))
	else:
		skip = 1
	i = 0
	mask_img = np.concatenate(arr[::skip]*255, axis=1).astype('uint8')
	imageio.imwrite(fname, mask_img)

def hardSoftMax(pred, threshold=0.3):
	
	content = np.sum(pred, axis=-1) > threshold
	pred_class = np.argmax(pred, axis=-1)
	pred *= 0
	for i in range(3):
		pred[pred_class==i,i]=1
	pred *= content[...,None]
	return pred

def padding(dims, pad_dim, pad_vals):
	pad = [(0,0) for i in range(dims)]
	pad[pad_dim] = pad_vals
	return pad


def getImageAt(c1,c2,data,sz=256):
	#extract a square image of size sz with the top left corner at the given point, cropping and padding as required
	return np.pad(data+0,((0,0),(0,0),(sz//2,sz//2),(sz//2,sz//2)))[:,:,c1:c1+sz,c2:c2+sz]

def getSeg(data, m, sz=256, use_tta=False):
	# iterativley search for the 256x256 window centered on the LV

	_,_,c1,c2 = data.shape
	c1,c2 = c1//2,c2//2
	center_moved = True

	all_c1c2 = [(c1,c2)]

	center_moved_counter = -1
	while center_moved:
		center_moved_counter += 1
		center_moved = False
		roi = getImageAt(c1,c2,data).reshape((-1,sz,sz,1))

		pred = m.predict(roi)
		
		pred = hardSoftMax(pred)

		# imageio.imwrite('center_search_debug_%d.png' % (center_moved_counter,), np.mean(pred, axis=0))

		new_c1, new_c2 = center_of_mass(np.mean(pred, axis=0)[...,2])

		if np.isnan(new_c1) or np.isnan(new_c2):
			print(new_c1, new_c2)
			print('nothing found in center of image(s) for this series, aborting. Exclude the series or shift it to center on the heart')
			sys.exit()

		new_c1, new_c2 = int(np.round(new_c1)), int(np.round(new_c2))
		
		new_c1 = c1 + new_c1-sz//2
		new_c2 = c2 + new_c2-sz//2
		# print("%d --> %d, %d --> %d" %(c1,new_c1,c2,new_c2))
		if np.abs(c1 - new_c1) > 2 or np.abs(c2 - new_c2) > 2:
			center_moved = True
			c1 = new_c1
			c2 = new_c2

			if (c1,c2) in all_c1c2:
				#stuck in a loop
				all_c1c2 = all_c1c2[all_c1c2.index((c1,c2)):]
				print('averaging points in loop:')
				print(all_c1c2)
				c1, c2 = np.mean(all_c1c2, axis=0).astype('int')
				print(c1,c2)
				break

			all_c1c2.append((c1,c2))

	#now we have found the ROI location, segment with test-time averaging if required:
	if use_tta:
		roi = getImageAt(c1,c2,data).reshape((-1,sz,sz,1))

		old_stdout = sys.stdout # backup current stdout
		sys.stdout = open(os.devnull, "w")
		all_preds = predictTTA(m, roi)
		sys.stdout = old_stdout # reset old stdout

		all_preds = np.stack(all_preds)
		pred = np.mean(all_preds, axis=0)
		pred = hardSoftMax(pred)

	pred = np.pad(pred,((0,0),(c1,data.shape[2]-c1),(c2,data.shape[3]-c2),(0,0)))
	pred = pred[:,sz//2:-sz//2,sz//2:-sz//2]
	pred = pred.reshape(data.shape+(3,))

	if use_tta:
		# print(sz, all_preds.shape)
		all_preds = np.pad(all_preds,((0,0),(0,0),(c1,data.shape[2]-c1),(c2,data.shape[3]-c2),(0,0)))
		all_preds = all_preds[:,:,sz//2:-sz//2,sz//2:-sz//2]
		all_preds = all_preds.reshape((all_preds.shape[0],)+data.shape+(3,))
		# print(sz, all_preds.shape, data.shape)
	else:
		all_preds = None

	# print('getSeg (c1, c2) =', c1, c2)

	return pred, c1, c2, all_preds

def setShape(data, szs, center_point=None):

	data += 0

	if data.shape[2] > szs[2]:
		to_trim = data.shape[2] - szs[2]
		trim_a = to_trim//2
		trim_b = to_trim-trim_a
		data = data[:,:,trim_a:-trim_b]
	elif data.shape[2] < szs[2]:
		to_add = szs[2] - data.shape[2]
		add_a = to_add//2
		add_b = to_add - add_a
		p = padding(len(data.shape), 2, (add_a,add_b))
		data = np.pad(data,p)

	if data.shape[3] > szs[3]:
		to_trim = data.shape[3] - szs[3]
		trim_a = to_trim//2
		trim_b = to_trim-trim_a
		data = data[:,:,:,trim_a:-trim_b]
	elif data.shape[3] < szs[3]:
		to_add = szs[3] - data.shape[3]
		add_a = to_add//2
		add_b = to_add - add_a
		p = padding(len(data.shape), 3, (add_a,add_b))
		data = np.pad(data,p)

	return data

def produceSegAtRequiredRes(data, pixel_spacing, is_sax=True, use_tta=False):

	dicom_data_shape = data.shape

	#zoom to the correct resolution (i.e. 1mmx1mm):
	data = zoom(data,(1, 1, pixel_spacing[1], pixel_spacing[2]), order=1)

	# normalize intensities:
	data = data - data.min()
	data = np.clip(data, 0, np.percentile(data, 99.5))
	data = data / data.max()

	#load the segmentation network:
	model = makeTernausNet16((256,256,1))
	if is_sax:
		model.load_weights('SegmentationModels/my_model.h5')
	else:
		model.load_weights('SegmentationModels/my_LAX_model.h5')

	#segment image, finding 256x256 window centered on LV
	pred, c1, c2, all_preds = getSeg(data, model, use_tta=use_tta) #returns the segmentation and the pixel co-orinates for the center of the LV

	pred = np.sum(pred*[[[[[1,2,3]]]]], axis=-1)

	# if all_preds is not None:
	# 	all_preds = np.sum(all_preds*[[[[[1,2,3]]]]], axis=-1)

	K.clear_session() # delete the model

	return data, pred, c1, c2, all_preds

	'''
	#make mask prediction:
	original_shape = data.shape
	data = setShape(data, (0,0,256,256))
	# print('data shape ready for prediction (a)', data.shape)
	prepped_shape = data.shape
	data = np.reshape(data,(-1,256,256,1))
	# print('data shape ready for prediction (b)', data.shape)

	if is_sax:
		# pred = predictTTA(model, data)#[:,7:-7,7:-7])
		model = makeTernausNet16((256,256,1))
		model.load_weights('/home/tom/Desktop/projects/iterativeSegmentation/my_model.h5')
		pred = model.predict(data)
		print('SAX pred min max', pred.min(), pred.max())
		# pred = np.mean(pred, axis=0)
	else:
		# pred = predictTTA(LAmodel, data)#[:,7:-7,7:-7])
		LAmodel = makeTernausNet16((256,256,1))
		LAmodel.load_weights('/home/tom/Desktop/projects/iterativeSegmentation/my_LAX_model.h5')
		pred = LAmodel.predict(data)
		print('LAX pred min max', pred.min(), pred.max())
		# pred = np.mean(pred, axis=0)

	pred = hardSoftMax(pred)
	# pred = correctMask(pred)

	print('pred shape after prediction (b)', pred.shape)
	'''

	# data = np.reshape(data, prepped_shape)
	# pred = np.reshape(pred, prepped_shape+(3,))

	# print('pred shape after prediction (a)', pred.shape)

	# data = setShape(data, original_shape)
	# pred = setShape(pred, original_shape)

	# pred = np.sum(pred*[[[[[1,2,3]]]]], axis=-1)

	# print('pred shape at target resolution', pred.shape)

	# return data, pred

	# pred_os = np.sum(pred*[[[[[1,2,3]]]]], axis=-1)
	# pred_os = np.pad(pred_os, ((0,0),(0,0),(w//2-sz//2,w-(w//2+sz//2)),(h//2-sz//2,h-(h//2+sz//2))), 'constant', constant_values=0)
	# pred_os = pred_os[:, :, sz:-sz, sz:-sz]
	# pred_os = zoom(pred_os,(1, 1, dicom_data_shape[2]/pred_os.shape[2], dicom_data_shape[3]/pred_os.shape[3]), order=1)
	# pred_os = np.round(np.clip(pred_os,0,3)).astype('uint8')

	# return pred, data, pred_os



# def produceSegmentations(data, pixel_spacing, is_sax=True):

# 	print('produceSegmentations -->',data.shape)

# 	dicom_data_shape = data.shape

# 	#zoom to the correct resolution (i.e. 1mmx1mm):
# 	data = zoom(data,(1, 1, pixel_spacing[1], pixel_spacing[2]), order=1)
# 	print('data shape after zoom:', data.shape)

# 	# pad or crop to a 256x256px in-plane image:
# 	sz = 256
# 	data = np.pad(data, ((0,0),(0,0),(sz,sz),(sz,sz)), 'constant', constant_values=data.min())
# 	w,h = data.shape[2:4]
# 	data = data[:, :, w//2-sz//2:w//2+sz//2, h//2-sz//2:h//2+sz//2]
# 	print('data shape after crop/pad:', data.shape)

# 	# normalize:
# 	data = data - data.min()
# 	data = np.clip(data, 0, np.percentile(data, 99.5))
# 	data = data / data.max()

# 	# normalize per slice:
# 	# print(data.dtype)
# 	# for i in range(data.shape[1]):
# 	# 	data[:,i] = data[:,i] - data[:,i].min()
# 	# 	# data[:,i] = np.clip(data[:,i], 0, np.percentile(data[:,i], 99.5))
# 	# 	data[:,i] = data[:,i] / data[:,i].max()

# 	#make mask prediction:
# 	original_shape = data.shape
# 	data = np.reshape(data,(-1,sz,sz,1))

# 	if is_sax: #'sax' in series_dir.lower() or 'sa' in series_dir.lower():
# 		# pred = predictTTA(model, data)#[:,7:-7,7:-7])
# 		model = makeTernausNet16((256,256,1))
# 		model.load_weights('/home/tom/Desktop/projects/iterativeSegmentation/my_model.h5')
# 		pred = model.predict(data)
# 		print('SAX pred min max', pred.min(), pred.max())
# 		# pred = np.mean(pred, axis=0)
# 	else:
# 		# pred = predictTTA(LAmodel, data)#[:,7:-7,7:-7])
# 		LAmodel = makeTernausNet16((256,256,1))
# 		LAmodel.load_weights('/home/tom/Desktop/projects/iterativeSegmentation/my_LAX_model.h5')
# 		pred = LAmodel.predict(data)
# 		print('LAX pred min max', pred.min(), pred.max())
# 		# pred = np.mean(pred, axis=0)

# 	# saveImage(results_directory+'/prepre.png', pred)

# 	pred = hardSoftMax(pred)

# 	# saveImage(results_directory+'/pre.png', pred)

# 	# pred = correctMask(pred)

# 	# saveImage(results_directory+'/post.png', pred)

# 	# pred = np.pad(pred, ((0,0),(7,7),(7,7),(0,0)), 'constant', constant_values=0)
# 	data = np.reshape(data, original_shape+(1,))
# 	pred = np.reshape(pred, original_shape+(3,))

# 	pred_os = np.sum(pred*[[[[[1,2,3]]]]], axis=-1)
# 	pred_os = np.pad(pred_os, ((0,0),(0,0),(w//2-sz//2,w-(w//2+sz//2)),(h//2-sz//2,h-(h//2+sz//2))), 'constant', constant_values=0)
# 	pred_os = pred_os[:, :, sz:-sz, sz:-sz]
# 	pred_os = zoom(pred_os,(1, 1, dicom_data_shape[2]/pred_os.shape[2], dicom_data_shape[3]/pred_os.shape[3]), order=1)
# 	pred_os = np.round(np.clip(pred_os,0,3)).astype('uint8')

# 	return pred, data, pred_os

# if __name__ == '__main__':
# 	#load network:
# 	model = makeTernausNet16((256,256,1))
# 	model.load_weights('my_model.h5')

# 	LAmodel = makeTernausNet16((256,256,1))
# 	LAmodel.load_weights('my_LAX_model.h5')


# 	all_data, all_segs, slice_obs = [], [], []

# 	base_dir = "../exampleDICOM"
# 	output_dir = "../exampleDICOM/results"

# 	ordered_datasets = sorted_nicely(os.listdir(base_dir))
# 	ordered_datasets = [x for x in ordered_datasets if '.DS' not in x]
# 	ordered_datasets = [x for x in ordered_datasets if 'results' not in x] #ignore results directory

# 	ordered_datasets = ['JS']

# 	for dataset_dir in ordered_datasets:

# 		ordered_patients = sorted_nicely(os.listdir( os.path.join(base_dir, dataset_dir)))
# 		ordered_patients = [x for x in ordered_patients if x[0] != '.']

# 		for patient_dir in ordered_patients:

# 			ordered_series = sorted_nicely(os.listdir( os.path.join(base_dir, dataset_dir, patient_dir)))
# 			ordered_series = [x for x in ordered_series if x[0] != '.']

# 			if dataset_dir == 'CW':
# 				ordered_series = [x for x in ordered_series if 'FUNCTION' in x]

# 			for series_dir in ordered_series:

# 				full_path = os.path.join(base_dir, dataset_dir, patient_dir, series_dir)

# 				#make output directory:
# 				results_directory = os.path.join(output_dir, dataset_dir, patient_dir, series_dir)
# 				if not os.path.exists(results_directory):
# 					os.makedirs(results_directory)

# 				print( '\nprocessing dicom:', full_path)

# 				#load DICOM as 4D numpy array with metadata
# 				data, pixel_spacing, image_ids, dicom_details, slice_locations, trigger_times, image_positions, is3D, multifile = dataArrayFromDicom(full_path)
# 				print('original data shape:', data.shape)
# 				print('pixel spacing =', pixel_spacing)

# 				is_sax = 'sa' in series_dir.lower()
# 				pred, data, pred_os = produceSegmentations(data, pixel_spacing, is_sax=is_sax)

# 				# is_sax = 'sa' in series_dir.lower()
# 				# data = np.moveaxis(data,1,3)
# 				# pred, data, pred_os = produceSegmentations(data.astype('float'), pixel_spacing, is_sax=False)

# 				# data = data[...,None]

# 				'''
# 				dicom_data_shape = data.shape

# 				#zoom to the correct size:
# 				data = zoom(data,(1, 1, pixel_spacing[1], pixel_spacing[2]), order=1)
# 				print('data shape after zoom:', data.shape)

# 				# pad or crop to a 270x270px in-plane image:
# 				sz = 256
# 				data = np.pad(data, ((0,0),(0,0),(sz,sz),(sz,sz)), 'constant', constant_values=data.min())
# 				w,h = data.shape[2:4]
# 				data = data[:, :, w//2-sz//2:w//2+sz//2, h//2-sz//2:h//2+sz//2]
# 				print('data shape after crop/pad:', data.shape)

# 				# normalize:
# 				data = data - data.min()
# 				data = np.clip(data, 0, np.percentile(data, 99.5))
# 				data = data / data.max()

# 				#make mask prediction:
# 				original_shape = data.shape
# 				data = np.reshape(data,(-1,sz,sz,1))
# 				if 'sax' in series_dir.lower() or 'sa' in series_dir.lower():
# 					pred = predictTTA(model, data)#[:,7:-7,7:-7])
# 					pred = np.mean(pred, axis=0)
# 				else:
# 					pred = predictTTA(LAmodel, data)#[:,7:-7,7:-7])
# 					pred = np.mean(pred, axis=0)

# 				saveImage(results_directory+'/prepre.png', pred)

# 				pred = hardSoftMax(pred)

# 				saveImage(results_directory+'/pre.png', pred)

# 				pred = correctMask(pred)

# 				saveImage(results_directory+'/post.png', pred)

# 				# pred = np.pad(pred, ((0,0),(7,7),(7,7),(0,0)), 'constant', constant_values=0)
# 				data = np.reshape(data, original_shape+(1,))
# 				pred = np.reshape(pred, original_shape+(3,))

# 				pred = np.round(pred)
# 				pred_os = np.clip(np.sum(pred*[[[[[1,2,3]]]]], axis=-1),0,3)
# 				pred_os = np.pad(pred_os, ((0,0),(0,0),(w//2-sz//2,w-(w//2+sz//2)),(h//2-sz//2,h-(h//2+sz//2))), 'constant', constant_values=0)
# 				pred_os = pred_os[:, :, sz:-sz, sz:-sz]
# 				pred_os = zoom(pred_os,(1, 1, dicom_data_shape[2]/pred_os.shape[2], dicom_data_shape[3]/pred_os.shape[3]), order=1)
# 				print('pred shape after reversal:', pred_os.shape)
# 				'''

# 				#create output directory (if required):
# 				makeSegDicom(full_path, pred_os, results_directory)

# 				#save a gif if there is multiple time frames:
# 				if pred.shape[0] > 1:
# 					frames = []
# 					for i in range(pred.shape[0]):
# 						pred[i] = correctMask(pred[i])
# 						overlay = np.clip(pred[i]*255*0.5 + data[i]*255, 0, 255).astype('uint8')
# 						overlay = np.concatenate(overlay, axis=1)
# 						data_img = np.concatenate(np.tile(data[i]*255,(1,1,3)), axis=1)
# 						mask_img = np.concatenate(pred[i]*255, axis=1)
# 						if pred.shape[1] == 1:
# 							frame = np.concatenate([data_img, overlay, mask_img], axis=1).astype('uint8')
# 						else:
# 							frame = np.concatenate([data_img, overlay, mask_img], axis=0)[::2,::2].astype('uint8')
# 							# frame = np.concatenate([data_img, overlay, mask_img], axis=0).astype('uint8')
# 						frames.append(frame)
# 					imageio.mimsave(results_directory+'/movie.gif', frames)

# 				else: #otherwise just save a png:

# 					print('saving a PNG')
# 					print(pred.shape)
# 					print(data.shape)

# 					if pred.shape[1] > 15:
# 						skip = int(np.ceil(pred.shape[1]/15))
# 						data = data[:,::skip]
# 						pred = pred[:,::skip]

# 					i = 0
# 					# pred[i] = correctMask(pred[i])
# 					overlay = np.clip(pred[i]*255*0.5 + data[i]*255, 0, 255).astype('uint8')
# 					overlay = np.concatenate(overlay, axis=1)
# 					data_img = np.concatenate(np.tile(data[i]*255,(1,1,3)), axis=1)
# 					mask_img = np.concatenate(pred[i]*255, axis=1)

# 					print(data_img.shape, overlay.shape, mask_img.shape)

# 					img = np.concatenate([data_img, overlay, mask_img], axis=0).astype('uint8')
# 					imageio.imwrite(results_directory+'/vis.png', img)



