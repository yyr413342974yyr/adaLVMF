import numpy as np 
from matplotlib import pyplot as plt

from scipy.ndimage import label, binary_dilation
from collections import Counter

from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology.convex_hull import convex_hull_image

from scipy.ndimage.morphology import distance_transform_edt

def simpleShapeCorrection(msk):

	print(msk.shape, msk.min(), msk.max())

	#slicewise:
	for i in range(msk.shape[0]):
		for j in range(msk.shape[1]):

			### keep only largest cc for myo and LV-BP channels (i.e. green and blue):
			lvmyo = msk[i,j]==2
			labels, count = label(lvmyo) # value of 2 = green = LV-myo
			if count != 0:
				largest_cc = np.argmax( [np.sum(labels==k) for k in range(1, count + 1)] ) + 1
				msk[i,j] -= (1-(labels==largest_cc))*(lvmyo)*2

			lvbp = msk[i,j]==3
			labels, count = label(lvbp) # value of 3 = blue = LV-BP
			if count != 0:
				largest_cc = np.argmax( [np.sum(labels==k) for k in range(1, count + 1)] ) + 1
				msk[i,j] -= (1-(labels==largest_cc))*(lvbp)*2

			### remove any components smaller than 20pixels from the RV-BP channel:
			rvbp = msk[i,j]==1
			labels, count = label(rvbp) # value of 3 = blue = RV-BP
			for idx in range(1, count + 1):
				cc = labels==idx
				if np.sum(cc) < 20:
					msk[i,j] *= (1-cc)

	return msk

def impute(arr):
	# https://stackoverflow.com/questions/41550979/fill-holes-with-majority-of-surrounding-values-python

    imputed_array = np.copy(arr)

    mask = np.isnan(arr)
    labels, count = label(mask)
    for idx in range(1, count + 1):
        hole = labels == idx
        surrounding_values = arr[binary_dilation(hole) & ~hole]
        most_frequent = Counter(surrounding_values).most_common(1)[0][0]
        imputed_array[hole] = most_frequent

    return imputed_array


def fillClosedHoles(msk):
	#msk = Width x Height x Channels

	#find the holes:
	binark_mask = np.sum(msk, axis=-1).astype('int')
	filled_binark_mask = binary_fill_holes(binark_mask)
	holes = filled_binark_mask - binark_mask

	#for each hole:
	labels, count = label(holes)

	for i in range(1,labels.max()+1):
		label_i = np.where(labels==i,1,0)
		chan = np.argmax(np.sum( (binary_dilation(label_i) - label_i)[...,None]*msk, axis=(0,1) ) )

		msk[...,chan] += label_i

	return msk

def createDistanceGrid(msk):

	if len(msk.shape) == 3:
		binary_msk = np.sum(msk, axis=-1)
	elif len(msk.shape) == 2:
		binary_msk = msk+0
	else:
		print('error, passed mask should be HxWxC or HxW, recieved input with shape:', msk.shape)
		sys.exit()

	labels, count = label(binary_msk)
	if labels.max() == 1:
		return msk

	no_change = False
	while no_change == False:

		labels, count = label(binary_msk)

		cc_distances = np.zeros((labels.max(), labels.max()))
		cc_sizes = np.zeros((labels.max(),))

		final_mask = np.zeros(binary_msk.shape)

		for i in range(1,labels.max()+1):
			label_i = np.where(labels==i,1,0)
			dst = distance_transform_edt(1-label_i)
			cc_sizes[i-1] = np.sum(label_i)
			for j in range(1,labels.max()+1):
				res = np.where(labels==j, dst, np.inf)
				cc_distances[i-1][j-1] = np.min(res)
			cc_distances[i-1][i-1] = np.inf

			if cc_sizes[i-1] >= (np.min(cc_distances[i-1]))**2:
				final_mask += label_i

		if np.allclose(final_mask,binary_msk):
			no_change = True
		else:
			binary_msk = final_mask



	# plt.subplot(1,2,1)
	# plt.imshow(binary_msk)
	# plt.subplot(1,2,2)
	# plt.imshow(final_mask)
	# plt.show()

	# print(cc_distances)
	# print(cc_sizes)

	return msk * final_mask[...,None]


def unifyPeices(msk):

	arr = []
	for i in range(msk.shape[-1]):
		# plt.imshow(distance_transform_edt(1-msk[...,i]))
		# plt.show()
		# arr.append( distance_transform_edt(msk[...,i])[...,None] )
		arr.append( convex_hull_image(msk[...,i]).astype('int')[...,None] )

	arr = np.concatenate(arr,-1)

	arr[...,0] = arr[...,0] - (arr[...,0]*arr[...,1])
	arr[...,1] = arr[...,1] - arr[...,2]

	msk[...,0] += (1-msk[...,0])*arr[...,0]

	return arr*1.

def correctMask4D(mask):

	if len(mask.shape) == 4 or (len(mask.shape) == 5 and mask.shape[-1] == 1 ):
		multi_channel = False
		original_shape_len = len(mask.shape)
		#convert to multi-channel:
		if original_shape_len == 4:
			mask_mc = np.zeros(mask.shape+(3,))
		else:
			mask_mc = np.zeros(mask.shape[:-1]+(3,))

		mask_mc[mask==1,0]=1
		mask_mc[mask==2,1]=1
		mask_mc[mask==3,2]=1
		mask = mask_mc
	elif len(mask.shape) == 5 and mask.shape[-1] == 3:
		multi_channel = True
	else:
		print('unexpected mask input shape:', mask.shape)
		return mask

	for i in range(mask.shape[0]):
		mask[i] = correctMask(mask[i])

	if multi_channel == False:
		#if original mask wasn't multi channel, convert back to that format
		if original_shape_len == 5:
			keepdims=True
		else:
			keepdims=False
		mask = np.sum(mask*np.array([[[[[1,2,3]]]]]), axis=-1, keepdims=keepdims)

	return mask

def correctMask(mask):

	# fill any closed holes:
	for i,m in enumerate(mask):
		m = fillClosedHoles(m)
		mask[i] = m

	# keep largest connected component (in 3D)
	for i in range(3):

		labels, count = label(mask[...,i])

		#todo: make sure it is across multiple layers
		counts = np.bincount(labels.flatten())
		counts[0] = 0
		largest_component = counts.argmax()
		mask[...,i] *= (labels == largest_component)

	# fill any closed holes (again):
	for i,m in enumerate(mask):
		m = fillClosedHoles(m)
		mask[i] = m

	# crop LVBP near apex:
	# slices = mask.shape[0]
	# for i in range(2*slices//3, slices):
	# 	mask[i,...,0] = mask[i,...,0] * np.sum(mask[i-1,...,:3], axis=-1)


	return mask


if __name__ == '__main__':

	import os
	from imageio import imwrite as imsave

	#load a random time slice of a random volume:
	cohort = np.random.choice(['ACDC','MMWHS','KAGGLE'])
	basedir = './data/%s/' % (cohort,)
	folder = np.random.choice(os.listdir(basedir))
	print( os.path.join(basedir,folder) )

	image = np.load( os.path.join(basedir,folder,'image.npy') )
	mask = np.load( os.path.join(basedir,folder,'label.npy') )

	if len(image.shape) == 4:
		image = image[...,None]

	#convert mask to multi-channel:
	mask_mc = np.zeros(mask.shape+(3,))
	mask_mc[mask==1,0]=1
	mask_mc[mask==2,1]=1
	mask_mc[mask==3,2]=1
	mask = mask_mc

	# random_index = np.random.randint(image.shape[0])

	for i in range(image.shape[0]):
		img = image[i]
		msk = mask[i]

		print(img.shape,msk.shape)

		pre_overlay = np.clip(msk*255*0.5 + img*255, 0, 255).astype('uint8')
		pre_overlay = np.concatenate(pre_overlay, axis=1)

		msk = correctMask(msk)

		post_overlay = np.clip(msk*255*0.5 + img*255, 0, 255).astype('uint8')
		post_overlay = np.concatenate(post_overlay, axis=1)
		imsave("pre_and_post_overlay_%d.png" % (i,), np.concatenate([pre_overlay, post_overlay], axis=0))


	sys.exit()


	masks = np.load('/Users/tom/Desktop/academic/data/ACDC/ACDC_128x128_rotated_labels.npy')
	masks = np.concatenate(masks, axis=0)

	# plt.imshow(masks[23])
	# plt.show()

	mask_with_holes = masks[np.random.randint(masks.shape[0])]

	for i in range(10):

		x = np.random.randint(0, mask_with_holes.shape[0])
		y = np.random.randint(0, mask_with_holes.shape[1])

		sz = np.random.randint(1,20)

		mask_with_holes[x:x+sz,y:y+sz] = 0
		mask_with_holes[x:x+sz,y:y+sz,np.random.randint(0,3)] = 1

	for i in range(50):

		x = np.random.randint(0, mask_with_holes.shape[0])
		y = np.random.randint(0, mask_with_holes.shape[1])

		mask_with_holes[x:x+5,y:y+5] = 0

	for i in range(5):

		x = np.random.randint(0, mask_with_holes.shape[0])
		y = np.random.randint(0, mask_with_holes.shape[1])
		xw = np.random.randint(1,8)
		yw = np.random.randint(1,8)
		mask_with_holes[x:x+xw,:] = 0
		mask_with_holes[:,y:y+yw] = 0

	plt.subplot(1,4,1)
	plt.imshow(mask_with_holes)

	mask_with_holes = fillClosedHoles(mask_with_holes)

	plt.subplot(1,4,2)
	plt.imshow(mask_with_holes)

	mask_with_holes = np.concatenate([createDistanceGrid(mask_with_holes[...,0:1]),createDistanceGrid(mask_with_holes[...,1:2]),createDistanceGrid(mask_with_holes[...,2:3])], axis=-1)

	plt.subplot(1,4,3)
	plt.imshow(mask_with_holes)

	# plt.subplot(1,2,1)
	# plt.imshow(mask_with_holes)

	mask_with_holes = unifyPeices(mask_with_holes)

	plt.subplot(1,4,4)
	plt.imshow(mask_with_holes)
	plt.show()

	# plt.subplot(1,2,2)
	# plt.imshow(mask_with_holes)
	# plt.show()




