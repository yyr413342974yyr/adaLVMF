import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm
import itertools

def predictTTA(model, data):

	res = []

	# zoom_sizes = [0.975, 1.0, 1.025]
	zoom_sizes = [1.0]
	upper_bounds = [100]#,99]
	flips = [1,-1]
	rotations = [0,1,2,3]

	total_batches = len(zoom_sizes) * len(upper_bounds) * len(flips) * len(rotations)

	sz = data.shape[1]

	print('TTA...')

	with tqdm(total=total_batches) as pbar:

		for zsize in zoom_sizes:

			batch = zoom(data,(1, zsize, zsize, 1), order=1)
			batch = np.pad(batch, ((0,0),(sz,sz),(sz,sz),(0,0)), 'constant', constant_values=batch.min())
			w,h = batch.shape[1:3]
			batch = batch[:, w//2-sz//2:w//2+sz//2, h//2-sz//2:h//2+sz//2]

			for ub in upper_bounds:
				batch = batch - batch.min()
				batch = np.clip(batch, 0, np.percentile(batch, ub))
				batch = batch / batch.max()

				for flip in flips:
					for rot in rotations:

						pred = np.rot90(model.predict(np.rot90(batch[:,::flip], rot, axes=(1,2))), (4-rot), axes=(1,2))[:,::flip]

						pred = np.pad(pred, ((0,0),(w//2-sz//2,w-(w//2+sz//2)),(h//2-sz//2,h-(h//2+sz//2)),(0,0)), 'constant', constant_values=0)
						pred = pred[:, sz:-sz, sz:-sz]
						pred = zoom(pred,(1, 1/zsize, 1/zsize, 1), order=1)
						res.append( pred )

						pbar.update(1)

	return res