# LV Mesh Fitting

This repo contains the code for segmenting CMR image data and fitting a left ventricle mesh model to it (it also contains the LV mesh model itself). It can work on mixtures of both SAX and LAX slices. The code supports the publication on Medical Image Analysis. If you use any part of this reposotory, please cite: 

Thomas Joyce, Stefano Buoso, Christian T. Stoeck, Sebastian Kozerke, Rapid inference of personalised left-ventricular meshes by deformation-based differentiable mesh voxelization, Medical Image Analysis, Volume 79, 2022, 102445, https://doi.org/10.1016/j.media.2022.102445.

## Installation
You can create a conda environment as

		conda create --name lvmeshfitting -c numpy matplotlib scipy vtk ipython pytorch torchvision 
		conda activate meshFromMask_env 
		conda install -c pytorch torchvision 
		conda install -c conda-forge pydicom meshio pyvista tqdm 
		conda install -c menpo imageio 
		conda install scikit-image tensorflow keras 

## Data structure
Each CMR exam should be placed in a folder, and all exams for the same patient should be in the same patient folder identified with 'input path'.

## Segmentation

		de = DicomExam(input_path) #create a dicom exam object, by passing the dicom folder path
		de.standardiseTimeframes() #re-sample data in the time-dimension so that all series have the same number of time steps
		de.segment(use_tta=tta) #segment all the data (possibly using test-time augmentation)
		de.save()
		de.proxyCardiacIndecies('network') #produces a plot of various cardiac indicies (saved as proxy_strains_from_network_predictions.png in the plots folder)
		de.summary() #prints a summary of the data, including its shape after resampling and cropping for segmentation
		de.calculateSegmentationUncertainty()
		de.estimateValvePlanePosition() #heuristically remove predicted mask on atrial slices (eroneously predicted by the segmentation algorithm)
		de.estimateLandmarks()
		de.saveImages(show_landmarks=True) #save a visualisation of the data (+ segmentations if they exist)
		de.save()

## Mesh model fitting

		de = loadDicomExam(input_path)
		de.resetMeshFitting()
		de.summary()
		de.fitMesh(training_steps=50, time_frames_to_fit='all_loop', save_training_progress_gif=False, 
		mode_loss_weight = 0.05, global_shift_penalty_weigth = 0.3, lr = 0.001) #fits a mesh to every time frame. Check the function definition for a list of its arguments
		de.save()
		de.proxyCardiacIndecies('mesh') #produces a plot of various cardiac indicies calculated using the rendered and sliced mesh
		de.calculateSegmentationUncertainty('mesh')
		de.saveImages(use_mesh_images=True)

Available flags are

		'training_steps' = 50. number of iterations for the fitting of a single time step 
		'time_frames_to_fit' = 'all_loop'. Fit all cardiac phases and force first and last to be close 
		'mode_loss_weight' = 0.05. Weight for the loss function related to the shape model constraint
		'global_shift_penalty_weigth' = 0.3. Weight for the slice shifting loss function
		'lr' = 0.001. Learning rate value

