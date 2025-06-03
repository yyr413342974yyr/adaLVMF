'''
This is intended to be an easily usable script for segmentation and mesh fitting directly from dicom files
'''

import sys
import os
from utilsForDicom import DicomExam, loadDicomExam
import _pickle as pickle
import numpy as np
import torch

def wirte_Result_to_txt(dataName,result_list, out_path=None):
	with open(out_path, 'a', encoding='utf-8') as file:
		result = ", ".join(str(item) for item in result_list)
		data_to_append = dataName + ": " + result + "\n"
		file.write(data_to_append)

# data_dir = '/home/yurunyang/lvmeshfitting/DATA/training'
# datasets = [f'patient{i:03d}/patient{i:03d}_4d.nii.gz' for i in range(71, 100)]

# data_dir = '/home/yurunyang/lvmeshfitting/DATA/TEST'
# datasets = ['SCD0000101']

data_dir = "/home/yurunyang/lvmeshfitting/DATA/TEST"
datasets = ['dianfenyangbian']

# data_dir = '/home/yurunyang/lvmeshfitting/DATA/TEST/'
# datasets = ['case142']

# data_dir = '/home/yurunyang/lvmeshfitting/DATA/DCM'
# datasets = ["0002697426", "0051960086", "0052743254", "0052793705", "0052829354", "0052854529", "0052870898", "0052874671"]
# datasets = ["DCM_single"]


# data_dir = 'autodl-tmp'
# datasets = ['040all']

# data_dir = 'autodl-tmp/newdata'
# datasets = ['DCM']

# data_dir = 'autodl-tmp'
# datasets = ['16years']

# data_dir = 'autodl-tmp'
# datasets = ['0002697426']

stage = 2 # <-- stage 1 = segment, stage 2 = fit mesh (stage 1 must be run before stage 2 can be)
tta = False

# if len(sys.argv) > 1:
# 	stage = int(sys.argv[1])
# if stage==1 and len(sys.argv) > 2:
# 	tta = int(sys.argv[2])

print('running stage %d' % (stage,))

if tta:
	print('using TTA')

if stage == 1:

	for dataset_to_use in datasets:

		input_path = os.path.join(data_dir, dataset_to_use)

		# output_path = os.path.join(data_dir, '/Output/')
		# if not os.path.exists(output_path):
		# 	os.makedirs(output_path)

		de = DicomExam(input_path) #create a dicom exam object, by passing the dicom folder path
		de.standardiseTimeframes() #re-sample data in the time-dimension so that all series have the same number of time steps
		de.segment(use_tta=tta) #segment all the data (possibly using test-time augmentation)
		de.save()
		de.proxyCardiacIndecies('network') #produces a plot of various cardiac indicies (saved as proxy_strains_from_network_predictions.png in the plots folder)
		de.summary() #prints a summary of the data, including its shape after resampling and cropping for segmentation
		de.calculateSegmentationUncertainty()
		de.estimateValvePlanePosition() #heuristically remove predicted mask on atrial slices (eroneously predicted by the segmentation algorithm)
		de.estimateLandmarks()
		de.saveImages(show_landmarks=True, overlay=True) #save a visualisation of the data (+ segmentations if they exist)
		de.save()

if stage == 2:

	for dataset_to_use in datasets:

		input_path = os.path.join(data_dir, dataset_to_use)

		de = loadDicomExam(input_path)
		de.resetMeshFitting()
		de.summary()
        
		iteration_time = 2500        # 不能低于50, >50

		# fitted_timeframe = []
		fitted_timeframe = list(range(de.time_frames))

		final_dice = []        
        
		for fit_tf in fitted_timeframe:
			# 拟合外膜，返回的第三个值就是拟合最好的mesh      
			best_epi_dice, _, best_epi_meshes, best_epi_outputs, all_target_label, epi_target = de.fitMesh(training_steps=iteration_time, time_frames_to_fit=[fit_tf], add_modes_label=True, save_training_progress_gif=False, burn_in_length=0, train_mode='normal',
			mode_loss_weight = 0.05, #how strongly to penalise large mode values
			global_shift_penalty_weigth = 0.3,
			lr = 0.001,
			is_epi_pts=True,
			cp_frequency = 200) 
			# 获得外模拟合的mesh
			# de.saveEpiEndoMeshes(best_epi_meshes, True)        
			# de.test_getFinalDice(best_epi_outputs, epi_target)

			# 保存外膜dice
			print("看看外膜结果：", best_epi_dice)
			dataName = dataset_to_use.split("/")
			out_path = '/home/yurunyang/lvmeshfitting/epi_endo_dice/DCM/epi_dice.txt'
			wirte_Result_to_txt(dataName[0], best_epi_dice, out_path)
			import torchvision as tc
			tc.utils.save_image(best_epi_outputs[0][0,0,:,:,3], '%d-003-15-outputs_no_axis.png' % fit_tf)

        
        
        
        
			# 拟合内膜
			best_endo_dice, _, best_endo_meshes, best_endo_outputs, _, endo_target = de.fitMesh(training_steps=iteration_time, time_frames_to_fit=[fit_tf], add_modes_label=True, save_training_progress_gif=False, burn_in_length=0, train_mode='normal',
			mode_loss_weight = 0.05, #how strongly to penalise large mode values
			global_shift_penalty_weigth = 0.3,
			lr = 0.001,
			is_epi_pts=False)
			# 获得内模拟合的mesh
			# de.saveEpiEndoMeshes(best_endo_meshes, False)  
			# de.test_getFinalDice(best_epi_outputs, best_endo_outputs, epi_target, endo_target)

			# 保存内膜dice
			print("看看内膜结果：", best_endo_dice)
			dataName = dataset_to_use.split("/")
			out_path = '/home/yurunyang/lvmeshfitting/epi_endo_dice/DCM/endo_dice.txt'
			wirte_Result_to_txt(dataName[0], best_endo_dice, out_path)
        
            
			# yyr 整合mesh
			final_meshes = de.mergeMeshes(best_epi_meshes, best_endo_meshes) 
			# 保存mesh
			de.saveMeshes(final_meshes, fit_tf)		        
        
        
        
			# 获得final_outputs
			final_outputs = de.getFinalOutputs(best_epi_outputs, best_endo_outputs)
			

			# import torchvision as tc
			# tc.utils.save_image(final_outputs[0][0,0,:,:,3], '003-15-outputs_no_axis.png')

			# 获得final_target
			final_target = de.getFinalOutputs(epi_target, endo_target)   
			# yyr 计算dice
			final_dice.append(de.getFinalDice(final_outputs, final_target, fit_tf))
    
		########################################################################################  
        
		for i in range(len(fitted_timeframe)):
			print("=====================================================")
			print("第%d个时期的dice：" % fitted_timeframe[i], final_dice[i])        
			print("=====================================================")
        
		de.save()
		de.proxyCardiacIndecies('mesh') #produces a plot of various cardiac indicies calculated using the rendered and sliced mesh
		de.calculateSegmentationUncertainty('mesh')
		de.saveImages(use_mesh_images=True)

		# print("看看结果：", final_dice)
		# dataName = dataset_to_use.split("/")
		# wirte_Result_to_txt(dataName[0], final_dice)