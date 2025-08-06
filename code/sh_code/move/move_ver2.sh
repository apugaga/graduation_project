#!/bin/bash

cd /media/HD14TB1/zephyr/MCI/Voxel_222/early_MCI/DWI

export CN_path=/media/HD14TB1/zephyr/MCI/Voxel_222/early_MCI/DWI
export output_path=/media/HD14TB1/zephyr/apugaga_data/CN

for files in *;do

	#mkdir ${output_path}/${files}
	
	#cp ${CN_path}/${files}/Average_b0_222.nii.gz ${output_path}/sub-${files}
	cp ${CN_path}/${files}/${files}_222_T1_mask.nii.gz ${output_path}/sub-${files}
	#cp ${CN_path}/${files}/${files}_tensor.nii.gz ${output_path}/sub-${files}
	#cp ${CN_path}/${files}/4_T1preproc/T1w-deGibbs-BiasCo.nii.gz ${output_path}/sub-${files}

	echo ${files} is finish!
done

