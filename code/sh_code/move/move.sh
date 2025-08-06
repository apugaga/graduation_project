#!/bin/bash

cd /f/laboratory/Graduation_thesis/data/CN_change

export CN_path=/f/laboratory/Graduation_thesis/data/CN_change
export output_path=/f/laboratory/Graduation_thesis/data/CN

for files in *;do

	mkdir ${output_path}/sub-${files}
	
	#cp ${CN_path}/${files}/Average_b0_222.nii.gz ${output_path}/sub-${files}
	cp ${CN_path}/${files}/Average_b0_222.nii.gz ${output_path}/sub-${files}/Average_b0_222_demo.nii.gz
	cp ${CN_path}/${files}/${files}_222_T1_mask.nii.gz ${output_path}/sub-${files}/sub-${files}_T1_finalmask.nii.gz
	cp ${CN_path}/${files}/${files}_FCT.nii.gz ${output_path}/sub-${files}/sub-${files}_FCT.nii.gz
	cp ${CN_path}/${files}/${files}_tensor.nii.gz ${output_path}/sub-${files}/sub-${files}_tensor.nii.gz
	cp ${CN_path}/${files}/sub-${files}_FA.nii.gz ${output_path}/sub-${files}/sub-${files}_FA.nii.gz
	cp ${CN_path}/${files}/sub-${files}_FCT_FA.nii.gz ${output_path}/sub-${files}/sub-${files}_FCT_FA.nii.gz
	cp ${CN_path}/${files}/T1w-deGibbs-BiasCo.nii.gz ${output_path}/sub-${files}/T1w-deGibbs-BiasCo.nii.gz
	#cp ${CN_path}/${files}/${files}_tensor.nii.gz ${output_path}/sub-${files}
	#cp ${CN_path}/${files}/4_T1preproc/T1w-deGibbs-BiasCo.nii.gz ${output_path}/sub-${files}

	echo ${files} is finish!
	
done

