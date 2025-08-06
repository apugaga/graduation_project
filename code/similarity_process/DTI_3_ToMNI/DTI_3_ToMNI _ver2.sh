#!/bin/bash

cd /Users/bil/Desktop/DWI_files_3/CN # 這行之後要打在terminal裡面 # 路徑要改

export DTI_path=/Users/bil/Desktop/DWI_files_3/CN # 可更改最後檔案夾名稱

for files in *;do
	###################### native T1 to MNI ######################
	# $files = SubjName
	mkdir ${DTI_path}/${files}/ToMNI_file
	cp /usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz ${DTI_path}/${files}/ToMNI_file/MNI152_T1_2mm.nii.gz
	cp /usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz ${DTI_path}/${files}/ToMNI_file/MNI152_T1_2mm_brain_mask.nii.gz

	cd ${DTI_path}/${files}/ToMNI_file

	temp=MNI152_T1_2mm

	# 生成T1 to MNI的矩陣(線性)
	# output: ${files}_str2MNI_affine_transf.mat
	flirt -ref ${temp}.nii.gz -in ${DTI_path}/${files}/Process/4_T1preproc/T1w-deGibbs-BiasCo.nii.gz -omat ${files}_str2MNI_affine_transf.mat

	# 生成T1 to MNI的矩陣(非線性)
	# output: ${files}_str2MNI_nonlinear_transf
	fnirt --in=${DTI_path}/${files}/Process/4_T1preproc/T1w-deGibbs-BiasCo.nii.gz --aff=${files}_str2MNI_affine_transf.mat --ref=${temp}.nii.gz --refmask=${temp}_brain_mask.nii.gz --cout=${files}_str2MNI_nonlinear_transf --config=T1_2_MNI152_2mm

	# 將T1對位至MNI中
	# output: T1w-deGibbs-BiasCo_MNI.nii.gz
	applywarp --ref=${temp}.nii.gz --in=${DTI_path}/${files}/Process/4_T1preproc/T1w-deGibbs-BiasCo.nii.gz --warp=${files}_str2MNI_nonlinear_transf --out=${DTI_path}/${files}/ToMNI_file/T1w-deGibbs-BiasCo_MNI.nii.gz

	echo ${files} part1 is finish!


	###################### DTI to native T1 ######################
	EPIinput=${DTI_path}/${files}/Average_b0_222_demo

	# 生成DWI對位至T1空間的矩陣
	epi_reg --epi=${EPIinput}.nii.gz --t1=${DTI_path}/${files}/Process/4_T1preproc/T1w-deGibbs-BiasCo.nii.gz --t1brain=${DTI_path}/${files}/Process/4_T1preproc/T1w-deGibbs-BiasCo-Brain.nii.gz --out=${files}_dwi2str

	echo ${files} part2 is finish!


	###################### native T1 to DTI ######################
	# Get DWI to T1 's reverse matrix (str2dwi)
	convert_xfm -omat ${files}_str2dwi.mat -inverse ${files}_dwi2str.mat

	echo ${files} part3 is finish!


	###################### DTI to MNI(combine transform) ######################
	# str2MNI_nonlinear_transf is T1 to MNI 's matrix
	# dwi2str is DWI to T1 's matrix
	# dwi2mni_comprehensive_warps is DWI to MNI 's matrix
	convertwarp --ref=${temp}.nii.gz --warp1=${files}_str2MNI_nonlinear_transf.nii.gz --premat=${files}_dwi2str.mat --out=${files}_dwi2mni_comprehensive_warps.nii.gz --relout

	echo ${files} part4 is finish!


	###################### MNI to DTI ######################
	# mni2dwi_comprehensive_warps is DWI to MNI 's reverse matrix
	invwarp --ref=${EPIinput}.nii.gz --warp=${files}_dwi2mni_comprehensive_warps.nii.gz --rel --out=${files}_mni2dwi_comprehensive_warps.nii.gz

	echo ${files} part5 is finish!


	###################### transform atlas to dwi space ######################
	# atlas=/usr/local/fsl/data/atlases/JHU/JHU-ICBM-labels-2mm.nii.gz

	# 獲得MNI(atlas)對位至DWI的影像，其影像包含50個腦區(小腦有10區)
	applywarp --ref=${EPIinput}.nii.gz --in=/usr/local/fsl/data/atlases/JHU/JHU-ICBM-labels-2mm.nii.gz --warp=${files}_mni2dwi_comprehensive_warps --out=JHU-ICBM-labels-dwi-native.nii.gz --interp=nn

	echo ${files} part6 is finish!

	# break
done

