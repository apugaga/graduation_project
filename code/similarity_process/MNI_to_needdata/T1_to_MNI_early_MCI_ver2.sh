#!/bin/bash

# 下面路徑要更改
cd /media/HD14TB2/apugaga/CN_data/DWI_files


export MNI_path=/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz
export MNImask_path=/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz
export T1_2_MNI_config=/usr/local/fsl/src/fnirt/fnirtcnf/T1_2_MNI152_2mm.cnf

# 下面路徑要更改
export T1_path=/media/HD14TB2/apugaga/CN_data/DWI_files
export output_path=/media/HD14TB2/apugaga/CN_data/DWI_files


# 需要applywarp _FCT.nii.gz _tensor.nii.gz _FCT_FA.nii.gz _FA.nii.gz _T1_finalmask.nii.gz
for files in *; do

	mkdir -p ${output_path}/${files}/T1_to_MNI

	# create T1 to MNI affine transfer
	# flirt -in ${T1_path}/${files}/4_T1preproc/T1w-deGibbs-BiasCo.nii.gz -ref ${MNI_path} -omat ${output_path}/${files}/T1_to_MNI/T1_to_MNI_affine_transfer.mat

	# create T1 to MNI nonlinear transfer
	# fnirt --in=${T1_path}/${files}/4_T1preproc/T1w-deGibbs-BiasCo.nii.gz --aff=${output_path}/${files}/T1_to_MNI/T1_to_MNI_affine_transfer.mat --ref=${MNI_path} --refmask=${MNImask_path} --cout=${output_path}/${files}/T1_to_MNI/T1_to_MNI_nonlinear_transfer.nii.gz --config=${T1_2_MNI_config}

	# applywarp T1 to MNI 檢查
	# 這邊的in要放
	applywarp --in=${T1_path}/${files}/${files}_FCT.nii.gz --warp=${output_path}/${files}/ToMNI_file/${files}_dwi2mni_comprehensive_warps.nii.gz --ref=${MNI_path} --out=${output_path}/${files}/T1_to_MNI/T1_to_MNI_FCT_registration.nii.gz
	echo ${files} FCT is finish!
	
	applywarp --in=${T1_path}/${files}/${files}_tensor.nii.gz --warp=${output_path}/${files}/ToMNI_file/${files}_dwi2mni_comprehensive_warps.nii.gz --ref=${MNI_path} --out=${output_path}/${files}/T1_to_MNI/T1_to_MNI_tensor_registration.nii.gz
	echo ${files} tensor is finish!
	
	applywarp --in=${T1_path}/${files}/${files}_FCT_FA.nii.gz --warp=${output_path}/${files}/ToMNI_file/${files}_dwi2mni_comprehensive_warps.nii.gz --ref=${MNI_path} --out=${output_path}/${files}/T1_to_MNI/T1_to_MNI_FCT_FA_registration.nii.gz
	echo ${files} FCT_FA is finish!
	
	applywarp --in=${T1_path}/${files}/${files}_FA.nii.gz --warp=${output_path}/${files}/ToMNI_file/${files}_dwi2mni_comprehensive_warps.nii.gz --ref=${MNI_path} --out=${output_path}/${files}/T1_to_MNI/T1_to_MNI_FA_registration.nii.gz
	echo ${files} FA is finish!
	
	applywarp --in=${T1_path}/${files}/${files}_T1_finalmask.nii.gz --warp=${output_path}/${files}/ToMNI_file/${files}_dwi2mni_comprehensive_warps.nii.gz --ref=${MNI_path} --out=${output_path}/${files}/T1_to_MNI/T1_to_MNI_T1_finalmask_registration.nii.gz
	echo ${files} T1_finalmask is finish!

	echo ${files} Finish!
done
