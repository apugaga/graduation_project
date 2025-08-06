import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def save_correlations_to_nifti(correlations, mask_file, output_file):
    """
    保存相關性結果為 .nii.gz 文件。
    :param correlations: 一維相關性數組
    :param mask_file: 用於提供空間參考的 mask 文件路徑
    :param output_file: 輸出的 .nii.gz 文件路徑
    """
    # 載入 mask 文件以獲取空間信息
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()

    # 創建與 mask 相同形狀的空數組
    correlation_map = np.zeros_like(mask_data)

    # 填充相關性值到對應的 voxel
    locations = np.argwhere(mask_data == 1)  # 找出值為 1 的位置
    for i, (x, y, z) in enumerate(locations):
        correlation_map[int(x), int(y), int(z)] = correlations[i]

    # 創建新的 NIfTI 文件
    correlation_nifti = nib.Nifti1Image(correlation_map, mask_img.affine, mask_img.header)

    # 保存為 .nii.gz 文件
    nib.save(correlation_nifti, output_file)
    print(f"相關性結果已保存為: {output_file}")

def construct_voxel_matrix(similarity_folder, locations):
    """
    构建人-voxel 的矩阵。
    :param similarity_folder: similarity 檔案存放的資料夾路徑
    :param locations: mask 中值為 1 的座標列表
    :return: 人-voxel 的矩陣 (人數, voxel 數) 和對應的檔案名稱
    """
    files = sorted([f for f in os.listdir(similarity_folder) if f.startswith("sub-NR1_") and f.endswith(".nii.gz")])
    voxel_matrix = []
    valid_files = []

    for file_name in files:
        file_path = os.path.join(similarity_folder, file_name)
        img = nib.load(file_path)
        img_data = img.get_fdata()

        # 提取所有 voxel 的值
        values = [img_data[int(x), int(y), int(z)] for x, y, z in locations]
        if not np.all(np.isnan(values)):  # 如果數據全部為 NaN，忽略該檔案
            voxel_matrix.append(values)
            valid_files.append(file_name)

    return np.array(voxel_matrix), valid_files

def calculate_correlation(voxel_matrix, age_vector):
    """
    計算每個 voxel 的值與年齡的相關性。
    :param voxel_matrix: 人-voxel 矩陣 (人數, voxel 數)
    :param age_vector: 年齡向量 (1, 人數)
    :return: 每個 voxel 的相關性 (1, voxel 數)
    """
    correlations = []
    for col in range(voxel_matrix.shape[1]):  # 按列計算
        voxel_values = voxel_matrix[:, col]
        
        valid_indices = ~np.isnan(voxel_values) & ~np.isinf(voxel_values)
        cleaned_values = voxel_values[valid_indices]
        cleaned_ages = age_vector[valid_indices]
        
        if np.sum(~np.isnan(voxel_values)) > 1:  # 至少有兩個非 NaN 值
            corr, _ = pearsonr(cleaned_values, cleaned_ages)
            correlations.append(corr)
        else:
            correlations.append(np.nan)
    return np.array(correlations)

# 使用範例
mask_file = r"F:\laboratory\Graduation_thesis\data\file\MNI152_T1_2mm_brain_mask.nii.gz"  # mask 檔案路徑
similarity_folder = r"F:\laboratory\Graduation_thesis\data\output_similar"  # similarity 檔案存放資料夾
excel_file = r"F:\laboratory\Graduation_thesis\patient_data\filtered_output_ver2.xlsx"  # 包含年齡資料的 Excel 檔案路徑
output_file = r"F:\laboratory\Graduation_thesis\data\output_similar\correlation_results.nii.gz"
# Step 1: 找到 mask 中值為 1 的位置
mask_img = nib.load(mask_file)
mask_data = mask_img.get_fdata()
locations = np.argwhere(mask_data == 1).tolist()

# Step 2: 构建人-voxel 矩陣
voxel_matrix, valid_files = construct_voxel_matrix(similarity_folder, locations)
print(f"人-voxel 矩陣形狀: {voxel_matrix.shape}")

# Step 3: 構建年齡向量
df = pd.read_excel(excel_file)
df['File_ID'] = df['Subject'].apply(lambda x: f"sub-NR1_{x[-3:]}")  # 將 VNRxxxx 轉為 sub-NR1_xxx 格式
age_mapping = df.set_index('File_ID')['age'].to_dict()

# 根據 valid_files 構建年齡向量
age_vector = [age_mapping[file.split('_DTIFCT_weighted_similar.nii.gz')[0]] for file in valid_files]
age_vector = np.array(age_vector)
print(f"年齡向量形狀: {age_vector.shape}")

# Step 4: 計算相關性
correlations = calculate_correlation(voxel_matrix, age_vector)
print(f"相關性向量形狀: {correlations.shape}")

save_correlations_to_nifti(correlations, mask_file, output_file)