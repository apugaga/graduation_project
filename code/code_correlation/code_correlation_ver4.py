import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
#以下不使用
def save_correlations_to_nifti1(correlations, mask_file, output_file):
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
    temp = 0;
    for i, (x, y, z) in enumerate(locations):
        correlation_map[int(x), int(y), int(z)] = correlations[i]
        if correlations[i] > temp:
            temp = correlations[i]
            max_x = x
            max_y = y
            max_z = z
            max_i = i
    print(max_x, " ", max_y, " ", max_z, " ",max_i)

    # 創建新的 NIfTI 文件
    correlation_nifti = nib.Nifti1Image(correlation_map, mask_img.affine, mask_img.header)

    # 保存為 .nii.gz 文件
    #nib.save(correlation_nifti, output_file)
    print(f"相關性結果已保存為: {output_file}")

def map_results_to_template(mask_file, correlations, output_file, voxel_matrix):
    """
    將相關性數據填回大腦模板，並保存為新 .nii.gz 文件。
    :param mask_file: 模板文件（NIfTI 文件）路徑
    :param correlations: 一維相關性數據，與模板中的 voxel 對應
    :param output_file: 儲存的 .nii.gz 文件路徑
    """
    # 載入大腦模板
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()

    # 創建與模板相同形狀的數據陣列
    result_map = np.zeros_like(mask_data)

    # 找出模板中值為 1 的位置
    locations = np.argwhere(mask_data == 1)
    assert len(correlations) == len(locations), "相關性數據與模板中的位置數量不匹配！"
    
    # 將相關性數據填入模板
    # 初始化變數
    temp = float("-inf")  # 第一大的數值
    max_x = max_y = max_z = max_i = None

    # 搜尋第一大與第二大數值
    for i, (x, y, z) in enumerate(locations):
        result_map[int(x), int(y), int(z)] = correlations[i]
        
        if correlations[i] > temp:
            if np.all(~np.isnan(voxel_matrix[:, i])):
                # 更新最大值
                temp = correlations[i]
                max_x, max_y, max_z, max_i = x, y, z, i


    # 打印結果
    print("Max Corr location: ", max_x, max_y, max_z, max_i, "Value: ", temp)
        

    # 保存為新的 NIfTI 文件
    result_img = nib.Nifti1Image(result_map, mask_img.affine, mask_img.header)
    nib.save(result_img, output_file)
    print(f"相關性結果已成功保存到模板中: {output_file}")

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
    temp = 0
    # 填充相關性值到對應的 voxel
    locations = np.argwhere(mask_data == 1)  # 找出值為 1 的位置
    for i, (x, y, z) in enumerate(locations):
        correlation_map[int(x), int(y), int(z)] = correlations[i]
        if correlations[i] > temp:
            temp = correlations[i]
            max_x = x
            max_y = y
            max_z = z
            max_i = i
    print("Corr location: ",max_x, " ", max_y, " ", max_z, " ",max_i)

    # 創建新的 NIfTI 文件
    correlation_nifti = nib.Nifti1Image(correlation_map, mask_img.affine, mask_img.header)

    # 保存為 .nii.gz 文件
    nib.save(correlation_nifti, output_file)
    print(f"相關性結果已保存為: {output_file}")

def construct_voxel_matrix(similarity_folder, locations):
    """
    構建人-voxel 的矩阵。
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
mask_file = r"D:\laboratory\Graduation_thesis\data\file\MNI152_T1_2mm_brain_mask.nii.gz"  # mask 檔案路徑
similarity_folder = r"D:\laboratory\Graduation_thesis\data\output_similar"  # similarity 檔案存放資料夾
excel_file = r"D:\laboratory\Graduation_thesis\patient_data\filtered_output_ver2.xlsx"  # 包含年齡資料的 Excel 檔案路徑
output_file = r"D:\laboratory\Graduation_thesis\data\output_similar\correlation_results.nii.gz"
# Step 1: 找到 mask 中值為 1 的位置
mask_img = nib.load(mask_file)
mask_data = mask_img.get_fdata()
locations = np.argwhere(mask_data == 1).tolist()

# Step 2: 構建人-voxel 矩陣
#目的:形成受試者-voxel的二維矩陣(voxel值為1)
voxel_matrix, valid_files = construct_voxel_matrix(similarity_folder, locations)
print(f"人-voxel 矩陣形狀: {voxel_matrix.shape}")
max_list = voxel_matrix[:, 14036]


# Step 3: 構建年齡向量
#目的:從Excel中讀取受試者年齡，並與valid_files影像對應
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

#相關性結果回填大腦模板
#目的:找出相關性最大值
map_results_to_template(mask_file, correlations, output_file, voxel_matrix)