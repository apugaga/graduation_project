import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def find_mask_locations(mask_file):
    """
    載入mask檔案，找出所有值為1的位置座標。
    :param mask_file: mask的.nii.gz檔案路徑
    :return: 一個包含所有座標的列表 [(x1, y1, z1), (x2, y2, z2), ...]
    """
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()
    locations = np.argwhere(mask_data == 1)  # 找出值為1的位置
    return locations.tolist()

def read_similarity_files(similarity_folder, locations):
    """
    根據提供的座標讀取similarity nifti檔案，並記錄座標對應的數值。
    :param similarity_folder: similarity檔案存放的資料夾路徑
    :param locations: mask中值為1的座標列表
    :return: 一個字典，包含每個檔案和對應座標的數值
    """
    similarity_data = {}

    # 遍歷資料夾內所有similarity檔案
    for file_name in sorted(os.listdir(similarity_folder)):
        if file_name.startswith("sub-NR1_") and file_name.endswith(".nii.gz"):
            file_path = os.path.join(similarity_folder, file_name)
            img = nib.load(file_path)
            img_data = img.get_fdata()

            # 提取每個座標的數值
            values = []
            for loc in locations:
                x, y, z = loc
                values.append(img_data[int(x), int(y), int(z)])
                
            similarity_data[file_name] = values
            print(len(similarity_data))
    return similarity_data



def calculate_correlation(similarity_results, excel_file):
    """
    與 Excel 檔案中的年齡進行相關性計算。
    :param similarity_results: similarity檔案的數據字典
    :param excel_file: 包含年齡資料的 Excel 檔案路徑
    """
    # 讀取 Excel 檔案
    df = pd.read_excel(excel_file)
    
    # 建立 Subject 名稱與年齡的對應
    df['File_ID'] = df['Subject'].apply(lambda x: f"sub-NR1_{x[-3:]}")  # 將 VNRxxxx 轉為 sub-NR1_xxx 格式
    
    age_mapping = df.set_index('File_ID')['age'].to_dict()
    # print(f"age_mapping 的內容: {age_mapping}")
    
    '''
    for file_name in similarity_results.keys():
        if file_name in age_mapping:
            print(f"找到匹配的檔案: {file_name}")
        else:
            print(f"未匹配的檔案: {file_name}")
    '''




    correlations = {}
    all_nan_keys = [] 

    # 計算相關性
    # value代表特定座標的數值
    for file_name, values in similarity_results.items():
        base_file_name = file_name.split('_DTIFCT_weighted_similar.nii.gz')[0]
        if values is None or np.all(np.isnan(values)):
            all_nan_keys.append(file_name)
            continue
        if base_file_name in age_mapping:
            age = age_mapping[base_file_name]
            if len(values) > 0:  # 確保有數據
                cleaned_values = [v for v in values if not np.isnan(v)]
                if len(cleaned_values) > 1:
                    corr, _ = pearsonr(cleaned_values, [age] * len(cleaned_values))
                    correlations[base_file_name] = corr
            else:
                correlations[base_file_name] = np.nan
        else:
            correlations[base_file_name] = np.nan
    print(f"以下檔案的鍵值完全為 NaN：{all_nan_keys}")

    return correlations

# 使用範例
mask_file = r"F:\laboratory\Graduation_thesis\data\file\MNI152_T1_2mm_brain_mask.nii.gz"  # 修改為mask檔案的實際路徑
similarity_folder = r"F:\laboratory\Graduation_thesis\data\output_similar"  # 修改為similarity檔案存放的資料夾路徑
excel_file = r"F:\laboratory\Graduation_thesis\patient_data\filtered_output_ver2.xlsx"  # 修改為包含年齡資料的Excel路徑

# 步驟1：找到mask中值為1的位置
locations = find_mask_locations(mask_file)
print(f"找到 {len(locations)} 個位置值為1的座標。")

# 步驟2：讀取所有similarity檔案的座標數值
similarity_results = read_similarity_files(similarity_folder, locations)
#print(f"similarity_results 中有 {len(similarity_results)} 個檔案的數據")
# 檢查 similarity_results 中是否有鍵值包含 NaN
nan_keys = []

for file_name, values in similarity_results.items():
    if values is not None and np.any(np.isnan(values)):  # 檢查是否有 NaN
        nan_keys.append(file_name)

print(f"以下鍵名的鍵值包含 NaN: {nan_keys}")

# 步驟3：計算與年齡的相關性
correlations = calculate_correlation(similarity_results, excel_file)

# 輸出相關性結果
print("相關性結果：")
for file_name, corr in correlations.items():
    print(f"{file_name}: {corr:.3f}")
