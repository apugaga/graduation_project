import os
import nibabel as nib
import numpy as np

def process_nii_files(directory):
    """
    處理指定資料夾內所有符合條件的 .nii.gz 檔案。
    :param directory: 資料夾路徑
    """
    # 確保資料夾存在
    if not os.path.exists(directory):
        print(f"資料夾 {directory} 不存在！")
        return
    
    # 遍歷資料夾內所有檔案
    for file_name in sorted(os.listdir(directory)):
        if file_name.startswith("sub-ACN") and file_name.endswith(".nii.gz"):
            file_path = os.path.join(directory, file_name)
            
            try:
                # 載入 NIfTI 檔案
                nifti_img = nib.load(file_path)
                nifti_data = nifti_img.get_fdata()

                # 計算統計值
                nan_count = np.isnan(nifti_data).sum()
                greater_than_zero_count = np.sum(nifti_data > 0)
                less_than_zero_count = np.sum(nifti_data < 0)
                max_value = np.nanmax(nifti_data)

                # 輸出結果
                print(f"檔案: {file_name}")
                print(f"  NaN 數量: {nan_count}")
                print(f"  大於 0 的數值數量: {greater_than_zero_count}")
                print(f"  小於 0 的數值數量: {less_than_zero_count}")
                print(f"  最大值: {max_value}\n")

            except Exception as e:
                print(f"無法處理檔案 {file_name}，錯誤訊息: {e}")

# 指定資料夾路徑
directory_path = r"D:\laboratory\Graduation_thesis\data\data_fish\no_nan_JHU"  # 替換為您的資料夾路徑

# 執行函數
if __name__ == "__main__":
    process_nii_files(directory_path)
