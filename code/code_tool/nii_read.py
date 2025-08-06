
import nibabel as nib
import numpy as np

def analyze_nifti(file_path):
    """
    讀取 NIfTI 檔案並計算其統計資訊：
    - 最大值、最小值、平均值
    - 最大值數量、最小值數量
    :param file_path: .nii 檔案的路徑
    """
    # 讀取 NIfTI 檔案
    img = nib.load(file_path)
    data = img.get_fdata()

    # 計算統計數據
    max_value = np.nanmax(data)  # 最大值
    min_value = np.nanmin(data)  # 最小值
    mean_value = np.nanmean(data)  # 平均值

    # 計算最大值和最小值的數量
    max_count = np.sum(data == max_value)
    min_count = np.sum(data == min_value)

    # 輸出結果
    print(f"📂 檔案名稱: {file_path}")
    print(f"📊 影像維度: {data.shape}")
    print(f"🔹 最大值: {max_value} (數量: {max_count})")
    print(f"🔹 最小值: {min_value} (數量: {min_count})")
    print(f"🔹 平均值: {mean_value}")

# 範例使用
file_path = input("請輸入位址:")  # 這裡替換成你的 NIfTI 檔案路徑
analyze_nifti(file_path)
