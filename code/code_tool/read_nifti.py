import nibabel as nib
import numpy as np

def read_nifti_file(file_path):
    """
    讀取並檢查 NIfTI 檔案 (.nii 或 .nii.gz) 的內容。
    :param file_path: NIfTI 檔案的路徑
    """
    # 讀取 NIfTI 文件
    img = nib.load(file_path)

    # 提取數據矩陣
    data = img.get_fdata()

    # 顯示基本資訊
    print(f"檔案路徑: {file_path}")
    print(f"數據形狀: {data.shape}")
    print(f"數據類型: {data.dtype}")
    print(f"Affine 矩陣: \n{img.affine}")

    # 檢查數據內容的基本統計
    print(f"數據最小值: {np.nanmin(data)}")
    print(f"數據最大值: {np.nanmax(data)}")
    print(f"數據平均值: {np.nanmean(data)}")
    print(f"數據中位數: {np.nanmedian(data)}")
    print(f"數據中 NaN 值數量: {np.sum(np.isnan(data))}")

    # 返回數據陣列
    return data

# 使用範例
file_path = r"F:\laboratory\Graduation_thesis\data\data_fish\output_similar\sub-ACN0004_DTIFCT_weighted_similar.nii.gz"  # 修改為你的文件路徑
data = read_nifti_file(file_path)

# 選擇性打印部分數據 (檢查內容是否正確)
print("數據部分內容 (前10個非零值):")
non_zero_values = data[data != 0]  # 過濾掉零值
print(non_zero_values[:10])  # 打印前 10 個非零值
