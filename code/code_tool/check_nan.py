import nibabel as nib
import numpy as np

# 輸入你的 NIfTI 檔案路徑
nii_path = r"D:\laboratory\Graduation_thesis\data\data_fish\output_similar_no_nan\sub-ACN0007_DTIFCT_weighted_similar.nii.gz"  # 或 .nii

# 讀取 NIfTI 檔案
img = nib.load(nii_path)
data = img.get_fdata()

# 檢查是否有 NaN
has_nan = np.isnan(data).any()

# 顯示結果
if has_nan:
    print("⚠️ 該 NIfTI 檔案中包含 NaN 值。")
else:
    print("✅ 該 NIfTI 檔案中不包含 NaN 值。")
