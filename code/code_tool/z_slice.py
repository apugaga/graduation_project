import os
import numpy as np
import nibabel as nib

# 修改為你本地的資料夾路徑
nii_folder = r"F:\laboratory\Graduation_thesis\data\data_fish\output_similar"
# nii_folder = r"F:\laboratory\Graduation_thesis\data\data_hung\output_similar"

# 尋找所有 .nii 或 .nii.gz 檔案
nii_paths = [
    os.path.join(nii_folder, f)
    for f in os.listdir(nii_folder)
    if f.endswith(".nii") or f.endswith(".nii.gz")
]

slice_means = []
for path in nii_paths:
    img = nib.load(path).get_fdata()
    z_mean = []
    
    # 每個 z 軸切片計算平均（排除整片都是 NaN 的情況）
    for z in range(img.shape[2]):
        slice_data = img[:, :, z]
        if np.isnan(slice_data).all():
            z_mean.append(np.nan)
        else:
            z_mean.append(np.nanmean(slice_data))

    z_mean = np.array(z_mean)
    max_z = np.nanargmax(z_mean)  # 用 nanargmax 排除 nan
    print(f"{os.path.basename(path)} → 最大強度 z = {max_z}")
    slice_means.append(z_mean)

# 對所有 subject 的 z 平均進行整體平均
slice_means = np.array(slice_means)
mean_across_subjects = np.nanmean(slice_means, axis=0)
best_z = np.nanargmax(mean_across_subjects)
print(f"\n建議使用的 axial 切片（整體平均最強）：z = {best_z}")
