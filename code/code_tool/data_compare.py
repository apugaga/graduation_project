import os
import numpy as np
import nibabel as nib

# === [1] 讀取 mask 的有效位置座標 ===
mask_path = r"D:\laboratory\Graduation_thesis\data\data_fish\output_similar\sub-ACN0004_DTIFCT_weighted_similar.nii.gz"
mask_img = nib.load(mask_path)
mask_data = mask_img.get_fdata()
# valid_mask = mask_data == 1
z_index =41
valid_mask = mask_data[:, :, z_index] == 1  # shape 為 (x, y)
# 計算有多少個 True（即有值的位置）
count = np.count_nonzero(valid_mask)
print(f"Z = {z_index} 層中，mask == 1 的 voxel 數量：{count}")

# === [2] 設定輸入與輸出資料夾 ===
nii_folder = r"F:\laboratory\Graduation_thesis\data\data_fish\output_similar"
output_folder = os.path.join(nii_folder, "z41_voxel_values")
os.makedirs(output_folder, exist_ok=True)

# === [3] 尋找所有 .nii 或 .nii.gz 檔案 ===
nii_files = [
    f for f in os.listdir(nii_folder)
    if f.endswith(".nii") or f.endswith(".nii.gz")
]

# === [4] 處理每個影像檔案 ===
for filename in nii_files:
    nii_path = os.path.join(nii_folder, filename)
    img = nib.load(nii_path)
    data = img.get_fdata()

    # 檢查 z 軸是否包含 z=41
    if data.shape[2] <= z_index:
        print(f"⚠️ 檔案 {filename} 的 z 軸長度不足，跳過。")
        continue

    slice_data = data[:, :, z_index]
    valid_values = slice_data[valid_mask]

    # 儲存為 .npy（也可以改為 .csv）
    base_name = os.path.splitext(os.path.splitext(filename)[0])[0]
    output_path = os.path.join(output_folder, f"{base_name}_z41_voxels.npy")
    np.save(output_path, valid_values)

    print(f"✅ 已儲存：{output_path}，共 {len(valid_values)} 個值")

# =============================================================================
# # 逐一讀取並印出 shape
# nii_files = [
#     f for f in os.listdir(masked_folder)
#     if f.endswith(".nii") or f.endswith(".nii.gz")
# ]
# for filename in nii_files:
#     path = os.path.join(masked_folder, filename)
#     img = nib.load(path)
#     shape = img.shape
#     print(f"{filename} → shape = {shape}")
# =============================================================================
