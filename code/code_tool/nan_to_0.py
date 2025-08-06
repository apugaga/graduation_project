import os
import numpy as np
import nibabel as nib

# === 設定路徑 ===
template_path = r"D:\laboratory\Graduation_thesis\data\mask\MNI152_T1_2mm_brain_mask.nii"  # 你的模板檔案
main_folder = r"D:\laboratory\Graduation_thesis\data\data_fish\output_similar"           # 含多個 nii 檔案的資料夾
output_folder = os.path.join(main_folder, "processed_nii")

# === 建立輸出資料夾 ===
os.makedirs(output_folder, exist_ok=True)

# === 載入模板並取出資料副本 ===
template_img = nib.load(template_path)
template_data = template_img.get_fdata()
template_shape = template_data.shape

# === 走訪主資料夾 ===
for file_name in os.listdir(main_folder):
    if not (file_name.endswith(".nii") or file_name.endswith(".nii.gz")):
        continue  # 跳過非 NIfTI 檔案

    file_path = os.path.join(main_folder, file_name)
    nii_img = nib.load(file_path)
    nii_data = nii_img.get_fdata()
    nii_affine = nii_img.affine

    # === 檢查 shape 是否一致 ===
    if nii_data.shape != template_shape:
        print(f"❌ 檔案 {file_name} 尺寸與模板不同")
        print("不同")
        break  # 立即中斷程式

    # === 建立副本做處理 ===
    nii_copy = np.copy(nii_data)
    # === 將 NaN 轉為 0 ===
    nii_copy[np.isnan(nii_copy)] = 0
    # === 套用 template 掩膜：template 為 0 的區域全部設為 0 ===
    nii_copy[template_data == 0] = 0

# =============================================================================
#     # === 條件檢查：NaN 對應到 0，非 NaN 對應到 1 ===
#     nan_mask = np.isnan(nii_copy)
#     not_nan_mask = ~nan_mask
# =============================================================================

# =============================================================================
#     # === 檢查 NaN 應對應到 template=0，非 NaN 對應 template=1 ===
#     nan_mismatch = np.logical_and(nan_mask, template_data != 0)
#     not_nan_mismatch = np.logical_and(not_nan_mask, template_data != 1)
#     
#     total_mismatch = np.count_nonzero(nan_mismatch) + np.count_nonzero(not_nan_mismatch)
#     
#     if total_mismatch > 0:
#         print(f"⚠️ 檔案 {file_name} 有異常：NaN 或 非NaN 區域對應不符模板")
#         print(f"❗ 不符位置總數：{total_mismatch}")
#         continue
# 
# 
#     # === 替換 NaN 為 0 ===
#     nii_copy[nan_mask] = 0
# =============================================================================

    # === 儲存處理後影像 ===
    new_img = nib.Nifti1Image(nii_copy, affine=nii_affine)
    new_file_path = os.path.join(output_folder, file_name)
    nib.save(new_img, new_file_path)
    print(f"✅ 已處理並儲存：{file_name}")
