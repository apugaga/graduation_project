import os
import nibabel as nib
import numpy as np

# === [1] 讀取 mask，取得有效腦區位置 ===
mask_path = r"D:\laboratory\Graduation_thesis\data\mask\MNI152_T1_2mm_brain_mask.nii"
mask_img = nib.load(mask_path)
mask_data = mask_img.get_fdata()
valid_mask = mask_data == 1

# === [2] 指定資料夾：含多個 .nii/.nii.gz ===
nii_folder = r"F:\laboratory\Graduation_thesis\data\data_fish\output_similar"
output_folder = os.path.join(nii_folder, "voxel_values")  # 儲存位置
os.makedirs(output_folder, exist_ok=True)

# === [3] 尋找所有 .nii 或 .nii.gz 檔案 ===
nii_files = [
    f for f in os.listdir(nii_folder)
    if f.endswith(".nii") or f.endswith(".nii.gz")
]

# === [4] 處理每一個檔案：擷取有效 voxel 值並儲存 ===
for filename in nii_files:
    nii_path = os.path.join(nii_folder, filename)
    nii_img = nib.load(nii_path)
    nii_data = nii_img.get_fdata()

    # 檢查 shape 是否符合
    if nii_data.shape != valid_mask.shape:
        print(f"❌ Shape 不一致，跳過檔案：{filename}")
        continue

    # 擷取有效位置的 voxel 值
    valid_values = nii_data[valid_mask]

    # 儲存為 .npy（可改存成 .csv）
    base_name = os.path.splitext(os.path.splitext(filename)[0])[0]
    output_path = os.path.join(output_folder, f"{base_name}_voxels.npy")
    np.save(output_path, valid_values)

    print(f"✅ 已儲存：{output_path}，共 {len(valid_values)} 個值")
