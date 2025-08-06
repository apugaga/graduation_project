# 我們將根據用戶的需求，生成一個改寫版本的 age_voxel_correlation_pvalue_heatmap.py
# 使用外層迴圈遍歷固定的 Z slice，內層遍歷受試者，並計算相關性與 p-value

import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ================================
# 流程步驟 1：設定資料路徑與參數
# ================================
nifti_dir = r"E:\laboratory\Graduation_thesis\data\data_fish\no_nan_JHU"  # NIfTI 檔案路徑
age_csv   = r"E:\laboratory\Graduation_thesis\data\data_fish\paitient_data\paitient_age_only.csv"       # 年齡表，包含 subject_id 與 label
output_dir = r"E:\laboratory\Graduation_thesis\data\corr_heatmap"
os.makedirs(output_dir, exist_ok=True)

# 固定要處理的 Z slice
brain_regions = [
    ("Posterior corona radiata R", 46),
    ("Retrolenticular part of internal capsule R", 39),
    ("Anterior corona radiata R", 44),
    ("Tapetum L", 41),
    ("Superior corona radiata R", 52),
]

# ================================
# 流程步驟 2：讀取年齡資料
# ================================
df_age = pd.read_csv(age_csv)
age_dict = dict(zip(df_age["subject_id"], df_age["label"]))  # label 欄位為年齡

# 找出所有受試者的 NIfTI 檔案
nifti_files = sorted([f for f in os.listdir(nifti_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])

# ================================
# 流程步驟 3：外層迴圈 - 固定 Z slice
# ================================
for region_name, z in brain_regions:
    print(f"處理 Z slice = {z}")

    # 先讀第一個檔案確認切片尺寸
    sample_img = nib.load(os.path.join(nifti_dir, nifti_files[0]))
    slice_shape = sample_img.get_fdata()[:, :, z].shape

    # 建立存放該切面所有受試者數據的矩陣 (subjects × X × Y)
    # zero(shape,dtype=float,order)目標是建立空矩陣，然後標題放這兩個東西
    #  *slice_shape將裡面的(X,Y)拆開來變成X,Y
    all_slices = np.zeros((len(nifti_files), *slice_shape), dtype=np.float32)
    ages = []

    # ================================
    # 流程步驟 4：內層迴圈 - 讀取每位受試者該切面的數據
    # ================================
    for i, fname in enumerate(nifti_files):
        sub_id = fname.split("_")[0].replace("sub-", "")
        img_data = nib.load(os.path.join(nifti_dir, fname)).get_fdata()
        slice_2d = img_data[:, :, z]
        slice_2d = np.nan_to_num(slice_2d, nan=0.0)  # 處理 NaN
        all_slices[i, :, :] = slice_2d
        ages.append(age_dict[sub_id])

    ages = np.array(ages, dtype=np.float32)

    # ================================
    # 流程步驟 5：計算該切面每個 voxel 與年齡的相關性與 p-value
    # ================================
    corr_map_2d = np.zeros(slice_shape, dtype=np.float32)
    pval_map_2d = np.ones(slice_shape, dtype=np.float32)

    for x in range(slice_shape[0]):
        for y in range(slice_shape[1]):
            voxel_values = all_slices[:, x, y]
            if np.all(voxel_values == 0):
                corr_map_2d[x, y] = 0
                pval_map_2d[x, y] = 1
            else:
                r, p = pearsonr(voxel_values, ages)
                corr_map_2d[x, y] = r
                pval_map_2d[x, y] = p

    # ================================
    # 流程步驟 6：生成並儲存熱圖
    # ================================
    # 創建該切面專屬資料夾
    safe_region_name = region_name.replace(" ", "_")
    slice_output_dir = os.path.join(output_dir, safe_region_name)
    os.makedirs(slice_output_dir, exist_ok=True)


    # 相關性熱圖
    plt.figure(figsize=(10, 10))
    plt.imshow(corr_map_2d.T, cmap="coolwarm", origin="lower", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson r")
    plt.title(f"Correlation Map - Z={z}")
    plt.axis("off")
    plt.savefig(os.path.join(slice_output_dir, f"correlation_z{z}.png"), dpi=300)
    plt.close()

    # p-value 熱圖
    plt.figure(figsize=(10, 10))
    plt.imshow(pval_map_2d.T, cmap="hot_r", origin="lower", vmin=0, vmax=0.05)
    plt.colorbar(label="p-value")
    plt.title(f"P-value Map - Z={z}")
    plt.axis("off")
    plt.savefig(os.path.join(slice_output_dir, f"pvalue_z{z}.png"), dpi=300)
    plt.close()

print(f"所有切面熱圖已輸出到 {output_dir}")

