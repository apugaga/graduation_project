import nibabel as nib

# 載入 NIfTI 檔案
file_path = r"F:\laboratory\Graduation_thesis\data\output_similar\sub-NR1_001_DTIFCT_weighted_similar.nii.gz"  # 修改為實際檔案路徑
nii_file = nib.load(file_path)

# 取得數據矩陣和維度
data = nii_file.get_fdata()
data_shape = data.shape

# 顯示數據維度
print("數據的維度:", data_shape)

# 如果需要檢視數據某些區域的值，可以嘗試以下操作
# 例如，檢視某個切片的數據
slice_index = 70  # 修改為你要查看的切片索引
slice_data = data[:, :, slice_index]

print(f"第 {slice_index} 個切片的數據：")
print(slice_data)
