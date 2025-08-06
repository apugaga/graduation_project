import os
import nibabel as nib
import numpy as np

def find_mask_locations(mask_file):
    """
    載入mask檔案，找出所有值為1的位置座標。
    :param mask_file: mask的.nii.gz檔案路徑
    :return: 一個包含所有座標的列表 [(x1, y1, z1), (x2, y2, z2), ...]
    """
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()
    locations = np.argwhere(mask_data == 1)  # 找出值為1的位置
    return locations.tolist()

def read_similarity_files(similarity_folder, locations):
    """
    根據提供的座標讀取similarity nifti檔案，並記錄座標對應的數值。
    :param similarity_folder: similarity檔案存放的資料夾路徑
    :param locations: mask中值為1的座標列表
    :return: 一個字典，包含每個檔案和對應座標的數值
    """
    similarity_data = {}

    # 遍歷資料夾內所有similarity檔案
    for file_name in sorted(os.listdir(similarity_folder)):
        if file_name.startswith("sub-NR1_") and file_name.endswith(".nii.gz"):
            file_path = os.path.join(similarity_folder, file_name)
            img = nib.load(file_path)
            img_data = img.get_fdata()

            # 提取每個座標的數值
            values = []
            for loc in locations:
                x, y, z = loc
                values.append(img_data[int(x), int(y), int(z)])
            similarity_data[file_name] = values
            print(len(similarity_data))

    return similarity_data

# 使用範例
mask_file = r"F:\laboratory\Graduation_thesis\data\file\MNI152_T1_2mm_brain_mask.nii.gz"  # 修改為mask檔案的實際路徑
similarity_folder = r"F:\laboratory\Graduation_thesis\data\output_similar"  # 修改為similarity檔案存放的資料夾路徑

# 步驟1：找到mask中值為1的位置
locations = find_mask_locations(mask_file)
print(f"找到 {len(locations)} 個位置值為1的座標。")

# 步驟2：讀取所有similarity檔案的座標數值
similarity_results = read_similarity_files(similarity_folder, locations)



# 選擇性輸出或保存結果
for file_name, values in similarity_results.items():
    preview = values[:5]
    if len(values) > 5:
        print(f"{file_name}: {preview}... (還有 {len(values) - 5} 個數值未顯示)")
    else:
        print(f"{file_name}: {preview}")
