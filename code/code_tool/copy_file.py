import os
import shutil

def create_and_move_folders(source_parent_folder, target_parent_folder):
    """
    將 source_parent_folder 中每個子資料夾內的 T1_to_MNI 資料夾
    複製到 target_parent_folder，並保留原始資料夾結構。

    :param source_parent_folder: 原始父資料夾的路徑（包含 sub-NR1_xxx 資料夾）
    :param target_parent_folder: 目標父資料夾的路徑（將複製 T1_to_MNI 資料夾到此）
    """
    # 確保目標父資料夾存在
    if not os.path.exists(target_parent_folder):
        os.makedirs(target_parent_folder)
    
    # 遍歷原始父資料夾中的子資料夾
    for sub_folder in os.listdir(source_parent_folder):
        if sub_folder.startswith("sub-NR1_") and os.path.isdir(os.path.join(source_parent_folder, sub_folder)):
            source_sub_folder = os.path.join(source_parent_folder, sub_folder, "T1_to_MNI")
            target_sub_folder = os.path.join(target_parent_folder, sub_folder)

            # 檢查 T1_to_MNI 資料夾是否存在
            if os.path.exists(source_sub_folder):
                # 確保目標子資料夾存在
                if not os.path.exists(target_sub_folder):
                    os.makedirs(target_sub_folder)
                
                # 移動 T1_to_MNI 資料夾到目標子資料夾
                shutil.copytree(source_sub_folder, os.path.join(target_sub_folder, "T1_to_MNI"))
                print(f"已複製 {source_sub_folder} 到 {target_sub_folder}")
            else:
                print(f"警告: {source_sub_folder} 不存在，跳過")

    print("所有資料夾處理完成！")

# 使用範例
source_parent_folder = r"F:\laboratory\Graduation_thesis\data\DWI_files"  # 修改為原始父資料夾路徑
target_parent_folder = r"F:\laboratory\Graduation_thesis\data\MNI_CN"        # 修改為目標父資料夾路徑

create_and_move_folders(source_parent_folder, target_parent_folder)
