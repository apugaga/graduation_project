import os
import re

def extract_numbers_from_filenames(folder_path, output_file):
    """
    從資料夾中提取檔名中的數字部分（xxx）並保存到 .txt 檔案中。
    適用於檔案名稱格式：sub-NR1_xxx_...，其中 xxx 是數字。
    
    :param folder_path: 資料夾路徑
    :param output_file: 輸出的 .txt 檔案路徑
    """
    numbers = []

    # 遍歷資料夾內的所有檔案
    for file_name in os.listdir(folder_path):
        # 檢查檔名是否符合格式 sub-NR1_xxx
        match = re.search(r"sub-NR1_(\d+)", file_name)
        if match:
            numbers.append(int(match.group(1)))

    # 排序數字列表
    numbers = sorted(numbers)

    # 將數字寫入到 .txt 檔案中
    with open(output_file, "w") as f:
        for number in numbers:
            f.write(f"{number}\n")

    print(f"提取完成，共有 {len(numbers)} 個數字，結果已保存至 {output_file}")

# 使用範例
folder_path = r"F:\laboratory\Graduation_thesis\data\output_similar"  # 修改為資料夾路徑
output_file = r"F:\laboratory\Graduation_thesis\data\output.txt"    # 輸出的 .txt 檔案名稱
extract_numbers_from_filenames(folder_path, output_file)
