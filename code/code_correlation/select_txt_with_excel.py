import pandas as pd

def filter_excel_by_txt(txt_file, excel_file, output_file):
    """
    根據 .txt 文件中的數字內容篩選 Excel 檔案，保留匹配的 Subject 資料。
    
    :param txt_file: .txt 文件的路徑，包含需要保留的數字
    :param excel_file: Excel 文件的路徑
    :param output_file: 篩選後的輸出 Excel 文件路徑
    """
    # 讀取 .txt 文件，提取數字列表
    with open(txt_file, "r") as f:
        numbers = [line.strip() for line in f]
    
    # 格式化數字為 VNRxxxx 格式
    valid_subjects = [f"VNR{int(num):04d}" for num in numbers]

    # 讀取 Excel 文件
    df = pd.read_excel(excel_file)

    # 篩選符合條件的 Subject
    filtered_df = df[df["Subject"].isin(valid_subjects)]

    # 保存篩選結果到新的 Excel 文件
    filtered_df.to_excel(output_file, index=False)
    print(f"篩選完成！結果已保存至 {output_file}")

# 使用範例
txt_file = r"F:\laboratory\Graduation_thesis\data\output.txt"       # 修改為 .txt 文件路徑
excel_file = r"F:\laboratory\Graduation_thesis\patient_data\filtered_output.xlsx"        # 修改為 Excel 文件路徑
output_file = r"F:\laboratory\Graduation_thesis\patient_data\filtered_output_ver2.xlsx"  # 修改為輸出的 Excel 文件路徑

filter_excel_by_txt(txt_file, excel_file, output_file)
