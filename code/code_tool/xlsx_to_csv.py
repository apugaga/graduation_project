import pandas as pd
import os

def xlsx_to_csv(xlsx_path: str, csv_path: str, usecols: list = None, sheet_name=0):
    """
    將指定路徑的 .xlsx 轉成 .csv 並儲存到指定路徑。

    參數：
    - xlsx_path: 輸入 Excel 檔案路徑
    - csv_path:  輸出 CSV 檔案路徑
    - usecols:   (可選) 只保留的欄位（傳欄位名稱或 index list），預設全部欄位
    - sheet_name:要讀取的工作表（預設第 1 張，可用名稱或 index）
    """
    # 讀取 Excel
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl", usecols=usecols)
    # 輸出 CSV
    df.to_csv(csv_path, index=False)
    print(f"Converted {os.path.basename(xlsx_path)} → {os.path.basename(csv_path)}")

if __name__ == "__main__":
    # 請修改下面兩行為你的實際路徑：
    xlsx_path = r"E:\laboratory\Graduation_thesis\data\data_fish\paitient_data\paitient_age_only.xlsx"
    csv_path  =r"E:\laboratory\Graduation_thesis\data\data_fish\paitient_data\paitient_age_only.csv"
    # 如果只想保留第 A、D 欄，可這樣傳入 usecols：
    # usecols = [0, 3]            # 用 index
    # 或 usecols = ["subject_id", "age"]  # 用欄位名稱
    usecols = [0,1]

    xlsx_to_csv(xlsx_path, csv_path, usecols=usecols)
