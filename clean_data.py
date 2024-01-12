import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# import data
data = pd.read_csv("./Crime_Data_from_2020_to_Present.csv")


# 打印行数
print("Number of rows:", data.shape[0])


# Convert date columns to datetime
data["date_occurred"] = pd.to_datetime(data["date_occurred"])
data["month"] = data["date_occurred"].dt.month
data["day"] = data["date_occurred"].dt.day
data["hour"] = data["date_occurred"].dt.hour
data["minute"] = data["date_occurred"].dt.minute

# 删除没有用的列
data.drop(
    [
        "date_occurred",
        "division_number",
        "date_reported",
        "area_name",
        "reporting_district",
        "part",
        "crime_description",
        "modus_operandi",
        "premise_description",
        "weapon_description",
        "status_description",
        "crime_code_1",
        "crime_code_2",
        "crime_code_3",
        "crime_code_4",
        "cross_street",
    ],
    axis=1,
    inplace=True,
)

# 删除存在缺失值的行
data.dropna(inplace=True)



# 提取需要用到的数据：
# 返回三个DataFrame数据集：有用的数据（包括特征列、标签列以及全称（如犯罪描述列））、特征、标签
def get_usefulData_feature_label(data):
    # 用字符串"Unknown"代替object类型的列中的缺失值，-1代替float、int.
    def fill_the_blank(data):
        for column in data.columns:
            if data[column].dtype == "object":
                data[column].fillna("Unknown", inplace=True)
            elif data[column].dtype in ["float64", "int64"]:
                data[column].fillna(-1, inplace=True)

    # 检查数据中是否还有空值
    def check(data):
        return data.isnull().sum().sum() == 0

    # 2、填补缺失值
    fill_the_blank(data)

    # 3、如果检查没有空值则返回数据
    return


get_usefulData_feature_label(data)


print("Number of rows after removing missing values:", data.shape[0])

