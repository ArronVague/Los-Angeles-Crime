import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# import data
data = pd.read_csv("../Crime_Data_from_2020_to_Present.csv")

# 除了这些列，其他列都删除
# "date_occurred",
# "area",
# "crime_code",
# "victim_age",
# "victim_sex",
# "victim_descent",
# "premise_code",
# "weapon_code",
# "latitude",
# "longitude",

# data.drop(
#     [
#         "crime_code_3",
#         "crime_code_4",
#     ],
#     axis=1,
#     inplace=True,
# )

# 打印行数
print("Number of rows:", data.shape[0])
# 删除存在缺失值的行
# data.dropna(inplace=True)

# 删除没有用的列
data.drop(
    [
        "division_number",
        "date_reported",
        "area_name",
        "reporting_district",
        "part",
        "crime_description",
        "modus_operandi",
        "premise_description",
        "weapon_description",
        # "status",
        "status_description",
        "crime_code_1",
        "crime_code_2",
        "crime_code_3",
        "crime_code_4",
        # "location",
        "cross_street",
    ],
    axis=1,
    inplace=True,
)

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

    # Convert date columns to datetime
    data["date_occurred"] = pd.to_datetime(data["date_occurred"])
    data["month_day"] = data["date_occurred"].dt.strftime("%m-%d")  # 月日
    data["specific_time"] = data["date_occurred"].dt.strftime("%H:%M:%S")  # 时分秒

    # 2、填补缺失值
    fill_the_blank(data)

    # 3、如果检查没有空值则返回数据
    return


get_usefulData_feature_label(data)

data["month"] = data["date_occurred"].dt.month
data["day"] = data["date_occurred"].dt.day
data["hour"] = data["date_occurred"].dt.hour
data["minute"] = data["date_occurred"].dt.minute

# 用NaN替换无效的年龄（0和负数）
mean_age = data["victim_age"].replace({0: None, np.nan: None}).mean()
data["victim_age"].fillna(mean_age, inplace=True)

le = LabelEncoder()
data["victim_sex_enc"] = le.fit_transform(data["victim_sex"])
data["victim_descent_enc"] = le.fit_transform(data["victim_descent"])
data["crime_code_enc"] = le.fit_transform(data["crime_code"])
data["premise_code_enc"] = le.fit_transform(data["premise_code"])
data["weapon_code_enc"] = le.fit_transform(data["weapon_code"])
data["status_enc"] = le.fit_transform(data["status"])
data["location_enc"] = le.fit_transform(data["location"])

# data["victim_sex"] = le.inverse_transform(data["victim_descent_enc"])
# data["victim_descent"] = le.inverse_transform(data["victim_descent_enc"])
# data["crime_code"] = le.inverse_transform(data["crime_code_enc"])
# data["premise_code"] = le.inverse_transform(data["premise_code_enc"])
# data["weapon_code"] = le.inverse_transform(data["weapon_code_enc"])
# data["status"] = le.inverse_transform(data["status_enc"])
# data["location"] = le.inverse_transform(data["location_enc"])

data.drop(
    [
        "date_occurred",
        "month_day",
        "specific_time",
    ],
    axis=1,
    inplace=True,
)

# 删除victim_age中包含0的行
data = data[data["victim_age"] != 0]

print("Number of rows after removing missing values:", data.shape[0])


# 将data写入temp.csv
data.to_csv("temp.csv")
