import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import nbformat
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


# %config InlineBackend.figure_format = 'retina'

# import data
data = pd.read_csv("Crime_Data_from_2020_to_Present.csv")
# print(len(data))
duplicate_rows = data.duplicated().sum()  # 重复的行数
missing_values = data.isnull().sum()  # 每列的缺失值数量
# print("重复行数：", duplicate_rows)
# print("{:<18} {:<6} {}".format("字段名称", "字段类型", "缺失值数量"))
# for i in range(len(data.columns)):
#     print(
#         "{:<20} {:<10} {}".format(
#             data.columns[i], str(data.dtypes.iloc[i]), str(missing_values.iloc[i])
#         )
#     )

data.head()


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

    # 1、提取相应列
    useful_data = data[
        [
            "date_occurred",
            "month_day",
            "specific_time",
            "area",
            "area_name",
            "victim_age",
            "victim_sex",
            "victim_descent",
            "latitude",
            "longitude",
            "crime_code",
            "crime_description",
            "premise_code",
            "premise_description",
            "weapon_code",
            "weapon_description",
        ]
    ].copy()
    feature = data[
        [
            "month_day",
            "area",
            "victim_age",
            "victim_sex",
            "victim_descent",
            "latitude",
            "longitude",
        ]
    ].copy()
    label = data[["specific_time", "crime_code", "premise_code", "weapon_code"]].copy()

    # 2、填补缺失值
    fill_the_blank(useful_data)
    fill_the_blank(feature)
    fill_the_blank(label)
    fill_the_blank(data)

    # 3、如果检查没有空值则返回数据
    if check(useful_data) and check(feature) and check(label):
        print("空值已处理")
        return useful_data, feature, label

    raise ValueError("Some values are not valid.")


useful_data, feature, label = get_usefulData_feature_label(data)

data["month"] = data["date_occurred"].dt.month
data["day"] = data["date_occurred"].dt.day

# 用NaN替换无效的年龄（0和负数）
mean_age = data["victim_age"].replace({0: None, np.nan: None}).mean()
data["victim_age"].fillna(mean_age, inplace=True)

le = LabelEncoder()
data["victim_sex"] = le.fit_transform(data["victim_sex"])
data["victim_descent"] = le.fit_transform(data["victim_descent"])

X = data[
    [
        "month",
        "day",
        "area",
        "victim_age",
        "victim_sex",
        "victim_descent",
        "latitude",
        "longitude",
    ]
]
y = data[["specific_time", "crime_code", "premise_code", "weapon_code"]]
y = y.astype(str)

mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(y.values)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 创建和训练 RandomForestClassifier 模型
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf_classifier.predict(X_test)

# 计算准确性
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 输出分类报告
classification_rep = classification_report(y_test, y_pred, target_names=mlb.classes_)
print("Classification Report:\n", classification_rep)
