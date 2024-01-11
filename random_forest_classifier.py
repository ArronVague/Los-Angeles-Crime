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
import tensorflow as tf


# %config InlineBackend.figure_format = 'retina'

# import data
data = pd.read_csv("Crime_Data_from_2020_to_Present.csv")

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

data.drop(
    [
        "crime_code_4",
        "crime_code_3",
    ],
    axis=1,
    inplace=True,
)

# 打印行数
print("Number of rows:", data.shape[0])
# 删除存在缺失值的行
data.dropna(inplace=True)

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
        "status",
        "status_description",
        "crime_code_1",
        "crime_code_2",
        "location",
        "cross_street",
    ],
    axis=1,
    inplace=True,
)


print("Number of rows after removing missing values:", data.shape[0])


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

# 显示多几行
print(len(X))
print(X)
print(y)

y = y.astype(str)

mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(y.values)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train))
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test))

# batch_size = 32

# train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size)
# test_dataset = test_dataset.batch(batch_size)

# # 创建和训练 RandomForestClassifier 模型
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# num_epochs = 10

# for epoch in range(num_epochs):
#     for batch_data in train_dataset:
#         X_batch, y_batch = batch_data
#         rf_classifier.fit(X_batch, y_batch)

# # rf_classifier.fit(X_train, y_train)

# # 在测试集上进行预测
# # y_pred = rf_classifier.predict(X_test)
# y_pred = []

# for batch_data in test_dataset:
#     X_batch, _ = batch_data
#     y_pred.append(rf_classifier.predict(X_batch))

# y_pred = np.concatenate(y_pred)

# # 计算准确性
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")

# # 输出分类报告
# classification_rep = classification_report(y_test, y_pred, target_names=mlb.classes_)
# print("Classification Report:\n", classification_rep)
