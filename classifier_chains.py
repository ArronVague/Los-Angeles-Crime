import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import nbformat
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

data = pd.read_csv("temp.csv")

X = data[
    [
        "month",
        "day",
        "hour",
        "minute",
        "area",
        "victim_age",
        "victim_sex",
        "victim_descent",
        "latitude",
        "longitude",
    ]
]
# y = data[["crime_code", "premise_code", "weapon_code"]]
# 反转y
y = data[["weapon_code", "premise_code", "crime_code"]]

# y = y.drop(["specific_time"], axis=1)

# print(X)
# print(y)
# y = y.astype(str)

# mlb = MultiLabelBinarizer()
# y_encoded = mlb.fit_transform(y.values)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 定义基分类器
base_classifier = DecisionTreeClassifier()

# 定义分类器链
cc = ClassifierChain(base_classifier, order="random", random_state=42)

# 训练模型
cc.fit(X_train, y_train)

# 在测试集上进行预测
predictions = cc.predict(X_test)

# 输出每个标签的分类报告
for i, label in enumerate(y.columns):
    print(f"Classification Report for {label}:")
    print("accuracy:", accuracy_score(y_test[label], predictions[:, i]))
    print(classification_report(y_test[label], predictions[:, i]))
