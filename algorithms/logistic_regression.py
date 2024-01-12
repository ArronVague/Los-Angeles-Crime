import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# %config InlineBackend.figure_format = 'retina'

# import data
data = pd.read_csv("temp.csv")

# 将victiom_age中包含0的行删除
# data = data[data["victim_age"] != 0]

X = data[
    [
        "month",
        "day",
        "hour",
        "minute",
        "area",
        "victim_age",
        "victim_sex_enc",
        "victim_descent_enc",
        "latitude",
        "longitude",
        # "premise_code_enc",
        # "location_enc",
        # "weapon_code_enc",
        # "crime_code_enc",
    ]
]

print(X.head())

# crime_code犯罪类型的预测效果不好。
# 尝试将犯罪类型和武器都加入到特征中，将status作为标签
# "weapon_code", "status"
y = data["weapon_code_enc"]


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

features_to_scale = ["victim_age"]

# Standardizing the features (important for logistic regression)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X_train[features_to_scale])

X_train_scaled = np.hstack((X_train.drop(features_to_scale, axis=1), scaled_features))
X_test_scaled = np.hstack(
    (
        X_test.drop(features_to_scale, axis=1),
        scaler.transform(X_test[features_to_scale]),
    )
)

print(X_train_scaled)

# Logistic Regression Model
log_reg = LogisticRegression(solver="sag")
log_reg.fit(X_train_scaled, y_train)

# Making predictions and evaluating the models
log_reg_pred = log_reg.predict(X_test_scaled)

# 计算准确性
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_reg_pred))

# precision = precision_score(y_test, log_reg_pred, average="micro", zero_division=0)
# recall = recall_score(y_test, log_reg_pred, average="micro", zero_division=0)
# f1 = f1_score(y_test, log_reg_pred, average="micro", zero_division=0)

# print("Logistic Regression Precision:", precision)
# print("Logistic Regression Recall:", recall)
# print("Logistic Regression F1:", f1)


# You can also print out classification reports for more detailed performance analysis
print(
    "\nLogistic Regression Classification Report:\n",
    classification_report(y_test, log_reg_pred),
)
