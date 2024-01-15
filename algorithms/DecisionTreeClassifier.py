import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# %config InlineBackend.figure_format = 'retina'

# import data
data = pd.read_csv("temp.csv")

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

# print(X.head())

# crime_code犯罪类型的预测效果不好。
# 尝试将犯罪类型和武器都加入到特征中，将status作为标签
# "crime_code_enc", "premise_code_enc", "weapon_code_enc", "status_enc"
y = data["status_enc"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建和训练 DecisionTreeClassifier 模型
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = dt_classifier.predict(X_test)

# 计算准确性
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 输出分类报告
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)
