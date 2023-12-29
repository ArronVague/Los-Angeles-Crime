import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

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
        "victim_sex",
        "victim_descent",
        "latitude",
        "longitude",
        "premise_code",
    ]
]
# "weapon_code", "status"
# crime_code的准确率实在是太低了，所以就不用了
y = data["status"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardizing the features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建和训练 KNeighborsClassifier 模型
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)

# 在测试集上进行预测
y_pred = knn_classifier.predict(X_test_scaled)

# 计算准确性
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 输出分类报告
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)
