import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier


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
        "victim_sex",
        "victim_descent",
        "latitude",
        "longitude",
        "premise_code",
    ]
]
# "crime_code", "weapon_code"
y = data["crime_code"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sgd_classifier = SGDClassifier(random_state=42)
sgd_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = sgd_classifier.predict(X_test)

# 计算准确性
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 输出分类报告
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)
