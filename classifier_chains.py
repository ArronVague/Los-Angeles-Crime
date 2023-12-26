import pandas as pd
from sklearn.multioutput import ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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
# y = data[["crime_code", "premise_code", "weapon_code"]]
# 反转y
y = data[["status", "weapon_code"]]
# 将status从不规则的string转化为float
y.loc[y["status"] == "AA", "status"] = 0
y.loc[y["status"] == "AO", "status"] = 1
y.loc[y["status"] == "CC", "status"] = 2
y.loc[y["status"] == "IC", "status"] = 3
y.loc[y["status"] == "JA", "status"] = 4
y.loc[y["status"] == "JO", "status"] = 5
# 将status设置为离散值
y = y.astype(int)

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
