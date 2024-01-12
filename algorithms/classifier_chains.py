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
# y = data["status_enc"]
# y = data[["crime_code_enc", "premise_code_enc", "weapon_code_enc", "status_enc"]]
# 反转y
y = data[["status_enc", "weapon_code_enc", "premise_code_enc", "crime_code_enc"]]

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
