import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


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
# "weapon_code", "status"
y = data["weapon_code"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardizing the features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

# Making predictions and evaluating the models
log_reg_pred = log_reg.predict(X_test_scaled)

# 计算准确性
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_reg_pred))

# You can also print out classification reports for more detailed performance analysis
print(
    "\nLogistic Regression Classification Report:\n",
    classification_report(y_test, log_reg_pred),
)
