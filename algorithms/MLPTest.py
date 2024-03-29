import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data_1 = pd.read_csv("tmp_data.csv")

# # Convert date columns to datetime
# data["date_occurred"] = pd.to_datetime(data["date_occurred"])
# # 由于"date_occurred"列没有缺失值，直接操作：
# data["month"] = data["date_occurred"].dt.strftime("%m").astype(float)
# data["day"] = data["date_occurred"].dt.strftime("%d").astype(float)
# data["hour"] = data["date_occurred"].dt.strftime("%H").astype(float)

# # 删除没有用的列
# data.drop(
#     [
#         "date_occurred",
#         "division_number",
#         "date_reported",
#         "area_name",
#         "reporting_district",
#         "part",
#         "crime_description",
#         "modus_operandi",
#         "premise_description",
#         "weapon_description",
#         "status_description",
#         "crime_code_1",
#         "crime_code_2",
#         "crime_code_3",
#         "crime_code_4",
#         "cross_street",
#     ],
#     axis=1,
#     inplace=True,
# )

# # 删除存在缺失值的行
# data.dropna(inplace=True)

# print("Number of rows after removing missing values:", data.shape[0])

# data.to_csv("tmp_data.csv")

le = LabelEncoder()
data_1["victim_sex"] = le.fit_transform(data_1["victim_sex"])
mapping_sex = {index: label for index, label in enumerate(le.classes_)}

data_1["victim_descent"] = le.fit_transform(data_1["victim_descent"])
mapping_descent = {index: label for index, label in enumerate(le.classes_)}

data_1["weapon_code"] = le.fit_transform(data_1["weapon_code"])
mapping_weapon = {index: label for index, label in enumerate(le.classes_)}

data_1["premise_code"] = le.fit_transform(data_1["premise_code"])
mapping_premise_code = {index: label for index, label in enumerate(le.classes_)}

data_1["crime_code"] = le.fit_transform(data_1["crime_code"])
mapping_crime_code = {index: label for index, label in enumerate(le.classes_)}

data_1["status"] = le.fit_transform(data_1["status"])
mapping_status = {index: label for index, label in enumerate(le.classes_)}

# print("lenmmmmmm",data["weapon_code"].unique())


def check(data):
    return data.isnull().sum().sum() == 0


# 进一步处理
def get_train_test_dataset(df_feature,df_label):
    # 类型
    feature = np.array(df_feature).astype(float)
    label = np.array(df_label)
    # 数据集划分
    features_train, features_test, labels_train, labels_test = train_test_split(feature, label, test_size=0.2, random_state=42)
    # 转换为PyTorch张量
    features_train = torch.tensor(features_train).float()
    labels_train =torch.LongTensor(labels_train) #torch.tensor(labels_train).float()
    features_test = torch.tensor(features_test).float()
    labels_test = torch.LongTensor(labels_test)

    # return features_train[:80000], features_test[:20000], labels_train[:80000], labels_test[:20000]
    return features_train, features_test, labels_train, labels_test



feature=[]
label=[]
if check(data_1):
    feature = data_1[
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
    ].copy()
    
    label = data_1["status"].copy()  # , "hour", "crime_code", "premise_code", "weapon_code", "status"


features_train, features_test, labels_train, labels_test = get_train_test_dataset(feature,label)




# 定义多层感知器模型
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super(ImprovedMLP, self).__init__()

        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.BatchNorm1d(hidden_dim))
            self.hidden_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        # x = self.softmax(x)
        return x
    


# 创建 MLP 模型实例
input_dim = 8  # 输入维度

# # weapon_code
# hidden_dims = [16, 32, 64, 64]
# output_dim = 79  # 输出维度

# # premise_code
# hidden_dims = [16, 32, 64, 128, 256]
# output_dim = 305  # 输出维度

# # crime_code
# hidden_dims = [16, 32, 64, 128]
# output_dim = 138  # 输出维度

# status
hidden_dims = [12, 16, 12, 8]
output_dim = 6  # 输出维度

learn_rate = 0.01
dropout_rate = 0.2
model = ImprovedMLP(input_dim, hidden_dims, output_dim,dropout_rate)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

# 打印模型结构
# print(model)

# 创建训练数据集和数据加载器
train_dataset = TensorDataset(features_train, labels_train)
batch_size = 500  # 批处理大小
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(features_test, labels_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 训练循环
num_epochs = 2000 # 训练迭代次数
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for vectors, labels in train_dataloader:

        optimizer.zero_grad()

        logits = model(vectors)

        loss = criterion(logits, labels)
        total_loss += loss.item() #

        loss.backward()
        optimizer.step()
        # total_loss += loss.item() 

    train_loss = total_loss / len(train_dataloader)

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # with torch.no_grad():
    for inputs, labels in test_dataloader:  # 假设test_dataloader是你的测试数据加载器
        # 测试数据
        logits = model(inputs)

        loss = criterion(logits, labels)
        total_loss += loss.item()
        _, predicted = torch.max(logits, dim=1)

        # print(len(predicted))
        correct += (predicted == labels).sum().item()
        total += labels.size(0)


    # 计算平均损失和准确率
    test_loss = total_loss / len(test_dataloader)
    test_accuracy = correct / total

    # 打印训练过程中的损失
    if epoch%60 == 1:
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss} - TestLoss: {test_loss:.4f} - Accuracy: {test_accuracy:.2f}")
    