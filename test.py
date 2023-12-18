import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import nbformat
from plotly.subplots import make_subplots

# %config InlineBackend.figure_format = 'retina'

# import data
data = pd.read_csv("Crime_Data_from_2020_to_Present.csv")
# print(len(data))
duplicate_rows = data.duplicated().sum()  # 重复的行数
missing_values = data.isnull().sum()  # 每列的缺失值数量
# print("重复行数：", duplicate_rows)
# print("{:<18} {:<6} {}".format("字段名称", "字段类型", "缺失值数量"))
# for i in range(len(data.columns)):
#     print(
#         "{:<20} {:<10} {}".format(
#             data.columns[i], str(data.dtypes.iloc[i]), str(missing_values.iloc[i])
#         )
#     )

data.head()


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

    # 1、提取相应列
    useful_data = data[
        [
            "date_occurred",
            "month_day",
            "specific_time",
            "area",
            "area_name",
            "victim_age",
            "victim_sex",
            "victim_descent",
            "latitude",
            "longitude",
            "crime_code",
            "crime_description",
            "premise_code",
            "premise_description",
            "weapon_code",
            "weapon_description",
        ]
    ].copy()
    feature = data[
        [
            "month_day",
            "area",
            "victim_age",
            "victim_sex",
            "victim_descent",
            "latitude",
            "longitude",
        ]
    ].copy()
    label = data[["specific_time", "crime_code", "premise_code", "weapon_code"]].copy()

    # 2、填补缺失值
    fill_the_blank(useful_data)
    fill_the_blank(feature)
    fill_the_blank(label)

    # 3、如果检查没有空值则返回数据
    if check(useful_data) and check(feature) and check(label):
        print("空值已处理")
        return useful_data, feature, label

    raise ValueError("Some values are not valid.")


useful_data, feature, label = get_usefulData_feature_label(data)

feature.head()

label.head()

data["location"].value_counts()

# print("nbformat version:", nbformat.__version__)

district_crime_counts = (
    feature.groupby("area")
    .agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        counts=("area", "count"),
    )
    .reset_index()
)

fig = px.scatter_mapbox(
    district_crime_counts,
    lat="latitude",
    lon="longitude",
    size="counts",
    color="counts",
    hover_name="area",
    color_continuous_scale="jet",
    hover_data=["counts", "latitude", "longitude"],
    zoom=9,
    height=750,
    width=1_200,
    title="Map of LA Crime Counts by District",
)
fig.update_layout(mapbox_style="open-street-map")
# fig.show()

# Preparing data for bar plots
# status_counts = df['status_description'].value_counts()
# area_counts = df['area_name'].value_counts()
# victim_descent_counts = df['victim_descent'].value_counts()

# # Creating subplots
# fig = make_subplots(rows=1, cols=3, subplot_titles=("Crimes by Status", "Crimes by Area", 'Crimes by Victom Descent'))

month_day_counts = feature["month_day"].value_counts()
area_counts = feature["area"].value_counts()
victim_age_counts = feature["victim_age"].value_counts()
victim_sex_counts = feature["victim_sex"].value_counts()
victim_descent_counts = feature["victim_descent"].value_counts()

# Creating subplots
fig = make_subplots(
    rows=1,
    cols=5,
    subplot_titles=(
        "Crimes by month and day",
        "Crimes by area",
        "Crimes by victim age",
        "Crimes by victim sex",
        "Crimes by victim descent",
    ),
)

# fig.add_trace(go.Bar(x=status_counts.index, y=status_counts.values, name="Status"), row=1, col=1)
# fig.add_trace(go.Bar(x=area_counts.index, y=area_counts.values, name="Area"), row=1, col=2)
# fig.add_trace(go.Bar(x=victim_descent_counts.index, y=victim_descent_counts.values, name="Victim Descent"), row=1, col=3)

# fig.update_layout(height=600, width=1000, showlegend=False)
# fig.show()

fig.add_trace(
    go.Bar(x=month_day_counts.index, y=month_day_counts.values, name="Month and Day"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(x=area_counts.index, y=area_counts.values, name="Area"), row=1, col=2
)
fig.add_trace(
    go.Bar(x=victim_age_counts.index, y=victim_age_counts.values, name="Victim Age"),
    row=1,
    col=3,
)
fig.add_trace(
    go.Bar(x=victim_sex_counts.index, y=victim_sex_counts.values, name="Victim Sex"),
    row=1,
    col=4,
)
fig.add_trace(
    go.Bar(
        x=victim_descent_counts.index,
        y=victim_descent_counts.values,
        name="Victim Descent",
    ),
    row=1,
    col=5,
)

fig.update_layout(height=600, width=1000, showlegend=False)
fig.show()
