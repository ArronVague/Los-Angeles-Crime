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


# month_day_counts = feature["month_day"].value_counts()
# area_counts = feature["area"].value_counts()
# victim_age_counts = feature["victim_age"].value_counts()
# victim_sex_counts = feature["victim_sex"].value_counts()
# victim_descent_counts = feature["victim_descent"].value_counts()

# # Creating subplots
# fig = make_subplots(
#     rows=1,
#     cols=5,
#     subplot_titles=(
#         "Crimes by month and day",
#         "Crimes by area",
#         "Crimes by victim age",
#         "Crimes by victim sex",
#         "Crimes by victim descent",
#     ),
# )


# fig.add_trace(
#     go.Bar(x=month_day_counts.index, y=month_day_counts.values, name="Month and Day"),
#     row=1,
#     col=1,
# )
# fig.add_trace(
#     go.Bar(x=area_counts.index, y=area_counts.values, name="Area"), row=1, col=2
# )
# fig.add_trace(
#     go.Bar(x=victim_age_counts.index, y=victim_age_counts.values, name="Victim Age"),
#     row=1,
#     col=3,
# )
# fig.add_trace(
#     go.Bar(x=victim_sex_counts.index, y=victim_sex_counts.values, name="Victim Sex"),
#     row=1,
#     col=4,
# )
# fig.add_trace(
#     go.Bar(
#         x=victim_descent_counts.index,
#         y=victim_descent_counts.values,
#         name="Victim Descent",
#     ),
#     row=1,
#     col=5,
# )

# fig.update_layout(height=600, width=1000, showlegend=False)
# fig.show()

fig = px.histogram(
    feature, x="victim_age", nbins=30, color_discrete_sequence=["dodgerblue"]
)

fig.update_layout(
    title_text="Distribution of Victim Age",
    xaxis_title_text="Victim Age",
    yaxis_title_text="Frequency",
    bargap=0.2,
    template="plotly_white",
)

# fig.show()

# 用NaN替换无效的年龄（0和负数）
feature["victim_age"] = feature["victim_age"].apply(lambda x: x if x > 0 else None)
feature["victim_age"].replace(0, pd.NA, inplace=True)

fig = px.histogram(
    feature, x="victim_age", nbins=30, color_discrete_sequence=["dodgerblue"]
)

fig.update_layout(
    title_text="Distribution of Victim Age",
    xaxis_title_text="Victim Age",
    yaxis_title_text="Frequency",
    bargap=0.2,
    template="plotly_white",
)

# fig.show()

feature["month"] = data["date_occurred"].dt.month

fig = px.histogram(feature, x="month", nbins=12, color_discrete_sequence=["dodgerblue"])

fig.update_layout(
    title_text="Distribution of Month",
    xaxis_title_text="Month",
    yaxis_title_text="Frequency",
    bargap=0.2,
    template="plotly_white",
)

# fig.show()

# 展示一年中每天的犯罪数量
# nbins=365
# fig = px.histogram(
#     feature, x="month_day", nbins=365, color_discrete_sequence=["dodgerblue"]
# )

# # 横坐标需要显示为 月-日，而不是 年-月
# fig.update_layout(
#     title_text="Distribution of Month and Day",
#     xaxis_title_text="Month and Day",
#     yaxis_title_text="Frequency",
#     bargap=0.2,
#     template="plotly_white",
# )

# fig.show()

feature["month"] = data["date_occurred"].dt.month
feature["day"] = data["date_occurred"].dt.day

daily_crime_counts = (
    feature.groupby(["month", "day"]).size().reset_index(name="crime_count")
)
pivot_table = daily_crime_counts.pivot(
    index="day", columns="month", values="crime_count"
)

monthly_crime_counts = feature["month"].value_counts().sort_index()

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Daily Crime Frequency by Month", "Monthly Crime Distribution"),
)

for month in pivot_table.columns:
    fig.add_trace(
        go.Scatter(
            x=pivot_table.index,
            y=pivot_table[month],
            mode="lines",
            name=str(month),
        ),
        row=1,
        col=1,
    )

fig.add_trace(
    go.Bar(
        x=monthly_crime_counts.index,
        y=monthly_crime_counts.values,
        marker_color="dodgerblue",
    ),
    row=1,
    col=2,
)

fig.update_layout(height=600, width=1200, template="plotly_white", showlegend=True)
fig.update_xaxes(
    title_text="Day",
    row=1,
    col=1,
    tickmode="array",
    tickvals=list(range(1, 32)),
    ticktext=list(range(1, 32)),
)
fig.update_xaxes(title_text="Month", row=1, col=2)
fig.update_yaxes(title_text="Number of Crimes", row=1, col=1)
fig.update_yaxes(title_text="Number of Crimes", row=1, col=2)

# fig.show()

feature["hour"] = data["date_occurred"].dt.hour

hourly_crime_counts = feature["hour"].value_counts().sort_index()

fig = px.bar(
    x=hourly_crime_counts.index,
    y=hourly_crime_counts.values,
    labels={"x": "Hour of the Day (0-23)", "y": "Number of Crimes"},
    color_discrete_sequence=["dodgerblue"],
)

# Updating layout for the plot
fig.update_layout(
    title="Crime Distribution by Hour of the Day",
    template="plotly_white",
    showlegend=False,
)

fig.update_xaxes(
    tickmode="array",
    tickvals=list(range(24)),
    ticktext=[str(hour) for hour in range(24)],
)

# Display the plot
# fig.show()

victim_sex_data = feature["victim_sex"].value_counts()
victim_descent_data = feature["victim_descent"].value_counts()
total_cases = victim_descent_data.sum()

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]])

# Pie plot for victim_sex
fig.add_trace(
    go.Pie(
        labels=victim_sex_data.index,
        values=victim_sex_data,
        title="Victim Sex Distribution",
        textinfo="label+percent",
        insidetextorientation="radial",
    ),
    row=1,
    col=1,
)

# Horizontal bar chart for victim_descent
# 按从大到小的顺序排列
victim_descent_data = victim_descent_data.sort_values(ascending=False)

fig.add_trace(
    go.Bar(
        x=victim_descent_data.values,
        y=victim_descent_data.index,
        orientation="h",
        marker_color="dodgerblue",
        text=[
            f"{count} ({count/total_cases:.2%})" for count in victim_descent_data.values
        ],
        textposition="outside",
    ),
    row=1,
    col=2,
)

# Update layout for the bar chart
fig.update_layout(
    title_text="Victim Sex and Descent Distribution",
    template="plotly_white",
    showlegend=False,
    height=600,
)

fig.update_yaxes(title_text="Number of Cases", row=1, col=2)
fig.update_xaxes(title_text="Victim Descent", row=1, col=2)

# Display the plot
# fig.show()

# Top 10 most common crime descriptions (excluding 'Unknown')
top_crimes = (
    data[data["crime_description"] != "Unknown"]["crime_description"]
    .value_counts()
    .head(10)
)

# Top 10 most common weapons (excluding 'Unknown', 'UNKNOWN WEAPON/OTHER WEAPON')
top_weapons = (
    data[~data["weapon_description"].isin(["Unknown", "UNKNOWN WEAPON/OTHER WEAPON"])][
        "weapon_description"
    ]
    .value_counts()
    .head(10)
)

# Top 10 most common premise_description
top_premises = (
    data[data["premise_description"] != "Unknown"]["premise_description"]
    .value_counts()
    .head(10)
)

# Setting up the figure with two subplots
fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=(
        "Top 10 Crime Descriptions",
        "Top 10 Weapons Used in Crimes",
        "Top 10 Premises",
    ),
)

# Horizontal bar chart for top 10 crime descriptions
fig.add_trace(
    go.Bar(
        x=top_crimes.values,
        y=top_crimes.index,
        orientation="h",
        marker_color="dodgerblue",
    ),
    row=1,
    col=1,
)

# Horizontal bar chart for top 10 weapons used
fig.add_trace(
    go.Bar(
        x=top_weapons.values, y=top_weapons.index, orientation="h", marker_color="coral"
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Bar(
        x=top_premises.values,
        y=top_premises.index,
        orientation="h",
        marker_color="mediumseagreen",
    ),
    row=3,
    col=1,
)

# Update layout for the charts
fig.update_layout(
    height=800,
    showlegend=False,
    template="plotly_white",
    title_text="Top 10 Crime Descriptions, Weapons Used and Premises in Crimes",
)

# Inverting y-axis for both plots to display the highest value at the top
fig.update_yaxes(autorange="reversed", row=1, col=1)
fig.update_yaxes(autorange="reversed", row=2, col=1)
fig.update_yaxes(autorange="reversed", row=3, col=1)

# Update x-axis titles
fig.update_xaxes(title_text="Number of Cases", row=1, col=1)
fig.update_xaxes(title_text="Number of Cases", row=2, col=1)
fig.update_xaxes(title_text="Number of Cases", row=3, col=1)

# Display the plot
fig.show()
