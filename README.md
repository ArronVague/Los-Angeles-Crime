# Los-Angeles-Crime

机器学习实验课大作业

## 环境

Python 3.9.17

## 数据集

[Los Angeles Crime Data 2020-2023](https://www.kaggle.com/datasets/asaniczka/crimes-in-los-angeles-2020-2023/data)

特征

> "division_number","date_reported","date_occurred","area","area_name","reporting_district","part","crime_code","crime_description","modus_operandi","victim_age","victim_sex","victim_descent","premise_code","premise_description","weapon_code","weapon_description","status","status_description","crime_code_1","crime_code_2","crime_code_3","crime_code_4","location","cross_street","latitude","longitude"

“部门编号”、“报告日期”、“发生日期”、“区域”、“区域名称”、“报告地区”、“部分”、“犯罪代码”、“犯罪描述”、“作案方式”、“受害者年龄”、“受害者性别”、“受害者血统”、"前提代码"、"前提描述"、"武器代码"、"武器描述"、"状态"、"状态描述"、"犯罪代码1"、"犯罪代码2"、"犯罪代码3"、"犯罪代码4"、"位置"、"交叉街道"、"纬度 “，“经度”

## 特征

~~date_occurred 发生日期~~（实际上这个不好编码）

- ~~month_day~~
  - ~~month~~
  - ~~day~~

month 月份（由date_occured拆分而来）

day 日期（由date_occured拆分而来）

hour（由date_occured拆分而来）

minute（由date_occured拆分而来）

area (area_name) 地区

victim_age 受害者年龄

victim_sex 受害者性别

victim_descent 受害者血统

latitude 纬度坐标

longitude 经度坐标

## 标签

~~specific_time 具体时间（如01:00，由date_occured拆分而来）~~（效果极差）

crime_code (crime_descroption) 犯罪描述（训练效果差）

premise_code (premise_description) 遇害地点（如酒店、夜总会等）

weapon_code (weapon_description) 武器（训练效果较好）

status (status_descroption) 案件状态（训练效果好）

## 没用的特征

~~division_number 编号~~

~~date_reported 报告日期~~

~~reporting_district报告地点~~

~~part 犯罪事件的部分号~~

~~modus_operandi 作案手法~~

crime_code_1/2/3/4 犯罪编号

location 详细地址

cross_street 临近街道

## 算法汇总

```python
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
    ]
]
# "crime_code", "premise_code", "weapon_code"
y = data["weapon_code"]
```

|            | crime_code | premise_code | weapon_code    |
| ---------- | ---------- | ------------ | -------------- |
| knn        | 0.21       | 0.27         | 0.63           |
| 朴素贝叶斯 | 0.02       | 0.04         | 0.23           |
| 决策树     | 0.20       | 0.29         | 0.54           |
| 链         | 0.20       | 0.27         | 0.54           |
| svm        |            |              | 0.73（2w数据） |

```python
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
```

|           | crime_code | weapon_code |
| --------- | ---------- | ----------- |
| knn       | 0.27       | 0.65        |
| SGD       | 0.17       | 0.38        |
| LinearSVC |            | 0.65/0.54   |
|           |            |             |
|           |            |             |

```python
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
        "premise_code",
    ]
]
# "crime_code", "weapon_code"
y = data["crime_code"]
```

|      | crime_code | weapon_code |
| ---- | ---------- | ----------- |
| knn  | 0.27       | 0.65        |
|      |            |             |
|      |            |             |
|      |            |             |
|      |            |             |

```python
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
```

|            | weapon_code                 | status                  |
| ---------- | --------------------------- | ----------------------- |
| knn        | 0.65/0.47（删除空值数据后） | 0.76/0.54（删除空值后） |
| 朴素贝叶斯 | 0.21/0.23                   | 0.12/0.06               |
| 决策树     | 0.56/0.36                   | 0.68/0.51               |
| 链         | 0.56/0.36                   | 0.68/0.51               |
| 逻辑回归   | /0.55                       | /0.60                   |

## 参与贡献

1. clone仓库
2. 以main branch为基础new branch
3. 在新建分支上编写代码
4. commit代码到本地
5. publish branch到仓库
6. 创建pull request
7. 经审核人员审核后merge到main branch

忽略中英文表达 :triumph: 。

## 参考文献

[Los Angeles Crime Data Quick EDA 🦹🏼‍♂️](https://www.kaggle.com/code/guslovesmath/los-angeles-crime-data-quick-eda)

- 每个区域犯罪分布

- 犯罪状态统计

- 受害者血统

- 区域

[CrimeSolver Predictor](https://www.kaggle.com/code/safronov00/crimesolver-predictor#2.-Clean-Data)

- data overview
  - victim age
  - monthly crime（准备做成365天的，不看年份）
  - hour of the day
  - victim sex and descent distribution
  - top 10 crime descriptions and weapons used in crimes
