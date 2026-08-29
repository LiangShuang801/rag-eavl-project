import pandas as pd

DATASET_PATH = "./test_dataset/qa_set.csv"

df = pd.read_csv(
    DATASET_PATH,
    encoding="utf-8"
)

print("==============================")
print("测试数据集检查")
print("==============================")

print(f"测试数据总数：{len(df)}")

print()
print("测试类型统计：")

print(
    df["case_type"]
    .value_counts()
)

print()
print("字段检查：")
print(df.columns.tolist())

print()
print("缺失值检查：")
print(
    df.isnull().sum()
)

print()
print("==============================")
print("数据集检查完成")
print("==============================")