import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def data_preprocess(data_csv):
    """
    1. 对数据的预处理，获取训练集和测试集数据的全部特征
    """
    data = pd.read_csv(data_csv)
    print("数据维度：", data.shape)
    print("\n列名：", data.columns.tolist())
    print("\n每列缺失值数量：\n", data.isnull().sum())

    # 找出所有有缺失值的行和列
    missing_data = data[data.isnull().any(axis=1)]
    if not missing_data.empty:
        print("\n存在缺失值的具体行：")
        print(missing_data)
    else:
        print("\n没有缺失值的记录。")

    # 替换数值型特征中的负值为0
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    num_neg_before = (data[numeric_cols] < 0).sum().sum()
    print(f"负值总数量（替换前）：{num_neg_before}")
    data[numeric_cols] = data[numeric_cols].clip(lower=0)
    num_neg_after = (data[numeric_cols] < 0).sum().sum()
    print(f"负值总数量（替换后）：{num_neg_after}")

    # 输出数值型特征统计信息
    numeric_data = data[numeric_cols]

    # 生成describe信息
    desc = numeric_data.describe()

    # 计算方差并加到describe表格里
    desc.loc['var'] = numeric_data.var()

    print("\n数值的统计信息:\n", desc)

    # 生成新文件名并保存
    new_csv = data_csv.replace('.csv', '_processed.csv')
    data.to_csv(new_csv, index=False)
    print(f"处理后的数据已保存为新文件：{new_csv}")


if __name__ == "__main__":
    train_data_csv = '/home/project/code/train_data.csv'  # 替换为你的数据路径
    test_data_csv = '/home/project/code/test_data.csv'
    data_preprocess(train_data_csv)
    data_preprocess(test_data_csv)