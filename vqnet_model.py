from pyvqnet.dtype import *
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.tensor import tensor
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from pyvqnet.data.data import data_generator
from pyvqnet.nn.module import Module
from pyvqnet.nn import Linear, ReLu
from pyvqnet.optim.adam import Adam
from pyvqnet.nn.loss import CrossEntropyLoss

from sklearn.preprocessing import StandardScaler, LabelEncoder
from pyvqnet import nn
from pyvqnet import no_grad
import pyvqnet
from pyvqnet import kint64
from pyvqnet import kfloat32
from pyvqnet import kfloat64

def load_data():
    """
    1. 读取数据并转换为张量
    """
    # 加载训练和测试数据
    train_data = pd.read_csv('/home/project/code/train_data_processed.csv')
    test_data = pd.read_csv('/home/project/code/test_data_processed.csv')

    # 特征和标签分离
    X_train = train_data[['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO',
                          'Proximity_to_Industrial_Areas', 'Population_Density']].values
    y_train = train_data['Air Quality'].values

    X_test = test_data[['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO',
                        'Proximity_to_Industrial_Areas', 'Population_Density']].values
    y_test = test_data['Air Quality'].values

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # 标签编码
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # 转换为张量
    X_train_tensor = QTensor(X_train, dtype=kfloat32)
    y_train_tensor = QTensor(y_train, dtype=kint64)
    X_test_tensor = QTensor(X_test, dtype=kfloat32)
    y_test_tensor = QTensor(y_test, dtype=kint64)

    y_train_tensor = y_train_tensor.reshape([-1, 1])
    y_test_tensor = y_test_tensor.reshape([-1, 1])

    # print(X_train_tensor)
    # print(y_train_tensor)
    # print(X_test_tensor)
    # print(y_test_tensor)
    #
    # print(X_train_tensor.shape)
    # print(y_train_tensor.shape)
    # print(X_test_tensor.shape)
    # print(y_test_tensor.shape)



    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor



class AirQualityNN(Module):
    """
    2. 在这里完成初始化神经网络模型的代码
    """
    def __init__(self):
        super(AirQualityNN, self).__init__()

        # 构建神经网络架构
        self.fc1 = nn.Linear(9, 7)  # 输入层到第一个隐藏层
        self.relu = nn.ReLu()  # 激活函数
        self.fc2 = nn.Linear(7, 4)  # 隐藏层到输出层
        self.softmax = nn.Softmax(axis=1)  # 输出层的softmax函数

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def model_train():
    """
    3. 在这里完成训练模型的代码
    """
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_data()

    model = AirQualityNN()  # 初始化模型
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.005)

    # 记录训练和测试损失
    train_losses = []
    test_losses = []

    # 训练过程
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()  # 训练模式
        running_loss = 0.0
        for inputs, labels in data_generator(X_train_tensor.to_numpy(), y_train_tensor.to_numpy(), batch_size=20, shuffle=True):
            optimizer.zero_grad()  # 清除梯度
            outputs = model(inputs)  # 模型预测
            loss = criterion(labels, outputs)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()

        # 计算平均训练损失并记录
        train_loss = running_loss / 200 #4000/20
        train_losses.append(train_loss)

        # 计算测试集损失
        model.eval()
        test_loss = 0.0
        with no_grad():
            for inputs, labels in data_generator(X_test_tensor.to_numpy(), y_test_tensor.to_numpy(), batch_size=20, shuffle=False):
                outputs = model(inputs)
                loss = criterion(labels, outputs)
                test_loss += loss.item()

        # 计算平均测试损失并记录
        test_loss = test_loss / 200
        test_losses.append(test_loss)
        model.train()

        # 打印每个epoch的损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # 保存模型
    pyvqnet.utils.storage.save_parameters(model.state_dict(), "air_quality_model.pth")

    

def model_test():
    """
    4. 使用测试数据集验证模型
    """
    _, _, X_test_tensor, y_test_tensor = load_data()

    model = AirQualityNN()
    model.load_state_dict(pyvqnet.utils.storage.load_parameters('air_quality_model.pth'))  # 加载训练好的模型
    model.eval()  # 设置为评估模式

    y_true = []
    y_pred = []

    with no_grad():
        for inputs, labels in data_generator(X_test_tensor.to_numpy(), y_test_tensor.to_numpy(), batch_size=20, shuffle=False):
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)  # 获取预测类别
            y_true.extend(np.array(labels))
            y_pred.extend(predicted.numpy())

    # 计算准确率和F1分数
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    # 输出结果并保存到txt文件
    with open('model_evaluation.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Average F1 Score: {f1}\n')



if __name__ == "__main__":
    model_train()

    model_test()
