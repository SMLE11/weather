import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from data_pretreatment import data_pretreatment

df = data_pretreatment("data.csv")  # 数据预处理
data_train = df  # 用df作为训练集并命名为data_train
length = len(data_train)  # 读出训练集的长度
data_train_cut = data_train[30:length - 1]  # 第一个月是初始滑动窗口，把第一个月去掉剩下的是原始数据集，用于最后比较
head = 0  # 滑动窗口左边
tail = 29  # 滑动窗口右边
len_data = len(data_train_cut)  # 原始集长度

for i in range(0, int(len_data / 5)):
    m = Prophet()  # 创建模型
    t = data_train[head:tail + 1]  # t为长度为30的滑动窗口
    m.fit(t)  # 将t作为训练集                                                                                                                                                                                                                                                             ···                                                                                                                                                                                                                                                                                                                                                                                         进行拟合
    future = m.make_future_dataframe(5)  # 要在长度为30的训练集基础上，预测一个长度为5的数据
    forecast = m.predict(future)  # 进行预测，得到长度为30+5的数据
\两个子图
# 第一个子图画出预测集
ax1 = fig.add_subplot(211)  # 添加子图，第一个数字‘3’表示有三个行，第一个‘1’表示每一行只画一个图，第二个‘1’表示这是第一个图
ax1.plot(res[:, 0], res[:, len_res_column - 1], color='green', linewidth=0.5)  # 设置绘制的数据，绘制后的颜色和线宽
ax1.set_xlabel("time", color="black")
# 第二个子图画出原始集
ax1 = fig.add_subplot(212)  # 添加子图，第一个数字‘3’表示有三个行，第一个‘1’表
ax1.plot(data_train_cut["ds"], data_train_cut["y"], color='blue', linewidth=0.5)  # 设置绘制的数据，绘制后的颜色和线宽
ax1.set_xlabel("time", color="black")
# 将两个集放在一起画在第二张图上
fig = plt.figure()
plt.plot(res[:, 0], res[:, len_res_column - 1], color='green')
plt.plot(data_train_cut["ds"], data_train_cut["y"], color='blue')

val_theoretical = data_train_cut['y']
val_theoretical = val_theoretical[0:len(res)]
val_actual = res[0:len(val_theoretical) , 1]

error = abs(val_theoretical - val_actual)
error_percentage = error / val_theoretical
ans = sum(error_percentage) / len(error_percentage) * 100
print(ans)
