import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from data_pretreatment import data_pretreatment
df = data_pretreatment("t.csv")  # 数据预处理
data_train = df  # 用df作为训练集并命名为data_train
length = len(data_train)  # 读出训练集的长度
head = 0  # 滑动窗口左边
tail = 365 * 1  # 滑动窗口右边
flag = 365 * 1
data_train_cut = data_train[flag:length - 1]  # 第一年是初始滑动窗口，把第一年去掉剩下的是原始数据集，用于最后比较
len_data = len(data_train_cut)  # 原始集长度
res = data_train[0:flag]
for i in range(0, int(len_data / 5)):
    m = Prophet(changepoint_range=0.8
                , changepoint_prior_scale=0.5
                , interval_width=0.8
                , n_changepoints=20
                , yearly_seasonality=True
                , seasonality_prior_scale=50
                , weekly_seasonality=False)  # 创建模型
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    t = res[head:tail]
    t = pd.DataFrame(t, columns=["ds", "y"])
    m.fit(t)  # 将t作为训练集进行拟合
    future = m.make_future_dataframe(periods=5, freq='d')  # 要在长度为365*6的训练集基础上，预测一个长度为5的数据,采用日历日频率来预测
    forecast = m.predict(future)  # 进行预测，得到长度为365*6+5的数据
    forecast = forecast.loc[:, ['ds', 'yhat']]  # 仅取出时间列和数值列
    head += 5  # 滑动窗口进行滑动
    tail += 5
    res = np.vstack((res, forecast[flag:flag + 5]))  # 非首个循环的预测结果水平拼接上之前的res预测集

res_cut = res[flag:length - 1]
len_res_column = res_cut.shape[1]  # 列
len_res_row = res_cut.shape[0]  # 行

fig = plt.figure()  # 开始第一个图，其中有两个子图
# 第一个子图画出预测集
ax1 = fig.add_subplot(211)  # 添加子图，第一个数字‘3’表示有三个行，第一个‘1’表示每一行只画一个图，第二个‘1’表示这是第一个图
ax1.plot(res_cut[:, 0], res_cut[:, len_res_column - 1], color='green', linewidth=0.5)  # 设置绘制的数据，绘制后的颜色和线宽
ax1.set_xlabel("time", color="black")
# 第二个子图画出原始集
ax1 = fig.add_subplot(212)  # 添加子图，第一个数字‘3’表示有三个行，第一个‘1’表
ax1.plot(data_train_cut["ds"], data_train_cut["y"], color='blue', linewidth=0.5)  # 设置绘制的数据，绘制后的颜色和线宽
ax1.set_xlabel("time", color="black")
# 将两个集放在一起画在第二张图上
fig = plt.figure()
plt.plot(res_cut[:, 0], res_cut[:, len_res_column - 1], color='green')
plt.plot(data_train_cut["ds"], data_train_cut["y"], color='blue')

val_theoretical = data_train_cut['y']
val_theoretical = val_theoretical[0:len(res_cut)]
val_actual = res[0:len(val_theoretical), 1]
val_theoretical_mark = (1 - (val_theoretical == 0))


def MAPE(labels, predicts, mask):
    loss = np.abs(predicts - labels) / (np.abs(labels) + 1)
    loss *= mask
    non_zero_len = mask.sum()
    return np.sum(loss) / non_zero_len


MSE = np.mean(np.square(val_theoretical - val_actual))
RMSE = np.sqrt(np.mean(np.square(val_theoretical - val_actual)))
MAE = np.mean(np.abs(val_theoretical - val_actual))
mape = MAPE(val_theoretical, val_actual, val_theoretical_mark)
##MAPE = np.mean(np.abs((val_theoretical_mark - val_actual) / val_theoretical_mark)) * 100
