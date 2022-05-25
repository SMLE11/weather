import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet


def func(FileName, CountryName):
    data_train = pd.read_csv(FileName)
    length = len(data_train)
    data_train_cut = data_train[30:length - 1]
    head = 0
    tail = 29
    len_data = len(data_train_cut)

    for i in range(0, int(len_data / 5)):
        m = Prophet(changepoint_prior_scale=2, n_changepoints=20)
        m.add_country_holidays(country_name=CountryName)
        t = data_train[head:tail + 1]
        m.fit(t)
        future = m.make_future_dataframe(5)
        forecast = m.predict(future)
        head += 5
        tail += 5
        if i == 0:
            res = forecast[30:35]
        else:
            res = np.vstack((res, forecast[30:35]))

    data_train_cut["ds"] = pd.to_datetime(data_train_cut["ds"])
    len_res_column = res.shape[1]
    len_res_row = res.shape[0]

    fig = plt.figure()  # 开始画图

    ax1 = fig.add_subplot(211)  # 添加子图，第一个数字‘3’表示有三个行，第一个‘1’表示每一行只画一个图，第二个‘1’表示这是第一个图
    ax1.plot(res[:, 0], res[:, len_res_column - 1], color=(0.1, 0.1, 1.0), linewidth=0.5)  # 设置绘制的数据，绘制后的颜色和线宽
    ax1.set_xlabel("time", color="black")

    ax1 = fig.add_subplot(212)  # 添加子图，第一个数字‘3’表示有三个行，第一个‘1’表
    ax1.plot(data_train_cut["ds"], data_train_cut["y"], color=(0.1, 0.1, 1.0), linewidth=0.5)  # 设置绘制的数据，绘制后的颜色和线宽
    ax1.set_xlabel("time", color="black")

    fig = plt.figure()
    plt.plot(res[:, 0], res[:, len_res_column - 1])
    plt.plot(data_train_cut["ds"], data_train_cut["y"])

    val_theoretical = data_train_cut.loc[30:30 + len_res_row - 2, 'y']
    val_actual = res[0:len_res_row - 1, len_res_column - 1]

    error = abs(val_theoretical - val_actual)
    error_percentage = error / val_theoretical
    ans = sum(error_percentage) / len(error_percentage) * 100
    print(ans)
