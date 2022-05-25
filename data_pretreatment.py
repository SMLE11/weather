import pandas as pd


def data_pretreatment(file_name):
    df = pd.read_csv(filepath_or_buffer=file_name, nrows=5000)
    df['ds'] = pd.to_datetime(df['ds'])
    cnt = len(df)
    while cnt % 6 != 0:
        cnt = cnt - 1
    for i in range(0, cnt):
        if df.iloc[i, 0].minute == 10:
            ave = df.iloc[i, 1]
            for j in range(0, 5):
                ave += df.iloc[i + j + 1, 1]
            ave /= 6
            df.iloc[i, 1] = ave
    df = df.loc[df['ds'].dt.minute == 10]

    return df
