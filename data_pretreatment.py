import pandas as pd


def data_pretreatment(file_name):
    df = pd.read_csv(filepath_or_buffer=file_name, nrows=419184)
    df['ds'] = pd.to_datetime(df['ds'], format='%d.%m.%Y %H:%M:%S')
    length = len(df)
    while length % 144 != 0:
        length -= 1
    cnt = (int)(length / 144)
    for i in range(0, cnt):
        last = i
        if df.iloc[i, 0].minute == 0:
            ave = df.iloc[i, 1]
            for j in range(0, 143):
                ave += df.iloc[i + j + 1, 1]
            ave /= 144
            i += 144
            df.iloc[last, 1] = ave
    df = df.loc[df['ds'].dt.hour == 0]
    df = df.loc[df['ds'].dt.minute == 0]

    return df
