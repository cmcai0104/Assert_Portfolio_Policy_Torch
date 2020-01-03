import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def data_preprocess(path):
    dataframe = pd.read_csv(path, index_col=0, header=0)
    dataframe['trade_date'] = dataframe['trade_date'].astype('datetime64')
    dataframe = dataframe[dataframe['trade_date'] <= pd.datetime.strptime('20190809', '%Y%m%d')]
    dataframe = dataframe.set_index('trade_date').fillna(method='ffill', axis=0)
    # 剔除 399016
    colnames = dataframe.columns
    colnames = colnames[[col[:6] != '399016' for col in colnames]]
    dataframe = dataframe[colnames]
    dataframe = dataframe.dropna(axis=0, how='any')
    # 筛选出price列名及其对应的 dataframe
    price_columns = colnames[[col[-5:] == 'close' for col in colnames]]

    price_df = dataframe[price_columns]
    price_df['cash_close'] = 1

    onehot_encode = OneHotEncoder()
    onehot_encode.fit(price_df.columns.values.reshape(-1, 1))
    ret_df = np.log(price_df).diff()
    max_col = ret_df.apply(lambda x: x.argmax(), axis=1).values[1:]
    best_ratio = onehot_encode.transform(max_col.reshape(-1, 1)).toarray()
    dataframe = dataframe.iloc[1:, ]
    return dataframe, best_ratio


df, best_ratio = data_preprocess('./data/create_feature.csv')
df = df.values.astype(np.float32)
best_ratio = best_ratio.astype(np.float32)