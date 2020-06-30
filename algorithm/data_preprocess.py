import pandas as pd

path = './data/create_feature.csv'


def df_preprocess(path):
    df = pd.read_csv(path, index_col=0, header=0)
    df['trade_date'] = df['trade_date'].astype('datetime64')
    df = df[df['trade_date'] <= pd.datetime.strptime('20190809', '%Y%m%d')]
    df['trade_date'] = df['trade_date'].dt.date
    df = df.set_index('trade_date')
    colnames = df.columns.to_list()
    colnames = list(set(colnames) - set(['000001.SH_pe_y', '000300.SH_pe_y', '000905.SH_pe_y', '399006.SZ_pe_y']))
    colnames = [col for col in colnames if (col[:6] != '399016')]
    df = df[colnames].dropna(axis=0, how='all').fillna(method='ffill', axis=0).dropna(axis=0, how='any')
    for ind in [5, 10, 20, 30, 40, 60, 70, 125, 250, 500, 750]:
        df[[col + '_m' + str(ind) for col in colnames]] = df[colnames].rolling(window=ind, min_periods=1).mean()
        df[[col + '_q' + str(ind) for col in colnames]] = df[colnames].rolling(window=ind, min_periods=1).apply(
            lambda x: len(x[x <= x[-1]]) / len(x), raw=True)
    price_columns = [col for col in colnames if (col[-5:] == 'close')]
    return df, price_columns.to_list()
