import os

import pandas as pd

from ichimoku import datadir, nameStrategy, Strategy, getDataByTscode

s = Strategy()

def backtest_basic():
    targetList = read_excel(nameStrategy('strategy1'))
    for i in targetList[0]:
        id = i.split(' ')[0]
        name = i.split(' ')[1]
        index = 0
        data = getDataByTscode(id, 1)
        while index < len(list):
             print('start to calculation ' + list[0])


def read_excel(name):
    filedir = os.path.join(datadir, name)
    return pd.read_excel(filedir)

if __name__ == '__main__':
    backtest_basic();