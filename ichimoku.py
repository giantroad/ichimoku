# -*- coding: utf-8 -*-
import os
import sys
import time

import openpyxl as openpyxl
import pandas
import pandas as pd
import tushare as ts
import numpy as np
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import mplfinance as mpf
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMessageBox
from dateutil.relativedelta import relativedelta
from mpl_finance import candlestick_ohlc, candlestick2_ohlc
import numpy as np
import decimal
import sys

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QDialog, QApplication
from primodial import Ui_Dialog
from numpy import long

# author : ye
mode = 1
fig, ax = plt.subplots()
datadir = './data/'
strategydir = './strategy/'
financialdir = './financialdata/'
x, y, lastday, xminnow, xmaxnow = 1, 1, 0, 0, 0


# 云层细代表震荡，越来越细改变的趋势也越大，要看有没有最高点

# to avoid data collection, change return value to suffix of the file in 'data' dictionary -> enter offline mode!
def endDate():
    return time.strftime('%Y%m%d')


# return '20210818'


# 1:excel 0:tushare
def getDataByTscode(ts_code, mode):
    if mode == 1:
        filedir = os.path.join(datadir, nameStrategy(ts_code))
        byexcel = pd.read_excel(filedir)
        return byexcel
    if mode == 0:
        ts.set_token('30f769d97409f6b9ff133558703d4cbe8302b4e6452330b2c11af044')
        pro = ts.pro_api()
        t1 = endDate()
        t2 = (datetime.now() - relativedelta(years=1)).strftime('%Y%m%d')
        df = pro.daily(ts_code=ts_code, start_date=t2, end_date=t1)
        df = df.iloc[::-1]
        return df


def nameStrategy(code):
    if mode == 1:
        filelist = os.listdir(datadir)
        for i in filelist:
            if i.__contains__(code):
                return i
    return code + '-' + endDate() + '.xlsx'

def nameStrategyfortxt(code):
    return code +  '.txt'


def vision(data, ts_name):
    ichimoku = Ichimoku(data)
    ichimoku.run()
    ichimoku.plot(ts_name)


def call_back(event):
    axtemp = event.inaxes
    x_min, x_max = axtemp.get_xlim()
    fanwei = (x_max - x_min) / 10
    if event.button == 'up':
        axtemp.set(xlim=(x_min + fanwei, x_max - fanwei))
    elif event.button == 'down':
        axtemp.set(xlim=(x_min - fanwei, x_max + fanwei))
    fig.canvas.draw_idle()


def button_press_callback(click):
    global x
    global y
    x = click.xdata
    y = click.ydata
    point = (click.xdata, click.ydata)
    print(point)


def motion_notify_callback(event):
    global x, xminnow, xmaxnow
    if event.button != 1: return
    xnow = event.xdata
    print(x)
    delta = x - xnow
    plt.xlim(xmin=xminnow + delta, xmax=xmaxnow + delta)
    xminnow = xminnow + delta
    xmaxnow = xmaxnow + delta
    x = xnow
    point = (event.xdata, event.ydata, xminnow, xmaxnow)
    print(point)
    fig.canvas.draw_idle()


class Ichimoku():
    """
    @param: ohcl_df <DataFrame>

    Required columns of ohcl_df are:
        Date<Float>,Open<Float>,High<Float>,Close<Float>,Low<Float>
    """

    def __init__(self, ohcl_df):
        self.ohcl_df = ohcl_df
        ohcl_df['trade_date'] = pandas.to_datetime(ohcl_df['trade_date'].astype(str))

    def run(self):
        tenkan_window = 9
        kijun_window = 26
        senkou_span_b_window = 52
        cloud_displacement = 26
        chikou_shift = -26
        ohcl_df = self.ohcl_df

        # Dates are floats in mdates like 736740.0
        # the period is the difference of last two dates
        last_date = ohcl_df["trade_date"].iloc[-1].date()
        period = 1

        # Add rows for N periods shift (cloud_displacement)

        ext_beginning = last_date + timedelta(days=1)
        ext_end = last_date + timedelta(days=((period * cloud_displacement) + period))
        dates_ext = pd.date_range(start=ext_beginning, end=ext_end)
        dates_ext_df = pd.DataFrame({"trade_date": dates_ext})
        dates_ext_df.index = dates_ext  # also update the df index
        ohcl_df = ohcl_df.append(dates_ext_df)

        # Tenkan
        tenkan_sen_high = ohcl_df["high"].rolling(window=tenkan_window).max()
        tenkan_sen_low = ohcl_df["low"].rolling(window=tenkan_window).min()
        ohcl_df['tenkan_sen'] = (tenkan_sen_high + tenkan_sen_low) / 2
        # Kijun
        kijun_sen_high = ohcl_df["high"].rolling(window=kijun_window).max()
        kijun_sen_low = ohcl_df["low"].rolling(window=kijun_window).min()
        ohcl_df['kijun_sen'] = (kijun_sen_high + kijun_sen_low) / 2
        # Senkou Span A
        ohcl_df['senkou_span_a'] = ((ohcl_df['tenkan_sen'] + ohcl_df['kijun_sen']) / 2).shift(cloud_displacement)
        # Senkou Span B
        senkou_span_b_high = ohcl_df["high"].rolling(window=senkou_span_b_window).max()
        senkou_span_b_low = ohcl_df["low"].rolling(window=senkou_span_b_window).min()
        ohcl_df['senkou_span_b'] = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(cloud_displacement)
        # Chikou
        ohcl_df['chikou_span'] = ohcl_df["close"].shift(chikou_shift)
        self.ohcl_df = ohcl_df
        ohcl_df['trade_date'] = mdates.date2num(ohcl_df['trade_date'])
        ohcl_df.index = ohcl_df['trade_date']

        ohcl_df['MA10'] = ohcl_df['close'].rolling(10).mean()
        ohcl_df['MA5'] = ohcl_df['close'].rolling(5).mean()
        ohcl_df['MA20'] = ohcl_df['close'].rolling(20).mean()
        return ohcl_df

    def plot(self, ts_name):
        global xminnow, xmaxnow
        fig.canvas.mpl_connect('scroll_event', call_back)
        fig.canvas.mpl_connect('button_press_event', button_press_callback)
        fig.canvas.mpl_connect('motion_notify_event', motion_notify_callback)
        self.plot_candlesticks(fig, ax)
        self.plot_ichimoku(fig, ax)
        self.pretty_plot(fig, ax, ts_name)
        plt.xlim(xmin=lastday - 200, xmax=lastday)
        xminnow = lastday - 200
        xmaxnow = lastday

        plt.xlim(xmin=xminnow + 80, xmax=xmaxnow + 80)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

    def pretty_plot(self, fig, ax, ts_name):
        ax.legend()
        fig.autofmt_xdate()
        ax.set_xticks(range(len(self.ohcl_df['trade_date'])))
        d = mdates.num2date(self.ohcl_df['trade_date'])
        for i in range(0, len(d)):
            d[i] = datetime.strftime(d[i], '%m-%d')

        ax.set_xticklabels(d)
        ax.xaxis_date()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

        # Chart info
        title = ts_name
        bgcolor = '#f0f0f0'
        grid_color = '#363c4e'
        spines_color = '#0f0f0f'
        # Axes
        plt.title(title, color='black', fontproperties="SimHei")

        ax.set_facecolor(bgcolor)
        ax.grid(linestyle='-', linewidth='0.5', color=grid_color, alpha=0.4)
        ax.yaxis.tick_right()
        ax.set_yscale("linear")
        fig.patch.set_facecolor(bgcolor)
        fig.patch.set_edgecolor(bgcolor)
        plt.rcParams['figure.facecolor'] = bgcolor
        plt.rcParams['savefig.facecolor'] = bgcolor
        ax.spines['bottom'].set_color(spines_color)
        ax.spines['top'].set_color(spines_color)
        ax.spines['right'].set_color(spines_color)
        ax.spines['left'].set_color(spines_color)
        ax.tick_params(axis='x', colors=spines_color, size=7)
        ax.tick_params(axis='y', colors=spines_color, size=7)
        fig.tight_layout()
        ax.autoscale_view()

    def plot_ichimoku(self, fig, ax, view_limit=100):
        d2 = self.ohcl_df.loc[:,
             ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 'MA5', 'MA10', 'MA20']]
        #  d2 = d2.tail(view_limit)
        date_axis = range(0, len(d2))
        # ichimoku
        plt.plot(date_axis, d2['tenkan_sen'], label="tenkan", color='#004200', alpha=1, linewidth=2)
        plt.plot(date_axis, d2['kijun_sen'], label="kijun", color="#721d1d", alpha=1, linewidth=2)
        plt.plot(date_axis, d2['senkou_span_a'], label="span a", color="#008000", alpha=0.65, linewidth=0.5)
        plt.plot(date_axis, d2['senkou_span_b'], label="span b", color="#ff0000", alpha=0.65, linewidth=0.5)
        plt.plot(date_axis, d2['chikou_span'], label="chikou", color="black", alpha=0.65, linewidth=0.5)
        plt.plot(date_axis, d2['MA5'], label="MA5", color="green", alpha=0.8, linewidth=0.6)
        plt.plot(date_axis, d2['MA10'], label="MA10", color="blue", alpha=0.8, linewidth=1.2)
        plt.plot(date_axis, d2['MA20'], label="MA20", color="yellow", alpha=0.8, linewidth=0.6)
        # green cloud
        ax.fill_between(date_axis, d2['senkou_span_a'], d2['senkou_span_b'],
                        where=d2['senkou_span_a'] > d2['senkou_span_b'], facecolor='#008000',
                        alpha=0.25)
        # red cloud
        ax.fill_between(date_axis, d2['senkou_span_a'], d2['senkou_span_b'],
                        where=d2['senkou_span_b'] > d2['senkou_span_a'], facecolor='#ff0000',
                        alpha=0.25)

    def plot_candlesticks(self, fig, ax, view_limit=10):
        # plot candlesticks

        candlesticks_df = self.ohcl_df.loc[:, ['trade_date', "open", "high", "low", "close"]]
        #  candlesticks_df = candlesticks_df.tail(view_limit)
        # plot candlesticks
        #   candlesticks_df['trade_date'] = mdates.date2num(candlesticks_df['trade_date'])
        # candlestick_ohlc(ax, candlesticks_df.values, width=0.5, colorup='#83b987', colordown='#eb4d5c', alpha=0.5)

        candlestick2_ohlc(ax, candlesticks_df['open'], candlesticks_df['high'], candlesticks_df['low'],
                          candlesticks_df['close'], width=0.6, colorup='#83b987', colordown='#eb4d5c', alpha=1)

    # mpf.plot(candlesticks_df, width=0.6, colorup='#83b987', colordown='#eb4d5c', alpha=0.5)

    # Range generator for decimals
    def drange(self, x, y, jump):
        while x < y:
            yield float(x)
            x += decimal.Decimal(jump)


class DialogDemo(QDialog, Ui_Dialog):

    def __init__(self, shares, strategy, parent=None):
        self.ashares = shares
        self.allshares = shares
        self.strategy = strategy
        super(DialogDemo, self).__init__(parent)
        self.setupUi(self)

    def ichimoku_push(self):
        sharesId = self.share.split(' ')[0]
        t1 = endDate()
        t2 = (datetime.now() - relativedelta(years=1)).strftime('%Y%m%d')
        self.ichimokuplot(sharesId, self.share, t2, t1)

    def strategy_push(self):
        stext = self.comboBox.currentText()
        self.listWidget.clear()
        if stext == 'All stocks':
            self.listWidget.addItems(self.ashares[1] + ' ' + self.ashares[0])
            self.ashares = self.allshares
        if stext == 'ichimoku strategy':
            df = self.strategy.strategy_prediction()
            self.listWidget.addItems(df[0])
            df[1] = df[0].apply(lambda x: x.split(' ')[1])
            df[0] = df[0].apply(lambda x: x.split(' ')[0])
            self.ashares = df


    def list_click(self, item):
        self.share = item.text()
        sharesId = self.share.split(' ')[0]
        self.textBrowser.setText(getfinancialdatafromlocal(sharesId))
        print(self.share)

    def double_click(self):
        sharesId = self.share.split(' ')[0]
        t1 = endDate()
        t2 = (datetime.now() - relativedelta(years=1)).strftime('%Y%m%d')
        self.ichimokuplot(sharesId, self.share, t2, t1)

    def text_edit_search(self):
        text = self.shareSearch.text()
        ls = self.ashares
        ls = ls[ls[0].str.contains(text) | ls[1].str.contains(text)]
        self.listWidget.clear()
        self.listWidget.addItems(ls[1] + ' ' + ls[0])

    def reset_push(self):
        global  mode
        mode = 0
        if not os.path.exists(datadir):
            os.mkdir(datadir)
        if not os.path.exists(financialdir):
            os.mkdir(financialdir)
        if not os.path.exists(strategydir):
            os.mkdir(strategydir)
        ashares = getProducts()
        ashares.reset_index(drop=True, inplace=True)
        getdailyData(ashares)
        # getFinancialData(ashares)
        s = Strategy()
        s.strategy_prediction()
        mode = 1

    def createDialog(self):
        app = QApplication(sys.argv)
        # 创建对话框
        # 显示对话框
        self.listWidget.addItems(self.ashares[1] + ' ' + self.ashares[0])
        # binding
        self.listWidget.itemClicked.connect(self.list_click)
        self.pushButton_2.clicked.connect(self.ichimoku_push)
        self.pushButton_3.clicked.connect(self.reset_push)
        self.shareSearch.textChanged.connect(self.text_edit_search)
        self.listWidget.doubleClicked.connect(self.double_click)
        self.listWidget.doubleClicked.connect(self.double_click)
        self.pushButton.clicked.connect(self.strategy_push)
        self.comboBox.addItem('All stocks')
        self.comboBox.addItem('ichimoku strategy')
        self.comboBox.addItem('trend tracking strategy')
        self.show()
        sys.exit(app.exec_())

    def ichimokuplot(self, ts_code, ts_name, start_date, end_date):
        global lastday
        global fig, ax
        plt.close(fig)
        fig, ax = plt.subplots()
        df = getDataByTscode(ts_code, mode)
        lastday = len(df)
        vision(df, ts_name)


class Strategy:

    def __init__(self):
        self.sl = getProducts()
        self.sl.reset_index(drop=True, inplace=True)

    def getIchimoku(self, s):
        data = getDataByTscode(s, 1)
        print(s + " doing calculating")
        if len(data) == 0: return
        ichimoku = Ichimoku(data)
        return ichimoku.run()

    def basic_stategy(self, start_index, end_index, i):
        if len(i[(i['chikou_span'].isna()) & (~i['open'].isna())]) == 0: return False
        lastDaysData = i[(i['chikou_span'].isna()) & (~i['open'].isna())]
        lastdayDataInrange = lastDaysData.iloc[start_index:]
        l = lastdayDataInrange;
        ##########Basic principle##############
        # 1. tenkan and kijun cross
        if ~((l.iloc[start_index]['tenkan_sen'] <= l.iloc[start_index]['kijun_sen']) & (
                l.iloc[end_index]['kijun_sen'] <= l.iloc[end_index]['tenkan_sen'])):
            return False
        # 2. Beyond the cloud
        if ~((l.iloc[end_index]['senkou_span_a'] > l.iloc[end_index]['senkou_span_b']) & (
                l.iloc[end_index]['close'] >= min(l.iloc[end_index]['senkou_span_a'], l.iloc[end_index]['senkou_span_b']))):
            return False
        return True
        ##########Basic principle##############



    # ichimoku strategy -> seeking Low priced stocks with potential
    def strategy_prediction(self):
        filelist = os.listdir(datadir)
        if filelist.__contains__(nameStrategy('strategy1')):
            return pd.read_excel(datadir + nameStrategy('strategy1'))
        sl = self.sl
        res = []
        range = 7
        for s in sl[1]:
            try:
                i = self.getIchimoku(s)
                if self.basic_stategy(-1-range, -1, i):
                    print(s + "looks good")
                    res.append(s + " " + sl[sl[1] == s][0].iloc[0])
            except:
                print(s + " is wrong during calculating")
        df = pd.DataFrame(res)
        filelist = os.listdir(datadir)
        for i in filelist:
            if i.__contains__('strategy1'): os.remove(os.path.join(strategydir, i))
        df.to_excel(datadir + nameStrategy('strategy1'),  index=False)
        return df

    # # average strategy
    # def strategy2(self):
    #     filelist = os.listdir(strategydir)
    #     if filelist.__contains__(nameStrategy('strategy2')):
    #         return pd.read_excel(strategydir + nameStrategy('strategy2'), index_col=0)
    #     sl = self.sl
    #     res = []
    #     for s in sl[1]:
    #         data = getDataByTscode(s, 1)
    #         print(s)
    #         if len(data) == 0: continue
    #         ichimoku = Ichimoku(data)
    #         i = ichimoku.run()
    #         if len(i[(i['chikou_span'].isna()) & (~i['open'].isna())]) == 0: continue
    #         ldd = i[(i['chikou_span'].isna()) & (~i['open'].isna())].iloc[-1]  # lastdaydata
    #         #
    #         if ldd['tenkan_sen'] < ldd['kijun_sen']: continue
    #         #
    #         smin = min(ldd['senkou_span_a'], ldd['senkou_span_b'])
    #         if ldd['high'] < smin: continue
    #         #
    #         if len(i[~i['chikou_span'].isna()]) == 0: continue
    #         ckd = i[~i['chikou_span'].isna()].iloc[-1]  # chikouData
    #         if ckd['chikou_span'] < ckd['high']: continue
    #
    #         # average
    #         data['MA10'] = data['close'].rolling(10).mean()
    #         data['MA100'] = data['close'].rolling(100).mean()
    #         data['MA10diff'] = data['MA10'].diff()
    #         # volatility
    #         data['std10'] = data['close'].rolling(10).std(ddof=0).rolling(10).mean()
    #         x = -35
    #         if len(data) <= -x: continue
    #         MAdata = data[x:-1]
    #         xx = -x - 2
    #         #
    #         data60 = data[-61:-1]
    #         data180 = data[-181:-1]
    #         data250 = data[-251:-1]
    #         min60 = data60['low'].min()
    #         max60 = data60['high'].max()
    #         min180 = data180['low'].min()
    #         max180 = data180['high'].max()
    #         min250 = data250['low'].min()
    #         max250 = data250['high'].max()
    #         if (min60 * 1.3 > max60) | (min180 * 1.6 > max180) | (min250 * 2 > max250): continue
    #         if (MAdata.iloc[xx]['std10'] < MAdata.iloc[xx - 1]['std10']) | (
    #                 MAdata.iloc[xx]['MA10'] < MAdata.iloc[xx - 1]['MA100']): continue
    #         if (MAdata.iloc[xx]['MA10diff'] < 0) | (MAdata.iloc[xx - 1]['MA10diff'] < 0): continue
    #         if (MAdata.iloc[xx]['high'] < MAdata.iloc[xx]['MA10']) | (
    #                 MAdata.iloc[xx - 1]['high'] < MAdata.iloc[xx - 1]['MA10']): continue
    #         MAdata['negativebias'] = MAdata['low'] - MAdata['MA10']
    #         if MAdata['negativebias'].min() > 0: continue
    #         res.append(s + " " + sl[sl[1] == s][0].iloc[0])
    #     df = pd.DataFrame(res)
    #     filelist = os.listdir(strategydir)
    #     for i in filelist:
    #         if i.__contains__('strategy2'): os.remove(os.path.join(strategydir, i))
    #     df.to_excel(strategydir + nameStrategy('strategy2'))
    #     return df


def getSingledata(ts_code):
    ts.set_token('30f769d97409f6b9ff133558703d4cbe8302b4e6452330b2c11af044')
    pro = ts.pro_api()
    t1 = endDate()
    t2 = (datetime.now() - relativedelta(years=1)).strftime('%Y%m%d')
    res = ''
    income = pro.income(ts_code=ts_code, start_date=t2, end_date=t1,
                        fields='ts_code,ann_date,f_ann_date,end_date,comp_type,basic_eps,total_revenue,total_cogs,operate_profit,total_profit,n_income')
    res = res + getreportbydata(income, '利润数据')
    balance = pro.balancesheet(ts_code=ts_code, start_date=t2, end_date=t1,
                               fields='ts_code,ann_date,f_ann_date,end_date,comp_type,total_cur_liab,total_ncl')
    res = res + getreportbydata(balance, '资产负债')
    forecast = pro.forecast(ts_code=ts_code, start_date=t2, end_date=t1,
                            fields='ts_code,ann_date,type, p_change_min,p_change_max,net_profit_min,net_profit_max')
    res = res + getreportbydata(forecast, '业绩预告')
    express = pro.express(ts_code=ts_code, start_date=t2, end_date=t1,
                          fields='ts_code,ann_date,end_date,revenue,total_profit,total_assets,diluted_roe,yoy_op,np_last_year')
    res = res + getreportbydata(express, '业绩快报')
    finaIndicator = pro.fina_indicator(ts_code=ts_code, start_date=t2, end_date=t1,
                                       fields='ts_code,ann_date,end_date,eps,current_ratio,assets_turn,netdebt,debt_to_assets')
    res = res + getreportbydata(finaIndicator, '财务指标数据')
    mainbzP = pro.fina_mainbz(ts_code=ts_code, start_date=t2, end_date=t1, type='P')
    res = res + getreportbydata(mainbzP, '主营业务构成(业务)')
    mainbzD = pro.fina_mainbz(ts_code=ts_code, start_date=t2, end_date=t1, type='D')
    res = res + getreportbydata(mainbzD, '主营业务构成（地区）')
    return res


def getreportbydata(data, title):
    res = title + ':         \n'
    data = data.drop_duplicates()
    dic = {'end_date': '报告期', 'ann_date': '公告日期', 'f_ann_date': '实际公告日期', 'comp_type': '公司类型(1一般工商业2银行3保险4证券)',
           'basic_eps': '基本每股收益', 'total_revenue': '营业总收入', 'total_cogs': '营业总成本', 'operate_profit': '营业利润',
           'total_profit': '利润总额', 'n_income': '净利润(含少数股东损益)',
           'total_cur_liab': '流动负债合计', 'total_ncl': '非流动负债合计', 'p_change_min': '预告净利润变动幅度下限（%）',
           'p_change_max': '预告净利润变动幅度上限（%）', 'net_profit_min': '	预告净利润下限（万元）', 'net_profit_max': '预告净利润上限（万元）',
           'revenue': '营业收入(元)', 'total_assets': '总资产(元)', 'diluted_roe': '净资产收益率(摊薄)(%)', 'yoy_op': '同比增长率:营业利润',
           'np_last_year': '去年同期净利润',
           'curr_type': '货币代码', 'type': '业绩预告类型', 'eps': '基本每股收益', 'current_ratio': '流动比率', 'assets_turn': '总资产周转率',
           'netdebt': '净债务', 'debt_to_assets': '资产负债率', 'bz_item': '主营业务', 'bz_profit': '主营业务利润(元)',
           'bz_sales': '主营业务收入(元)', 'bz_cost': '主营业务成本(元)'}
    c = data.columns.values
    for d in data.itertuples(index=False):
        for i in range(0, len(c)):
            if (c[i] == 'ts_code'): continue
            res = res + dic[c[i]] + ':' + str(d[i]) + '  '
        res = res + '\n'
    return res


def getProducts():
    filedir = nameStrategy("productsList")
    if os.path.exists(filedir):
        data = pd.read_excel(filedir)
    else:
        filelist = os.listdir(datadir)
        for i in filelist:
            if i.__contains__("productList"):
                os.remove(i)
        ts.set_token('30f769d97409f6b9ff133558703d4cbe8302b4e6452330b2c11af044')
        pro = ts.pro_api()
        data = pro.stock_basic(exchange='', list_status='L', fields='name, ts_code')
        data.rename(columns={"name": 0, "ts_code": 1}, inplace=True)
        data = data[[0, 1]]
        pandas.DataFrame(data).to_excel(nameStrategy("productsList"), index=False)
    return data


def getdailyData(self):
    t1 = endDate()
    t2 = (datetime.now() - relativedelta(years=1)).strftime('%Y%m%d')
    filelist = os.listdir(datadir)
    if (filelist.__contains__(nameStrategy(ashares[1][0]))) & (len(filelist) == len(ashares)): return
    if not filelist.__contains__(nameStrategy(ashares[1][0])):
        for i in filelist: os.remove(os.path.join(datadir, i))
    for tmp in self.iterrows():
        print(tmp[1][1] + ' daily data extracting')
        pro = ts.pro_api()
        ts.set_token('30f769d97409f6b9ff133558703d4cbe8302b4e6452330b2c11af044')
        if filelist.__contains__(nameStrategy(tmp[1][1])): continue
        try:
            df = pro.daily(ts_code=tmp[1][1], start_date=t2, end_date=t1)
        except:
            print(tmp[1][1] + '出错')
            continue
        df = df.iloc[::-1]
        df.to_excel(datadir + nameStrategy(tmp[1][1]), index=False)

def getFinancialData(self):
    filelist = os.listdir(financialdir)
    for tmp in self.iterrows():
        pro = ts.pro_api()
        ts.set_token('30f769d97409f6b9ff133558703d4cbe8302b4e6452330b2c11af044')
        if filelist.__contains__(nameStrategyfortxt(tmp[1][1])): continue
        try:
            print(tmp[1][1] + ' financial data extracting')
            df = getSingledata(tmp[1][1])
            time.sleep(4)
        except Exception as e:
            print(e)
            print(tmp[1][1] + '出错')
            continue
        f = open(financialdir + nameStrategyfortxt(tmp[1][1]), "w")
        f.write(df)
        f.close()
    filelist = os.listdir(financialdir)
    if not len(filelist) == len(self): self.getFinancialData()


def getfinancialdatafromlocal(sharesId):
    with open(financialdir+nameStrategyfortxt(sharesId),encoding='GBK') as file:
        return file.read()


if __name__ == '__main__':
    pro = ts.pro_api()
    ts.set_token('30f769d97409f6b9ff133558703d4cbe8302b4e6452330b2c11af044')
    df = pro.daily(ts_code='000001.SZ', start_date=20220801, end_date=20220811)
    ashares = getProducts()
    ashares.reset_index(drop=True, inplace=True)
    s = Strategy()
    diglogdemo = DialogDemo(ashares, s)
    diglogdemo.createDialog()
