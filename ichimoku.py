# -*- coding: utf-8 -*-
import time

import numpy
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
from mpl_finance import candlestick_ohlc, candlestick2_ohlc
import numpy as np
import decimal

# Press the green button in the gutter to run the script.
from numpy import long

fig, ax = plt.subplots()
x, y, lastday, xminnow, xmaxnow = 1, 1, 0, 0, 0


def vision(data):
    ichimoku = Ichimoku(data)
    ichimoku.run()
    ichimoku.plot()


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
    plt.show()


class Ichimoku():
    """
    @param: ohcl_df <DataFrame> 

    Required columns of ohcl_df are: 
        Date<Float>,Open<Float>,High<Float>,Close<Float>,Low<Float>
    """

    def __init__(self, ohcl_df):
        self.ohcl_df = ohcl_df
        ohcl_df['trade_date'] = pandas.to_datetime(ohcl_df['trade_date'])

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
        return ohcl_df

    def plot(self):
        global xminnow, xmaxnow
        fig.canvas.mpl_connect('scroll_event', call_back)
        fig.canvas.mpl_connect('button_press_event', button_press_callback)
        fig.canvas.mpl_connect('motion_notify_event', motion_notify_callback)
        self.plot_candlesticks(fig, ax)
        self.plot_ichimoku(fig, ax)
        self.pretty_plot(fig, ax)
        plt.xlim(xmin=lastday - 200, xmax=lastday)
        xminnow = lastday - 200
        xmaxnow = lastday
        plt.show()

    def pretty_plot(self, fig, ax):
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
        title = 'ichimoku'
        bgcolor = '#f0f0f0'
        grid_color = '#363c4e'
        spines_color = '#0f0f0f'
        # Axes
        plt.title(title, color='black')

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
        d2 = self.ohcl_df.loc[:, ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']]
        #  d2 = d2.tail(view_limit)
        date_axis = range(0, len(d2))
        # ichimoku
        plt.plot(date_axis, d2['tenkan_sen'], label="tenkan", color='#004200', alpha=1, linewidth=2)
        plt.plot(date_axis, d2['kijun_sen'], label="kijun", color="#721d1d", alpha=1, linewidth=2)
        plt.plot(date_axis, d2['senkou_span_a'], label="span a", color="#008000", alpha=0.65, linewidth=0.5)
        plt.plot(date_axis, d2['senkou_span_b'], label="span b", color="#ff0000", alpha=0.65, linewidth=0.5)
        plt.plot(date_axis, d2['chikou_span'], label="chikou", color="black", alpha=0.65, linewidth=0.5)
        # green cloud
        ax.fill_between(date_axis, d2['senkou_span_a'], d2['senkou_span_b'],
                        where=d2['senkou_span_a'] > d2['senkou_span_b'], facecolor='#008000', interpolate=True,
                        alpha=0.25)
        # red cloud
        ax.fill_between(date_axis, d2['senkou_span_a'], d2['senkou_span_b'],
                        where=d2['senkou_span_b'] > d2['senkou_span_a'], facecolor='#ff0000', interpolate=True,
                        alpha=0.25)

    def plot_candlesticks(self, fig, ax, view_limit=10):
        # plot candlesticks

        candlesticks_df = self.ohcl_df.loc[:, ['trade_date', "open", "high", "low", "close"]]
        #  candlesticks_df = candlesticks_df.tail(view_limit)
        # plot candlesticks
        #   candlesticks_df['trade_date'] = mdates.date2num(candlesticks_df['trade_date'])
        # candlestick_ohlc(ax, candlesticks_df.values, width=0.5, colorup='#83b987', colordown='#eb4d5c', alpha=0.5)

        candlestick2_ohlc(ax, candlesticks_df['open'], candlesticks_df['high'], candlesticks_df['low'],
                          candlesticks_df['close'], width=0.6, colorup='#83b987', colordown='#eb4d5c', alpha=0.5)

    # mpf.plot(candlesticks_df, width=0.6, colorup='#83b987', colordown='#eb4d5c', alpha=0.5)

    # Range generator for decimals
    def drange(self, x, y, jump):
        while x < y:
            yield float(x)
            x += decimal.Decimal(jump)


def ashareslist(excel):
  #  ashareExcel = openpyxl.load_workbook(excel)
    lsh = pd.read_excel(excel, sheet_name='上证', header=None, dtype=np.str)
    ashares = pd.DataFrame()
    ashares = ashares.append(pd.read_excel(excel, sheet_name='深证', header=None, dtype=np.str)).append(pd.read_excel(excel, sheet_name='创业板', header=None, dtype=np.str))
    ashares[1] = ashares[1] + '.SZ'
    lsh[1] = lsh[1] + '.SH'
    ashares = ashares.append(lsh)
    return ashares


if __name__ == '__main__':
    ts.set_token('30f769d97409f6b9ff133558703d4cbe8302b4e6452330b2c11af044')
    pro = ts.pro_api()
    ashares = ashareslist('ashares.xlsx')
    df = pro.daily(ts_code='300902.SZ', start_date='20200101', end_date='20210811')

    #   df2 = pd.read_csv('./sample-data/ohcl_sample.csv', index_col=0,)
    df = df.iloc[::-1]
    lastday = len(df)
    vision(df)
