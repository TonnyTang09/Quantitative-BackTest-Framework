import numpy as np

from Data import Data
from Backtest import Backtest
import Strategy
import os
from plotly.offline import plot
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from iFinDPy import *


class Plot:

    def __init__(self, backtest_result_data, code, strategy, show_indicator={}):

        self.data = backtest_result_data  # 回测后标记了买卖点位的数据

        self.strategy = strategy  # 当前回测的策略

        self.code = code  # 当前回测标的的合约代码

        self.name = strategy.name

        self.annotate = True  # 通过这个类属性来控制是否需要在k线图上展示买卖点

        self.show_indicator = show_indicator  # 特别指明的需要画出的技术指标 默认情况会沿用strategy中用于判断开仓平仓条件的指标

    def cumulative_return(self):
        """
        画出累积收益率图 无仓位的时间段都会显示横线
        """
        df = self.data

        start_date = df.loc[0, '日期时间'].strftime('%Y-%m-%d')  # 回测的开始时间

        end_date = df['日期时间'].iloc[-1].strftime('%Y-%m-%d')  # 回测的结束时间

        df['日期时间'] = pd.to_datetime(df['日期时间'])

        # 把权益变成百分数的形式
        self.data['权益'] = self.data['权益'].apply(lambda x: int(x.replace(',', '')))

        return_rate = np.array(df['权益']) / df.loc[0, '权益'] - 1

        return_percentage = np.array(["{:.2%}".format(num) for num in return_rate])

        hover_text = []  # 用这个变量来存鼠标悬停在图片上后展示的图例

        for i in range(len(df['日期时间'])):
            hover_text.append(
                f"Time: {df.loc[i, '日期时间'].strftime('%Y-%m-%d %H:%M:%S')}<br>Return: {return_percentage[i]}")

        # 设置一些画图的参数 画出图片
        fig = go.Figure(data=go.Scatter(x=df.index, y=df['权益'], mode='lines'))
        fig.update_traces(hoverinfo='text', hovertext=hover_text)
        fig.update_layout(
            title=f'Cumulative Return of {self.name} for {self.code} between {start_date} and {end_date}',
            xaxis_title='Time',
            yaxis_title='Cumulative Returns',
            xaxis=dict(tickangle=45),
        )

        label_pair = dict()
        for ind in df.index:
            label_pair[ind] = df['日期时间'][ind].strftime('%Y-%m-%d %H:%M:%S')

        fig.update_xaxes(labelalias=label_pair)

        # 获取当前文件所在的目录路径
        current_directory = os.getcwd()

        # 创建 "Figure" 文件夹路径
        figure_directory = os.path.join(current_directory, "Figure")

        # 确保 "Figure" 文件夹存在，如果不存在则创建
        if not os.path.exists(figure_directory):
            os.makedirs(figure_directory)

        # 拼接图片文件路径
        image_file_path = os.path.join(figure_directory,
                                       f'Cumulative Return of {self.name} for {self.code} between {start_date} and {end_date}.html')

        # 保存图片到指定路径
        plot(fig, filename=image_file_path, auto_open=True)

    def k_chart(self):
        """
        画出带有策略买卖点的k线图
        """

        df = self.data

        start_date = df.loc[0, '日期时间'].strftime('%Y-%m-%d')

        end_date = df['日期时间'].iloc[-1].strftime('%Y-%m-%d')

        # 如果当前不需要标记出策略的买卖点 说明只是在获取某一段时间内的k线 那么图片保存的名字要有所不同
        if not self.annotate:
            fig = make_subplots(rows=1, cols=1, subplot_titles=(f'{self.code} between {start_date} and {end_date}',))
        else:
            fig = make_subplots(rows=1, cols=1,
                                subplot_titles=(f'{self.name} for {self.code} between {start_date} and {end_date}',))

        # 设置一些画图的参数
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['开盘价'],
            high=df['最高价'],
            low=df['最低价'],
            close=df['收盘价'],
            increasing_line_color='red',
            decreasing_line_color='green',
            name='Candlestick'
        ), row=1, col=1)

        # 获取颜色序列
        colors = ['blue', 'yellow', 'black', 'green', 'pink', 'red', 'purple']

        # 用这个列表来存一下要画上哪些技术指标
        indicator_to_draw = []

        # 把原本指定技术指标的字典中的键值对转换为一个字符串形式（跟数据中的列名相同）
        if not self.show_indicator:
            temp_dict = self.strategy.indicator_require
        else:
            temp_dict = self.show_indicator

        for key in temp_dict.keys():

            for ele in temp_dict[key]:

                if isinstance(ele, bool):
                    temp_indicator = str(key)
                else:
                    temp_indicator = str(key) + str(ele)

                indicator_to_draw.append(temp_indicator)

        # 循环遍历每个列，并添加到图表中
        for i, col in enumerate(indicator_to_draw):
            if col not in df.columns:
                continue
            # 检查列是否是分类变量
            if len(df[col].unique()) <= 10:  # 假设10是一个阈值，可以根据实际情况调整
                # 如果是分类变量，添加注释
                for j in range(len(df)):
                    if df.loc[j, col] == 0:
                        continue
                    fig.add_annotation(x=df.index[j], y=df.loc[j, '最高价'], text=df.loc[j, col], showarrow=True,
                                       arrowcolor='red', arrowhead=1)
            else:
                # 如果不是分类变量，添加折线图
                color = colors[i % len(colors)]  # 使用取模运算来循环使用颜色序列
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[col], mode='lines', name=col.capitalize(), line=dict(color=color)),
                    row=1, col=1)

        # 在图上加入策略买卖点
        if self.annotate and '操作' in df.columns:
            for i in range(len(df)):
                if df.loc[i, '操作'] in ['多开', '空开', '多平', '空平', '移仓', '空平+多开', '多平+空开']:
                    fig.add_annotation(x=df.index[i], y=df.loc[i, '收盘价'], text=df.loc[i, '操作'], showarrow=True,
                                       arrowcolor='red' if df.loc[i, '操作'] in ['多开', '空开'] else 'green',
                                       arrowhead=1)

        # 设置悬停后展示的图例以及一些参数设定
        hover_text = []
        x_list = []
        for i in range(len(df)):
            x_list.append(df.loc[i, '日期时间'].strftime('%Y-%m-%d %H:%M:%S'))
            must_have_text = f"Time: {df.loc[i, '日期时间'].strftime('%Y-%m-%d %H:%M:%S')}<br>开盘价: {df.loc[i, '开盘价']}<br>最高价: {df.loc[i, '最高价']}<br>最低价: {df.loc[i, '最低价']}<br>收盘价: {df.loc[i, '收盘价']}<br>涨跌幅: {df.loc[i, '涨跌幅']}"
            for indicator in indicator_to_draw:
                if indicator not in df.columns:
                    continue
                must_have_text += f"<br>{indicator}: {df.loc[i, indicator]}"
            hover_text.append(must_have_text)

        fig.update_traces(hoverinfo='text', hovertext=hover_text)

        fig.update_layout(
            xaxis_title='Datetime',
            yaxis_title='Price',
            xaxis_tickangle=-45,
            xaxis_rangeslider_visible=False,
            hoverlabel=dict(
                bgcolor="black",
                font_color="white"
            )
        )

        label_pair = dict()
        for ind in df.index:
            label_pair[ind] = df['日期时间'][ind].strftime('%Y-%m-%d %H:%M:%S')

        fig.update_xaxes(labelalias=label_pair)

        # 获取当前文件所在的目录路径
        current_directory = os.getcwd()

        # 创建 "Figure" 文件夹路径
        figure_directory = os.path.join(current_directory, "Figure")

        # 确保 "Figure" 文件夹存在，如果不存在则创建
        if not os.path.exists(figure_directory):
            os.makedirs(figure_directory)

        if self.annotate:
            image_file_path = os.path.join(figure_directory,
                                           f'{self.name} for {self.code} between {start_date} and {end_date}.html')
        else:
            image_file_path = os.path.join(figure_directory,
                                           f'{self.code} between {start_date} and {end_date}.html')

        # 保存图片到指定路径
        plot(fig, filename=image_file_path, auto_open=True)


class specific_Plot(Plot):
    """
    通过这个类来获取某一个特定时间段内某品种的k线 用于复盘某段特定的行情
    """

    def __init__(self, code, start_time, end_time, frequency, indicator_require):

        self.start_time = start_time  # 时间段的开始点

        self.end_time = end_time  # 时间段的结束点

        self.code = code  # 标的的合约代码

        self.indicator_require = indicator_require  # 需要在图上画出哪些技术指标

        self.frequency = frequency  # 想要的数据频率

        self.annotate = False  # 不需要在图上标记买卖点 设置成False

    def create_obj(self):
        """
        先用这个函数构建一个Plot类对象 然后再调用Plot的类方法来画图
        """
        # 先获取一下指定标的的数据
        underlying = Data(self.code)

        if self.frequency == 'daily':

            underlying.data = THS_HQ(self.code, 'open,high,low,close,changeRatio', '', self.start_time,
                                     self.end_time).data

        else:
            self.start_time = self.start_time + ' 09:00:00'

            self.end_time = self.end_time + ' 15:00:00'

            underlying.data = THS_HF(self.code, 'open;high;low;close;volume;changeRatio_periodical',
                                     f'Fill:Original,Interval:{self.frequency}', self.start_time,
                                     self.end_time).data

        # 先遍历当前strategy需要哪些指标 self.strategy.indicator_require这一字典的key是指标种类 value是参数
        for key in self.indicator_require.keys():

            # 通过Data类中的data pipline来获取当前指标对应的函数
            func = underlying.data_pipline[key]  # Get the function used to calculate the current indicator

            # 遍历当前该指标需要哪些参数 对每一个参数都调用一次计算该指标的函数
            for ele in self.indicator_require[key]:

                # 有些指标不需要参数 如TR vwap 所以再self.strategy.indicator_require会以 'TR':[True]这种形式储存
                # 检查ele是否为布尔型 如果是 则不需要传参
                if isinstance(ele, bool):
                    func()

                    continue

                func(ele)  # Call the above function

        new_column_names = {
            'date': '日期时间',
            'time': '日期时间',
            'thscode': '合约代码',
            'open': '开盘价',
            'high': '最高价',
            'low': '最低价',
            'close': '收盘价',
            'changeRatio_periodical': '涨跌幅',
            'changeRatio': '涨跌幅'
        }

        underlying.data.rename(columns=new_column_names, inplace=True)

        underlying.data['日期时间'] = pd.to_datetime(underlying.data['日期时间'])

        underlying.data['涨跌幅'] = round(underlying.data['涨跌幅'], 2) / 100

        underlying.data['涨跌幅'] = np.array(["{:.2%}".format(num) for num in underlying.data['涨跌幅']])

        temp_plot = Plot(underlying.data, underlying.code, Strategy.Strategy(), show_indicator=self.indicator_require)

        temp_plot.annotate = False

        return temp_plot

    def specific_k_chart(self):

        obj = self.create_obj()

        obj.k_chart()
