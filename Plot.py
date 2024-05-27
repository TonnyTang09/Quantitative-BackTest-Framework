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

        self.data = backtest_result_data

        self.strategy = strategy

        self.code = code

        self.name = strategy.name

        self.annotate = True

        self.show_indicator = show_indicator

    def cumulative_return(self):
        '''
        画出累积收益率图 有仓位无操作和无仓位的时间段都会显示横线 只有调仓后会画出账户收益变动
        '''
        df = self.data  # Position data

        start_date = df['日期时间'].iloc[0].strftime('%Y-%m-%d')

        end_date = df['日期时间'].iloc[-1].strftime('%Y-%m-%d')

        hover_text = []  # List to store hover text for each point
        df['日期时间'] = pd.to_datetime(df['日期时间'])  # Convert to datetime

        self.data['权益'] = self.data['权益'].apply(lambda x: int(x.replace(',', '')))

        return_rate = np.array(df['权益']) / df['权益'].iloc[0] - 1  # 换成权益是浮盈浮亏

        return_percentage = np.array(["{:.2%}".format(num) for num in return_rate])

        for i in range(len(df['日期时间'])):
            hover_text.append(
                f"Time: {df['日期时间'].iloc[i].strftime('%Y-%m-%d %H:%M:%S')}<br>Return: {return_percentage[i]}")

        fig = go.Figure(data=go.Scatter(x=df.index, y=df['权益'], mode='lines'))  # 换成权益是浮盈浮亏
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

        df = self.data

        start_date = df['日期时间'].iloc[0].strftime('%Y-%m-%d')

        end_date = df['日期时间'].iloc[-1].strftime('%Y-%m-%d')

        # Create candlestick figure
        if not self.annotate:
            fig = make_subplots(rows=1, cols=1, subplot_titles=(f'{self.code} between {start_date} and {end_date}',))
        else:
            fig = make_subplots(rows=1, cols=1,
                                subplot_titles=(f'{self.name} for {self.code} between {start_date} and {end_date}',))

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

        indicator_to_draw = []

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
                    if df[col].iloc[j] == 0:
                        continue
                    fig.add_annotation(x=df.index[j], y=df['最高价'].iloc[j], text=df[col].iloc[j], showarrow=True,
                                       arrowcolor='red', arrowhead=1)
            else:
                # 如果不是分类变量，添加散点图
                color = colors[i % len(colors)]  # 使用取模运算来循环使用颜色序列
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[col], mode='lines', name=col.capitalize(), line=dict(color=color)),
                    row=1, col=1)

        # Add annotations
        if self.annotate and '操作' in df.columns:
            for i in range(len(df)):
                if df['操作'].iloc[i] in ['多开', '空开', '多平', '空平', '移仓', '空平+多开', '多平+空开']:
                    fig.add_annotation(x=df.index[i], y=df['收盘价'].iloc[i], text=df['操作'].iloc[i], showarrow=True,
                                       arrowcolor='red' if df['操作'].iloc[i] in ['多开', '空开'] else 'green',
                                       arrowhead=1)

        hover_text = []
        x_list = []
        for i in range(len(df)):
            x_list.append(df['日期时间'].iloc[i].strftime('%Y-%m-%d %H:%M:%S'))
            must_have_text = f"Time: {df['日期时间'].iloc[i].strftime('%Y-%m-%d %H:%M:%S')}<br>开盘价: {df['开盘价'].iloc[i]}<br>最高价: {df['最高价'].iloc[i]}<br>最低价: {df['最低价'].iloc[i]}<br>收盘价: {df['收盘价'].iloc[i]}<br>涨跌幅: {df['涨跌幅'].iloc[i]}"
            for indicator in indicator_to_draw:
                if indicator not in df.columns:
                    continue
                must_have_text += f"<br>{indicator}: {df[indicator].iloc[i]}"
            hover_text.append(must_have_text)

        fig.update_traces(hoverinfo='text', hovertext=hover_text)

        # Set layout
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

        # Save the figure as HTML
        if self.annotate:
            image_file_path = os.path.join(figure_directory,
                                           f'{self.name} for {self.code} between {start_date} and {end_date}.html')
        else:
            image_file_path = os.path.join(figure_directory,
                                           f'{self.code} between {start_date} and {end_date}.html')

        # 保存图片到指定路径
        plot(fig, filename=image_file_path, auto_open=True)


class specific_Plot(Plot):

    def __init__(self, code, start_time, end_time, frequency, indicator_require):

        self.start_time = start_time

        self.end_time = end_time

        self.code = code

        self.indicator_require = indicator_require

        self.frequency = frequency

        self.annotate = False

    def create_obj(self):

        underlying = Data(self.code)  # Initialize the Data object of underlying contract

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

    def specific_cumulative_return(self):

        obj = self.create_obj()

        obj.cumulative_return()
