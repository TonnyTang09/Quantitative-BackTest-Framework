import matplotlib
from iFinDPy import *
matplotlib.use('TkAgg')
from Data import Data
import Strategy
from Backtest import Backtest
from Plot import Plot, specific_Plot
import pandas as pd

if __name__ == '__main__':
    """
    进行回测的例子
    """

    THS_iFinDLogin("xhlh009", "5e1295")

    obj = Data('IZL.DCE')  # 初始化一个Data类对象 合约代码代表想要回测的标的

    strategy = Strategy.momentum_1h_updated()  # 初始化一个Strategy类对象 策略名字是想要回测的策略

    obj.initial_data(strategy, '2023-06-01', '2024-5-20')  # 调用Data类中的intial_data类方法来初始化数据

    result = Backtest(obj, strategy)  # 用初始化好数据后的Data类对象和想要回测的Strategy类对象初始化一个Backtest对象

    result.run_backtest()  # 调用run_backtest进行回测

    plot = Plot(result.data, obj.code, strategy)  # 通过回测后的结果（在result.data中）、回测标的的合约代码和回测的策略初始化Plot类

    plot.cumulative_return()  # 调用cumulative_return画出策略在回测区间内的账户权益变化

    plot.k_chart()  # 调k_chart画出标记上策略买卖点的k线图

    """
    给定合约代码画出指定区间和指定技术指标的例子
    """
    # test_plot = specific_Plot('SA309.CZC', '2023-06-01', '2023-10-01', 1, {'ma':[5,10]})
    #
    # test_plot.specific_k_chart()