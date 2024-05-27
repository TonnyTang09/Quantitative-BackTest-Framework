import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from iFinDPy import *
from sklearn.linear_model import LogisticRegression


class Strategy:
    def __init__(self):
        """
        全部策略的父类 Strategy。后续在编写新的策略时，可以在此基础上添加新的内容，但子类中需要覆写父类中已有的属性和方法。
        """
        self.name = None  # 策略的名字
        self.indicator_require = {}  # 策略所需要的技术指标
        # 例如：{'ma':[5,10],'TR':[True]}，第一个键值对表示需要5日和10日均线，第二个表示需要True Range（极差），因为有些指标不需要参数，所以只需放True即可。
        self.time_frequency = 0  # 策略所需要的数据频率
        self.need_daily = False  # 策略是否需要用到日频数据

    def open_criterion(self, data, i, direction=0):
        """
        策略的开仓条件
        :param data: 包含了K线和技术指标的数据框
        :param i: 当前时间步
        :param direction: 默认值为0，表示在没有头寸时的开仓条件。如果是1或-1，则表示在持有多头或空头头寸时的加仓条件
        :return: 返回0表示不开仓，返回正数或负数表示开多或开空。如果是小数，表示用当前账户余额的比例来交保证金；如果大于1的数，则表示具体开多少手。
        """
        return 0

    def close_criterion(self, data, i, direction):
        """
        策略的平仓条件
        :param data: 包含了K线和技术指标的数据框
        :param i: 当前时间步
        :param direction: 取1或-1，表示在持有多头或空头头寸时的平仓条件
        :return: 返回0表示不平仓，返回小数表示平掉当前头寸的比例大小；如果是大于1的数，则表示具体平多少手。
        """
        return 0


class momentum_1h_original(Strategy):
    """
    结合60小时线和60日线的双均线策略。在这两个均线以上时多开，在这两个均线以下时多平，其余情况不开仓或平仓。
    """

    def __init__(self):
        super().__init__()
        self.name = 'momentum_1h_original'
        self.indicator_require = {'ma': [60]}
        self.time_frequency = 60
        self.need_daily = True

    def open_criterion(self, data, i, direction=0):
        if direction == 0:
            # 当direction是0的时候，此时是新开仓，否则可以加仓
            temp_price = data['close'].iloc[i]
            temp_ma60 = data['ma60'].iloc[i]
            temp_ma60_daily = data['ma60_daily'].iloc[i]

            if temp_price > temp_ma60 and temp_price > temp_ma60_daily:
                return 0.2

            if temp_price < temp_ma60 and temp_price < temp_ma60_daily:
                return -0.2

        return 0

    def close_criterion(self, data, i, direction):
        temp_price = data['close'].iloc[i]
        temp_ma60 = data['ma60'].iloc[i]
        temp_ma60_daily = data['ma60_daily'].iloc[i]

        if direction == 1:
            if temp_price < temp_ma60 or temp_price < temp_ma60_daily:
                return 1

        if direction == -1:
            if temp_price > temp_ma60 or temp_price > temp_ma60_daily:
                return 1

        return 0


class momentum_1h_updated(Strategy):
    """
    60小时线和60日线双均线策略的改进版本，放宽了平仓条件，并加入了时间止损。
    """

    def __init__(self):
        super().__init__()
        self.name = 'momentum_1h_updated'
        self.indicator_require = {'ma': [60]}
        self.time_frequency = 60
        self.stop_loss_point = 0  # 止损点位
        self.ready_to_stop = False  # 是否准备止损
        self.wait_k_line_num = 0  # 等待K线的数量
        self.need_daily = True

    def open_criterion(self, data, i, direction=0):
        if direction == 0:
            temp_price = data['close'].iloc[i]
            temp_ma60 = data['ma60'].iloc[i]
            temp_ma60_daily = data['ma60_daily'].iloc[i]

            if temp_price > temp_ma60 and temp_price > temp_ma60_daily:
                return 0.2

            if temp_price < temp_ma60 and temp_price < temp_ma60_daily:
                return -0.2

        return 0

    def close_criterion(self, data, i, direction):
        temp_price = data['close'].iloc[i]
        temp_ma60 = data['ma60'].iloc[i]
        temp_ma60_daily = data['ma60_daily'].iloc[i]

        if direction == 1:
            if temp_price < temp_ma60 or temp_price < temp_ma60_daily:
                if not self.ready_to_stop:
                    # 当第一次触发止损条件时 开启止损记录 调整止损点位
                    self.ready_to_stop = True
                    self.stop_loss_point = temp_price * 0.99  # 设置止损点
                    return 0
                else:
                    # 当已经触发了止损时 如果当前价格已经跌破了止损点点位 则平仓 否则 把等待k线数加一 当等了5根k线没修复时 平仓
                    if temp_price < self.stop_loss_point:
                        self.ready_to_stop = False
                        self.wait_k_line_num = 0
                        return 1
                    self.wait_k_line_num += 1
                    if self.wait_k_line_num >= 5 and (
                            temp_price < temp_ma60 * 0.99 or temp_price < temp_ma60_daily * 0.99):
                        self.wait_k_line_num = 0
                        self.ready_to_stop = False
                        return 1

        if direction == -1:
            if temp_price > temp_ma60 or temp_price > temp_ma60_daily:
                if not self.ready_to_stop:
                    self.ready_to_stop = True
                    self.stop_loss_point = temp_price * 1.01  # 设置止损点
                    return 0
                else:
                    if temp_price > self.stop_loss_point:
                        self.ready_to_stop = False
                        self.wait_k_line_num = 0
                        return 1
                    self.wait_k_line_num += 1
                    if self.wait_k_line_num >= 5 and (
                            temp_price > temp_ma60 * 1.01 or temp_price > temp_ma60_daily * 1.01):
                        self.wait_k_line_num = 0
                        self.ready_to_stop = False
                        return 1

        return 0


class momentum_v3(Strategy):
    """
    改进的动量策略版本 以小时线的ma15 和 ma60作为判断指标 同时动态维护每一次交易的止盈点位 引入Trailing Stop Loss机制
    """

    def __init__(self):
        super().__init__()
        self.name = 'momentum_v3'
        self.indicator_require = {'ma': [15, 60]}
        self.time_frequency = 60
        self.temp_date = None  # 临时日期
        self.temp_high = None  # 临时最高价
        self.temp_low = None  # 临时最低价
        self.long_target = None  # 多头目标
        self.short_target = None  # 空头目标
        self.sl_target = None  # 止损目标
        self.count = 0  # 计数器
        self.temp_ZL = None  # 临时证券代码
        self.need_daily = False
        self.open_price = None  # 开仓价
        self.sl_price = None  # 止损价
        self.short_time = []  # 空头时间
        self.ready_to_stop = False  # 是否准备止损
        self.wait_k_line = 0  # 等待K线数量

    def open_criterion(self, data, i, direction=0):
        if direction == 0:
            temp_price = data['close'].iloc[i]
            data['date'] = pd.to_datetime(data['date'])
            temp_date = data['date'].iloc[i].date()
            temp_ma15 = data['ma15'].iloc[i]
            temp_ma60 = data['ma60'].iloc[i]
            temp_underlying = data['证券代码'].iloc[i]

            if self.temp_ZL != temp_underlying:
                self.count = 0

            self.temp_ZL = temp_underlying
            self.count += 1

            if self.count >= 30:
                self.temp_high = data['close'].iloc[i - 29:i + 1].max()
                self.temp_low = data['close'].iloc[i - 29:i + 1].min()

                if temp_ma60 < temp_ma15 < temp_price and temp_price >= self.temp_high:
                    hundred_days_ago_string = (temp_date - timedelta(days=100)).strftime("%Y-%m-%d")
                    temp_date_string = (temp_date - timedelta(days=1)).strftime("%Y-%m-%d")
                    self.long_target = THS_HQ(temp_underlying, 'close', '', hundred_days_ago_string,
                                              temp_date_string).data['close'].max() * 0.99

                    if temp_price >= self.long_target:
                        self.long_target = float('inf')

                    self.open_price = temp_price
                    self.sl_price = -float('inf')
                    return 0.3

                if temp_ma60 > temp_ma15 > temp_price and temp_price <= self.temp_low:
                    hundred_days_ago_string = (temp_date - timedelta(days=100)).strftime("%Y-%m-%d")
                    temp_date_string = (temp_date - timedelta(days=1)).strftime("%Y-%m-%d")
                    self.short_target = THS_HQ(temp_underlying, 'close', '', hundred_days_ago_string,
                                               temp_date_string).data['close'].min() * 1.01

                    if temp_price <= self.short_target:
                        self.short_target = -float('inf')

                    self.open_price = temp_price
                    self.sl_price = float('inf')
                    self.short_time.append(temp_date)
                    return -0.3

            return 0

        return 0

    def close_criterion(self, data, i, direction):
        temp_price = data['close'].iloc[i]
        temp_ma60 = data['ma60'].iloc[i]

        if direction == 1:
            if temp_price >= self.open_price * 1.03:
                self.sl_price = (temp_price - self.open_price) * 0.2 + self.open_price

            if temp_price >= self.long_target or temp_price <= temp_ma60 or temp_price <= self.sl_price:
                return 1

        if direction == -1:
            if temp_price <= self.open_price * 0.97:
                self.sl_price = self.open_price - (self.open_price - temp_price) * 0.2

            if temp_price <= self.short_target or temp_price >= temp_ma60 or temp_price >= self.sl_price:
                return 1

        return 0
