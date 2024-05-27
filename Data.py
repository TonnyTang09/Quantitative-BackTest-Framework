import pandas as pd
from iFinDPy import *
from datetime import datetime, timedelta
import numpy as np
import os


class Data:
    """
    通过data类获取标的数据、计算相关技术指标。
    """

    def __init__(self, code):
        """
        :param code: 以字符串的形式传入想要回测/获取数据的期货合约代码 e.g. IZL.DCE/SMZL.CZC/I2409.DCE
        """
        self.code = code
        self.data = pd.DataFrame()  # 初始化一个空的数据框 后续获取到的数据会储存在这里

        # 这个字典用来存储编写好的技术指标函数 后续如果需要加入新的技术指标 在编写好函数后以键值对的形式添加到这个字典中
        # 注意在Strategy中指定策略需要的指标时 要保证键的名字一致
        self.data_pipline = {
            'ema': self.EMA,  # Exponential Moving Average
            'tr': self.TR,  # True Range
            'atr': self.ATR,  # Average True Range
            'bollinger': self.Bollinger_band,  # Bollinger Bands
            'ma': self.MA,  # Moving Average
            'vwap': self.vwap,  # Volume Weighted Average Price
            'magic_nine_turn': self.magic_nine_turn  # 神奇九转
        }

        self.long_margin = 0  # 初始化一个空变量 用来记录当前标的开多需要交的保证金

        self.short_margin = 0  # 初始化一个空变量 用来记录当前标的开空需要交的保证金

        self.underlying_multiplier = 0  # 初始化一个空变量 用来记录当前标的合约乘数

        self.transaction_rate = 0  # 初始化一个空变量 用来记录当前标的交易手续费

    def getdata(self, frequency, start_time, end_time):
        """
        通过这个函数来获取当前标的在指定时间范围内指定频率的数据 当只想获取数据而不想回测时 调用这个函数 可以是主连代码也可以是单个合约代码
        :param frequency: 数据的频率 支持 1 3 5 10 15 30 60(分钟频) 和 'daily'(日频)
        :param start_time: 开始时间 以字符串传入 e.g. '2023-01-01'
        :param end_time: 结束时间
        :return: 直接修改类变量 无返回值
        """

        if frequency == 'daily':
            # 注意通过同花顺接口访问到的数据一定要在后面加上.data 才能访问到获取到的数据框
            data = THS_HQ(self.code, 'open,high,low,close,volume,changeRatio', '', start_time, end_time).data
        else:
            data = THS_HF(self.code, 'open;high;low;close;volume,changeRatio_periodical',
                          f'Fill:Original,Interval:{frequency}',
                          start_time, end_time).data

        # 重命名一下几个列
        new_column_names = {
            'time': 'date',
            'thscode': '证券代码',
            'changeRatio': 'change_ratio',
            'changeRatio_periodical': 'change_ratio'
        }

        data.rename(columns=new_column_names, inplace=True)

        # 将获取到的数据存在类变量中
        self.data = data

        # 获取当前文件所在的目录路径
        current_directory = os.getcwd()

        # 创建 "raw data" 文件夹路径
        raw_data_directory = os.path.join(current_directory, "raw data")

        # 确保 "raw data" 文件夹存在，如果不存在则创建
        if not os.path.exists(raw_data_directory):
            os.makedirs(raw_data_directory)

        # 拼接 Excel 文件路径
        excel_file_path = os.path.join(raw_data_directory, f'{self.code}_{frequency}.xlsx')

        # 将数据保存到 Excel 文件中
        self.data.to_excel(excel_file_path, index=False)

    def basic_info(self, time):
        """
        用这个函数获取当前标的的一些基本信息 包括保证金比例、合约乘数、手续费率
        :param time: 给一个代表时间的字符串 为了符合同花顺数据接口 意义不大
        :return: 直接修改类变量 无返回值
        """

        date_string_for_contract_info = (time + ";") * 4

        date_string_for_contract_info = date_string_for_contract_info[:-1]
        # 最后这个参数形如'2023-01-01 09:00:00;2023-01-01 09:00:00;2023-01-01 09:00:00;2023-01-01 09:00:00;'

        contract_info = THS_BD(self.code,
                               'ths_contract_short_deposit_future;ths_contract_long_deposit_future'
                               ';ths_contract_multiplier;ths_transaction_procedure_rate_future',
                               date_string_for_contract_info).data

        # 将获取到的基本信息更新到类参数中存好 方便后续调用 注意保证金的单位是% 所以要乘0.01转换为小数
        self.long_margin = contract_info['ths_contract_long_deposit_future'].iloc[0] * 0.01

        self.short_margin = contract_info['ths_contract_short_deposit_future'].iloc[0] * 0.01

        self.underlying_multiplier = contract_info['ths_contract_multiplier'].iloc[0]

        # 这里多加一个判断是因为可能有些合约的手续费为0 但是返回的是空值 避免产生错误 单位是千分之一 所以要乘0.001转换为小数
        if contract_info['ths_transaction_procedure_rate_future'].iloc[0] is not None:
            self.transaction_rate = contract_info['ths_transaction_procedure_rate_future'].iloc[0] * 0.001

    def combine_all_data(self, strategy, start_date, end_date):
        """
        通过同花顺接口获取所需要数据、计算策略所需要的技术指标 这个是为了后续回测策略表现服务
        获取当前交易的主力合约的对应频率（由策略指定）的数据 计算技术指标 清除缺失值
        当遍历到换月节点时 更换交易合约重复操作 最后把合并好的数据存在self.data中 并保存在当前工作目录下名为Raw Data的文件夹中
        这样就会有一个覆盖回测区间的数据框 同时处理好了换月 在回测时只需要遍历这个数据框 通过具体策略的细节来完成计算
        :param strategy: 当前回测的策略 是一个Strategy类对象
        :param start_date: 回测开始的时间 以字符串传入 如'2023-01-01'
        :param end_date: 回测结束的时间
        :return: 直接修改类变量 无返回值
        """
        # 通过该函数获取主连的当前交易合约和对应的时间序列
        data_obj = THS_DS(self.code, 'ths_month_contract_code_future', '', '',
                          start_date, end_date)

        # 当前交易合约设置为序列的第一个 在后续遍历过程中动态检查 发生换月时更新
        temp_ZL = data_obj.data['ths_month_contract_code_future'][0]

        # 获取第一个时间戳作为开始时间 这些字符串的操作都是为了符合同花顺接口的参数调用形式
        temp_start_date = data_obj.data['time'][0]

        # 初始化一个空的数据框 用来装最后合并好的全部数据
        combined_df = pd.DataFrame()

        # 这里是为了将获取交易所后缀将合约代码更改为符合同花顺接口的方式
        # 在上面得到的当前交易合约序列中 代码如I2309 但要加上交易所后缀才能正常调用获取高频数据的函数
        temp_suffix = self.code.split('.')[1]

        # 遍历交易合约序列 如果当前日期的交易合约跟下一天的一样 跳过（代表不需要换月）
        # 如果不一样 则更新temp end date 代表我们找到了当前这张合约是主力的时间段
        # 注意在完成了当前这张合约的数据清洗后 要对应更改temp start date，temp ZL 和 temp end date
        for i in range(len(data_obj.data['ths_month_contract_code_future'])):

            if i < len(data_obj.data['ths_month_contract_code_future']) - 1:
                if data_obj.data['ths_month_contract_code_future'][i] == \
                        data_obj.data['ths_month_contract_code_future'][
                            i + 1]:
                    continue
                else:
                    temp_end_date = data_obj.data['time'][i]
            else:
                temp_end_date = data_obj.data['time'][i]

            # 把上面获取到的后缀加上
            temp_ZL += f'.{temp_suffix}'

            # 这里有非常繁琐的时间戳和字符串来回转换的步骤 主要原因有两点：1，同花顺的数据接口必须要以字符串的方式传入；
            # 2，需要通过时间戳来完成对数据框的筛选；
            # 这里会有很多个起止时间 建议不要看代码 直接看注释来理解 首先我们有个开始时间 这个是我们回测区间段的开始时间
            # 因为滑动平均（假设用前60期）等指标会让前60行数据没有对应的滑动平均（也就是看盘软件中均线不从合约上市第一天开始）
            # 一张合约不会在上市初始成为主力 所以我们只需要在获取数据时 把开始时间往前推 就可以获取足够多的数据使得均线
            # 在开始时间就能画出来 另一方面 日频的数据需要往前推更长时间才能使得从开始时间这一时间点就有日频的ma60
            # 传入获取当前主力合约高频数据的开始时间字符串 夜盘第一分钟的k线要后面再加上
            # 因为可能策略不是1分钟频的 避免出现没办法在同一时间换月的问题
            temp_start_date_ori = temp_start_date + ' 21:02:00'

            # 传入获取当前主力合约高频数据的结束时间字符串
            temp_day_end_string = temp_end_date + ' 15:00:00'

            # 这个开始时间是为了获取加在数据开头的夜盘第一分钟的k线
            # 测试过如果想通过开始和结束时间相同的方式来获取这一分钟的k线 有些时候行有些时候不行
            # 为避免出现该情况 多设置一个开始时间
            transfer_append_start = temp_start_date + ' 15:00:00'

            # 这个结束时间是为了获取加在数据开头的夜盘第一分钟的k线
            transfer_append_end = temp_start_date + ' 21:01:00'

            # 这个结束时间是为了获取加在数据结尾的夜盘第一分钟的k线
            temp_night_start_string = temp_end_date + ' 21:01:00'

            # 将回测区间开始和结束时间转换为时间戳格式
            start_date_object = datetime.strptime(temp_start_date_ori, "%Y-%m-%d %H:%M:%S")

            end_date_object = datetime.strptime(temp_day_end_string, "%Y-%m-%d %H:%M:%S")

            # 这里是调整一下要往前回推多少天 比如说当前开始日期3月27 如果我们的策略是在小时频上去做 为了从3月27开始就获得
            # ma60 需要获取前60个小时的交易数据相当于10个交易日 还需要考虑到周末的情况所以回推30日是比较稳妥的选择
            # 这样可以获取到充足的数据来计算ma60 避免出现数据缺失
            # isinstance(self.strategy,str)用来检查self.strategy是否为字符串 因为有可能是日频策略 取值为‘daily’
            # 这种情况下要回推更长时间来计算日频的ma60 经过检查 从一张合约上市到有日频ma60出现大概需要3个月 回推100天比较稳妥
            if not isinstance(strategy.time_frequency, str):
                if strategy.time_frequency <= 5:
                    days_to_shift = 10
                else:
                    days_to_shift = 30
            else:
                days_to_shift = 100

            temp_start_date_shift = start_date_object - timedelta(days=days_to_shift)

            temp_start_date_shift_string = temp_start_date_shift.strftime("%Y-%m-%d")

            # 这里是不管策略在什么频率下 都希望添加上日频的均线 所以要往前回推100天
            hundred_days_ago = start_date_object - timedelta(days=120)

            hundred_days_ago_string = hundred_days_ago.strftime("%Y-%m-%d")

            temp_start_date = temp_start_date_shift_string + ' 09:00:00'

            # temp_end_date += ' 23:00:00'

            # 获取当前主力在策略需要频率下的数据
            if strategy.time_frequency == 'daily':
                temp_df = THS_HQ(temp_ZL, 'open,high,low,close,volume,changeRatio', '', hundred_days_ago_string,
                                 end_date).data
            else:
                temp_df = THS_HF(temp_ZL, 'open;high;low;close;volume,changeRatio_periodical',
                                 f'Fill:Original,Interval:{strategy.time_frequency}',
                                 temp_start_date, temp_day_end_string).data

            # 用当前主力初始化一个Data类对象 这一步是为了再初始化一个Backtest类对象然后调用calculate indicator的类方法
            # 这里其实可以修改calculate indicator然后直接把这个数据框丢进去计算 但我觉得通过面向对象的理解会封装的更好
            temp_data_obj = Data(temp_ZL)

            # 将这个Data类对象的data值赋为上面获取到的高频序列
            temp_data_obj.data = temp_df

            # 这里要重命名一下列名 因为同花顺接口拿到的数据的列名跟一开始用的不一样 在这里重命名一下比改整个框架中的变量名方便
            new_column_names = {
                'time': 'date',
                'thscode': '证券代码',
                'changeRatio': 'change_ratio',
                'changeRatio_periodical': 'change_ratio'
            }

            temp_data_obj.data.rename(columns=new_column_names, inplace=True)

            temp_data_obj.calculate_indicator(strategy.indicator_require)

            # 把带有缺失值的行剔除掉
            temp_data_obj.data.dropna(inplace=True)

            # 这里是对数据按照回测需要的时间段进行筛选 因为刚刚上面提到了为了保证回测一开始就有均线 我们在获取数据时把开始时间往前推了
            # 现在要根据回测指定的时间段把数据筛选出来
            temp_data_obj.data['date'] = pd.to_datetime(temp_data_obj.data['date'])

            temp_data_obj.data = temp_data_obj.data[temp_data_obj.data['date'] >= start_date_object]

            temp_data_obj.data = temp_data_obj.data[temp_data_obj.data['date'] <= end_date_object]

            # 筛选后的数据要重新编排索引
            temp_data_obj.data = temp_data_obj.data.reset_index(drop=True)

            if not isinstance(strategy.time_frequency, str):
                # 如果当前回测的数据并不是日频 则需要在数据开头和末尾各添加一根1分钟的k线用来换月
                # 日频数据能获取到非常长时间之前的 找不到这样的1分钟k线来添加 会报错 如果之后用了正式版的接口 则把这个判断去掉即可
                data_to_append_end_all = THS_HF(temp_ZL, 'open;high;low;close;volume;changeRatio_periodical',
                                                'Fill:Original', temp_day_end_string,
                                                temp_night_start_string).data

                data_to_append_start_all = THS_HF(temp_ZL, 'open;high;low;close;volume;changeRatio_periodical',
                                                  'Fill:Original',
                                                  transfer_append_start,
                                                  transfer_append_end).data

                data_to_append_end_all.rename(columns=new_column_names, inplace=True)

                data_to_append_start_all.rename(columns=new_column_names, inplace=True)

                data_to_append_end_all['date'] = pd.to_datetime(data_to_append_end_all['date'])

                data_to_append_start_all['date'] = pd.to_datetime(data_to_append_start_all['date'])

                data_to_append_end = data_to_append_end_all.iloc[-1]

                data_to_append_start = data_to_append_start_all.iloc[-1]
                # 向数据末尾添加夜盘开盘第一分钟k线
                temp_data_obj.data = pd.concat([temp_data_obj.data, data_to_append_end.to_frame().transpose()],
                                               ignore_index=True)

                # 使用上一行的值填充缺失的值
                temp_data_obj.data = temp_data_obj.data.ffill()

                # 向数据开头添加夜盘开盘第一分钟k线
                temp_data_obj.data = pd.concat([data_to_append_start.to_frame().transpose(), temp_data_obj.data],
                                               ignore_index=True)

                # 使用下一行的值填充缺失的列
                temp_data_obj.data = temp_data_obj.data.bfill()

            # 这里是先把当前的时间戳保存下来 因为后续跟日频数据合并时 是要转换成日
            # 具体来说 现在的时间戳形如2023-01-01 09:00:00 但如果想要添加日频均线则需要 转换成2023-01-01
            # 这里就涉及到 日频的均线应该要shift 比如说在2023-01-01 09:00:00 我们还没有当天收盘价 不应该画出包括2023-01-01的均线
            # 但如果shift一下 在2023-01-01 09:00:00后面画上包括2022-12-31的ma60 就合理了
            # 从另一个角度来说 一天的价格对ma60影响不会太大 shift or not可能不会产生本质影响
            reserve_date = pd.to_datetime(temp_data_obj.data['date'])

            temp_data_obj.data['date'] = temp_data_obj.data['date'].dt.date

            if strategy.need_daily:
                # 如果当前策略需要用到日频数据 则根据当前的主力获取一下

                # 再初始化一个Data类对象 这里是为了直接调用Data的类函数 去获取当前主力的日频数据
                data_obj_for_daily = Data(temp_ZL)

                # getdata完之后 日频数据已经封装到这一对象的类属性中 方便调用
                data_obj_for_daily.getdata('daily', hundred_days_ago_string, end_date)

                data_obj_for_daily.calculate_indicator(strategy.indicator_require)

                data_obj_for_daily.data['date'] = pd.to_datetime(data_obj_for_daily.data['date']).dt.date

                # 重命名一下日频数据 加上daily的后缀 避免跟高频数据那边冲突
                new_columns = {col: col + '_daily' if col != 'date' else col for col in data_obj_for_daily.data.columns}

                data_obj_for_daily.data = data_obj_for_daily.data.rename(columns=new_columns)

                # 因为高频数据那边已按日期筛选好 直接合并即可 此处左连接 保留全部的高频数据 多余的日频数据就自动舍弃了
                temp_merge_result = pd.merge(temp_data_obj.data, data_obj_for_daily.data, on='date', how='left')
            else:
                temp_merge_result = temp_data_obj.data

            # 把时间换回刚刚保存好的带时分秒的格式
            temp_merge_result['date'] = reserve_date

            # 此处我们已经针对某一个主力合约完成了数据清洗 把结果放到上面留空的combined df中 后续每一个主力的数据都会拼到这里
            combined_df = pd.concat([combined_df, temp_merge_result], ignore_index=True)

            # 这里多加一个判断条件 防止索引出界
            if i == len(data_obj.data['ths_month_contract_code_future']) - 1:
                break

            # 完成了前面的步骤后 更换当前主力和时间为下一个时间步 继续遍历
            temp_start_date = data_obj.data['time'][i]

            temp_ZL = data_obj.data['ths_month_contract_code_future'][i + 1]

        # 将最终合并的结果放到类变量中 这里我们就得到了一个数据框 每一行是一个时间步的数据 包含open high low close以及策略所需要的指标
        # 同时处理了换月 即当前时间步下的数据一定是当前主力合约的数据
        self.data = combined_df

        # 获取当前文件所在的目录路径
        current_directory = os.getcwd()

        # 创建 "raw data" 文件夹路径
        raw_data_directory = os.path.join(current_directory, "raw data")

        # 确保 "raw data" 文件夹存在，如果不存在则创建
        if not os.path.exists(raw_data_directory):
            os.makedirs(raw_data_directory)

        # 拼接 Excel 文件路径
        excel_file_path = os.path.join(raw_data_directory, f'{self.code}_{strategy.time_frequency}.xlsx')

        # 将数据保存到 Excel 文件中
        self.data.to_excel(excel_file_path, index=False)

    def calculate_indicator(self, indicator_require):
        # 先遍历当前strategy需要哪些指标 self.strategy.indicator_require这一字典的key是指标种类 value是参数
        for key in indicator_require.keys():

            # 通过Data类中的data pipline来获取当前指标对应的函数
            func = self.data_pipline[key]  # Get the function used to calculate the current indicator

            # 遍历当前该指标需要哪些参数 对每一个参数都调用一次计算该指标的函数
            for ele in indicator_require[key]:

                # 有些指标不需要参数 如TR vwap 所以再self.strategy.indicator_require会以 'TR':[True]这种形式储存
                # 检查ele是否为布尔型 如果是 则不需要传参
                if isinstance(ele, bool):
                    func()

                    continue

                func(ele)  # Call the above function

    def initial_data(self, strategy, start_date, end_date):

        current_dir = os.getcwd()  # 获取当前工作路径
        file_name = f"Raw Data/{self.code}_{strategy.time_frequency}.xlsx"
        file_path = os.path.join(current_dir, file_name)  # 组合文件路径
        if os.path.exists(file_path):  # 检查文件是否存在
            print(f"已经有数据")
            self.data = pd.read_excel(file_name)
        else:
            print(f"暂时无数据")
            if strategy.time_frequency == 'daily':
                self.getdata('daily', start_date, end_date)
                self.calculate_indicator(strategy.indicator_require)
            else:
                self.combine_all_data(strategy, start_date, end_date)

        self.basic_info(end_date)

    def MA(self, n):
        """

        :param n: The length of sliding window
        :return: A Data object with moving average of past n days added to self.data
        """

        data = self.data
        # calculating the moving average of past n days also the middle band in bollinger band
        data[f'ma{n}'] = data['close'].rolling(window=n, min_periods=n).mean().round(1)  # shift?

        # because the first n-1 data points can not find n previous data to look back so fill the missing value with 0
        data.fillna(0)


    def EMA(self, n):
        """

        :param n: The length of sliding window
        :return: A Data object with exponential moving average of past n days added to self.data
        """

        # calculating the ma of past n days first
        self.MA(n)

        data = self.data

        # calculating the ema of past n days
        data[f'ema{n}'] = ((data[f'ma{n}'].shift(1) * (n - 1) + 2 * data['close']) / (n + 1)).round(1)


    def STD(self, n):
        """

        :param n: The length of sliding window
        :return: A Data object with standard deviation of past n days added to self.data
        """

        data = self.data

        # calculating the standard deviation of past n days
        data[f'std{n}'] = data['close'].rolling(window=n, min_periods=n).std().round(1)

        # because the first n-1 data points can not find n previous data to look back so fill the missing value with 0
        data.fillna(0)


    def TR(self):
        """

        :return: A Data object with true range added to self.data
        """

        data = self.data

        # calculating the tr of current time gap
        data['tr'] = data['high'] - data['low']


    def ATR(self, n):
        """

        :param n: The length of sliding window
        :return: A Data object with average true range of past n days added to self.data
        """

        # calculaitng the tr first
        self.TR()

        data = self.data

        # calculating the atr of past n days
        data[f'atr{n}'] = data['tr'].rolling(window=n, min_periods=n).mean().round(1)

        # because the first n-1 data points can not find n previous data to look back so fill the missing value with 0
        data.fillna(0)


    def Bollinger_band(self, n):
        """
        计算参数为n的布林轨 n用来控制中间轨（moving average）的回溯窗口大小
        """

        # calculating the ma of past n days first(middle band)
        self.MA(n)

        # calculating the std of past n days
        self.STD(n)

        data = self.data

        # calculating the lower band
        data['lower_band'] = (data[f'ma{n}'] - 2 * data[f'std{n}']).round(1)

        # calculating the upper band
        data['upper_band'] = (data[f'ma{n}'] + 2 * data[f'std{n}']).round(1)

    def vwap(self):
        """
        计算每一根k线交易量加权下的价格 每一天重置
        """

        data = self.data

        data['date'] = pd.to_datetime(data['date'])

        temp_date = data['date'].iloc[0].date

        data['vwap'] = np.nan

        data['vwap'].iloc[0] = data['close'].iloc[0]

        cumulative_volume = data['volume'].iloc[0]

        cumulative_vol_times_price = data['volume'].iloc[0] * data['close'].iloc[0]

        for i in range(1, len(data)):
            current_date = data['date'].iloc[i].date
            if current_date != temp_date:
                temp_date = current_date
                data['vwap'].iloc[i] = data['close'].iloc[i]
                cumulative_volume = data['volume'].iloc[i]
                cumulative_vol_times_price = data['volume'].iloc[i] * data['close'].iloc[i]
                continue
            cumulative_volume += data['volume'].iloc[i]
            cumulative_vol_times_price += data['volume'].iloc[i] * data['close'].iloc[i]
            data['vwap'].iloc[i] = cumulative_vol_times_price / cumulative_volume

        data['vwap'] = round(data['vwap'], 2)

    def magic_nine_turn(self):
        """
        在每一天对比四天前的价格 如果连续9天都是小于或大于（不能交替） 则视为出现一个九转信号 从这连续的9天的第一天开始标号
        """

        data = self.data

        data['magic_nine_turn'] = np.nan

        direction = 0

        count = 0

        start = 0

        for i in range(4, len(data)):

            if count == 9:

                for k in range(9):
                    data['magic_nine_turn'].iloc[start + k] = int(k + 1)

                count = 0

                direction = 0

                continue

            temp_price = data['close'][i]

            prior_price = data['close'][i - 4]

            if temp_price < prior_price:

                temp_direction = -1

            else:

                temp_direction = 1

            data['magic_nine_turn'].iloc[i] = 0

            if direction == 0:
                direction = temp_direction

                count += 1

                start = i

                continue

            if direction != temp_direction:

                direction = 0

                count = 0

                continue
            else:

                count += 1


