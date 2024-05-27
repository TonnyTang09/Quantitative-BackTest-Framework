import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
from iFinDPy import *
from Data import Data
import akshare as ak
import Strategy
from datetime import date, timedelta, datetime
import time

# 利用SMTP协议自动发送邮件 后续可以在receiver_email中加入更多接收方邮件 也可以更换发送方邮件（需要在邮箱里开启smtp协议 更换邮箱和密码）
def send_email(subject, body):
    # 配置邮箱参数
    sender_email = "tangzheng0206@163.com"  # 发送方邮箱
    receiver_email = ["bingyan.liu@majestic-rock.com"]  # 接收方邮箱
    password = "YPOSCEDTIBFKJJAC"  # 发送方邮箱密码

    # 创建邮件内容
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = ", ".join(receiver_email)
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # 连接到SMTP服务器
    with smtplib.SMTP("smtp.163.com", 25) as server:
        server.starttls()  # 使用TLS加密通信
        server.login(sender_email, password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)


class monitor_price_movement:
    """
    用这个类来监视标的日盘价格是否突破夜盘最高或最低
    """

    def __init__(self, underlying_list):
        """

        :param underlying_list: 想要监控的标的 以合约代码的形式放在一个list中
        """

        self.underlying_list = underlying_list  # 监控标的

        self.dynamic_monitor = {}  # 用一个动态维护的字典来监视是否突破 只记录每天的第一次突破 避免冗余

        self.count_target = len(underlying_list)  # 用当前监控标的的数量来判断是否每一个品种当天都完成了上下两个方向的突破 如果是 提前结束程序

        self.count = 0  # 用该变量维护当前完成了几次突破

        current_date = date.today()  # 今日日期 因为可能会遇到当前日期是某个节假日后 所以要往前推几天 才能获取到最近的一个夜盘数据

        # 下面是初始化动态维护的数组 需要获取昨日夜盘的最高最低
        for underlying in underlying_list:

            self.dynamic_monitor[underlying] = {}

            self.dynamic_monitor[underlying]['high_symbol'] = False

            self.dynamic_monitor[underlying]['low_symbol'] = False

            for i in range(1, 10):

                previous_date = (current_date - timedelta(days=i)).strftime("%Y-%m-%d")

                start = previous_date + ' 15:00:01'

                end = previous_date + ' 23:00:01'

                temp_data = THS_HF(underlying, 'high;low', 'Interval:60', start, end).data

                if not temp_data.empty:
                    self.dynamic_monitor[underlying]['upper_barrier'] = temp_data['high'].max()

                    self.dynamic_monitor[underlying]['lower_barrier'] = temp_data['low'].min()

                    break

    def check_barrier(self):
        # 遍历监控的标的 获取当前时间的最新价格 与夜盘最高最低比较 如果符合条件 则发送邮件提醒
        for underlying in self.underlying_list:

            current_time = datetime.now()
            formatted_current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

            one_minute_ago = current_time - timedelta(minutes=1)
            formatted_one_minute_ago = one_minute_ago.strftime("%Y-%m-%d %H:%M:%S")

            temp_data = THS_HF(underlying, 'high;low', 'Fill:Original', formatted_one_minute_ago, formatted_current_time).data

            temp_high = temp_data['high'].iloc[-1]

            temp_low = temp_data['low'].iloc[-1]

            temp_time = temp_data['time'].iloc[-1]

            if temp_high >= self.dynamic_monitor[underlying]['upper_barrier']: #

                if not self.dynamic_monitor[underlying]['high_symbol']:
                    subject = f"{underlying}价格突破夜盘最高"

                    body = f"夜盘最高：{self.dynamic_monitor[underlying]['upper_barrier']}" + f"当前价格：{temp_high}" + f"突破时间：{temp_time}"

                    self.dynamic_monitor[underlying]['high_symbol'] = True

                    send_email(subject, body)

            if temp_low <= self.dynamic_monitor[underlying]['lower_barrier']: #

                if not self.dynamic_monitor[underlying]['low_symbol']:
                    subject = f"{underlying}价格跌破夜盘最低"

                    body = f"夜盘最低：{self.dynamic_monitor[underlying]['lower_barrier']}" + f"当前最低：{temp_low}" + f"突破时间：{temp_time}"

                    self.dynamic_monitor[underlying]['low_symbol'] = True

                    send_email(subject, body)

            if self.dynamic_monitor[underlying]['high_symbol'] and self.dynamic_monitor[underlying]['low_symbol']:

                self.count += 1

                if self.count == self.count_target:

                    return False

        return True


if __name__ == '__main__':

    THS_iFinDLogin("xhlh009", "5e1295")

    monitor = monitor_price_movement(['JM2409.DCE', 'I2409.DCE'])

    monitoring = True

    while monitoring:

        now = datetime.now().time()
        if now >= datetime.strptime('15:00', '%H:%M').time():
            break

        monitoring = monitor.check_barrier()

        monitor.run_monitor()

        time.sleep(60)