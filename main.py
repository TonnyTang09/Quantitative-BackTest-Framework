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
    Example for how to run a back test
    """
    pd.set_option('future.no_silent_downcasting', True)
    
    THS_iFinDLogin("xhlh009", "5e1295")

    obj = Data('SMZL.CZC')  # Initialize the Data object of underlying contract

    strategy = Strategy.momentum_1h_updated()  # Initialize the Strategy object

    obj.initial_data(strategy, '2023-07-01', '2024-5-27')

    # Use the Data object and the Strategy object to initialize a Backtest object
    result = Backtest(obj, strategy)

    result.run_backtest()

    plot = Plot(result.data, obj.code, strategy)  # Use the Backtest object to initialize a Plot object

    plot.cumulative_return()  # Plot the change of the balance in our account

    plot.k_chart()  # Plot the interactive K-chart

    """
    Example for how to plot a specific contract within specific time range
    # """
    # test_plot = specific_Plot('SA309.CZC', '2023-06-01', '2023-10-01', 1, {'ma':[5,10]})
    #
    # test_plot.specific_k_chart()