# Authors:
# - Ahmad Amine Loutfi
# - Mengato Sun

import sys
import time

sys.path.append(r'D:\Time Series')

from KNN import get_X_Y_B3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

datalist = [get_X_Y_B3()]

for i,each in enumerate(datalist):
    x, y, t_x, t_y,df = each
    x['Spot Prices (Auction) (EUR)']=y
    t_x['Spot Prices (Auction) (EUR)']=t_y
    result = pd.concat([x,t_x])


# Correlation coefficients
    cor1 = np.abs(result.corr(method='pearson'))
    cor2 = np.abs(x.corr(method='spearman'))
    cor3 = np.abs(x.corr(method='kendall'))
    listcor = [cor1,cor2,cor3]
   
    result['Spot Prices (Auction) (EUR)'].plot.box()
    plt.grid(linestyle="--", alpha=0.3)
    plt.show()
