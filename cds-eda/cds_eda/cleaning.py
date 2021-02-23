import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

class cleaning():
    
    def __init__(self, df, file):
        self.df = df
        self.file = file 

    def dataFrameCleaning(self):        
        if self.file == 'Quotes':
            self.df.drop(10, axis=1, inplace = True)
            self.df.columns = ['ticker', 'valueDate', 'priceType', 'priceDirection', 'size', 'spread', 'price', 'upfront', 'switchStatus', 'firm']
            self.df['valueDate'] = pd.to_datetime(self.df['valueDate'])
            self.df.set_index('valueDate', inplace = True)
        elif self.file == 'Universe':
            self.df.columns = ['ticker_universe','creditCurve', 'label', 'endDate', 'maturity','seniority','docClause','currency', 'coupon','instrument','family']
            self.df['endDate'] = pd.to_datetime(self.df['endDate'])
        elif self.file == 'Trades':
            self.df.drop(10, axis=1, inplace = True)
            self.df.columns = ['ticker', 'valueDate', 'priceType', 'priceDirection', 'size', 'spread', 'price', 'upfront', 'switchStatus', 'firm']
            self.df['valueDate'] = pd.to_datetime(self.df['valueDate'])
            self.df['valueDate_hour'] = self.df['valueDate']
            self.df.sort_index(inplace=True, ascending=True)
        return self.df



