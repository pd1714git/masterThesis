import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

class eda():
    """EDA class which enables the user to explore, format and clean the dataframe as he wants, 
    """
    def __init__(self, df, family=False):
        """Init function

        Args:
            df (df): the dataframe to be transformed
            family (list(str)): the cds families to be studied
        """
        self.df = df
        self.family = family
    
    def addUniverse(self, df_universe):
        """Adds information about the cds family

        Args:
            self.df (df): The dataframe to add the universe on
            df_universe (df): The universe data framke

        Returns:
            self.df (df): The dataframe with the information about the cds
        """
        self.df = self.df.merge(df_universe, how='inner', left_on = 'ticker', right_on = 'ticker_universe')
        return self.df
    
    def selectCdsFamily(self):
        """Keep specific families of cds, if no input was passed then the function returns all of the functions
        Args:
            self.df (df): The dataframe with all the cds

        Returns:
            self.df (df): The dataframe with the right cds
        """
        if not self.family:
            self.family = list(self.df.family.unique())
        ## Keep specific families of cds
        self.df = self.df.loc[self.df['family'].isin(self.family)]
        return self.df

    def versionsDay(self):
        """Compute the number of ticker versions per family and per day
        Args:
            self.df (df): The dataframe with all the cds

        Returns:
            self.df (df): The dataframe with the number of versions of a cds per day, the maximum version
        """
        # Compute the number of versions of a specfic cds is being traded every year.
        tradesVrsDay = self.df.set_index('valueDate')
        tradesVrsDay = tradesVrsDay.groupby(['family', pd.Grouper(freq='D')]).agg({'ticker': pd.Series.nunique})
        tradesVrsDay.columns = ['NbVersionTickers']
        tradesVrsDay.reset_index(inplace=True)
        tradesVrsDay.set_index('valueDate',inplace=True)
        
        # Add the number of versions traded each day per family to the main dataframe
        self.df.set_index('valueDate',inplace=True)
        self.df = self.df.merge(tradesVrsDay,
                                how='left',
                                left_on= ['family', self.df.index.year,self.df.index.month,self.df.index.day], 
                                right_on = ['family', tradesVrsDay.index.year,tradesVrsDay.index.month,tradesVrsDay.index.day])
        self.df.drop(['key_1','key_2','key_3'], axis=1, inplace= True)
        
        # Create a column with the version number of the cds 
        self.df['Version'] = self.df['ticker'].apply(lambda x: x[-2:])
        self.df['valueDate'] = self.df['valueDate_hour']
        self.df.set_index('valueDate_hour', inplace = True)
        
        # Create a column with the latest version of the cds per family and per day
        df_version = self.df.groupby(['family', pd.Grouper(freq='D')]).agg({'Version': 'max'})
        df_version.columns = ['MaxVersion']
        df_version.reset_index(inplace=True)
        df_version.set_index('valueDate_hour', inplace = True)

        # Add the maximum version of the cds per family and per day    
        self.df = self.df.merge(df_version,
            how='left',
            left_on= ['family',  self.df.index.year,self.df.index.month,self.df.index.day],
            right_on = ['family',  df_version.index.year,df_version.index.month,df_version.index.day])
        self.df.drop(['key_1', 'key_2', 'key_3'], axis=1, inplace=True)
        return self.df

    def keepLatestVersion(self):
        """Keep cds which correspond to the latest version of that cds available on a specific day
        Args:
            self.df (df): The dataframe with all the cds

        Returns:
            self.df (df): The dataframe with only the latest version per cds and per day.
        """
        self.df = self.df.query('Version == MaxVersion').copy()
        return self.df

    def removeInsufficientData(self, proportion, daysDiffLimit, Group = 'Y'):
        """Removes dates where there is not enough sufficient data for the data to be considered reliable
        Args:
            self.df (df): The dataframe with all the cds
            proportion (float): The proportion of cds with high number of days between consecutive trades
            daysDiffLimit (int): The number of days alllowed between consecutive trades.
            Group (str): compute and remove the data according to either year ('Y') or month ('M') 

        Returns:
            self.df (df): The dataframe with cds which are frequently traded.
        """
        # remove trades where no spread had been recorded.
        self.df.dropna(subset=['spread'], inplace=True)
        self.df.set_index('valueDate',inplace=True)
        g = self.df.groupby(['family',pd.Grouper(freq='D')])
        
        # aggregate each cds according to family and day.
        df_mean = g.apply(lambda x: pd.Series([np.average(x['spread'], weights=x['size'])], index =['weighted_spread']))
        df_mean['mean_spread'] = self.df.groupby(['family',pd.Grouper(freq = 'D')]).agg({'spread':'mean'})
        df_mean.reset_index(inplace=True)

        # Compute the difference in days for each cds family between the current date and the previous date where the trade was done.
        df_mean['date_diff'] = (df_mean['valueDate']
                        .groupby(df_mean['family'])
                        .diff()
                        .dt.days
                        .fillna(0, downcast='infer'))
        df_mean.set_index(['valueDate'], inplace=True)
        df_mean.sort_index(inplace=True, ascending=False)

        # Compute according to cds family and either month or year the number trades occuring too late and the total number of day traded.
        df_grouped = df_mean.groupby(['family',pd.Grouper(freq=Group)]).agg({'date_diff':lambda x: x[x > daysDiffLimit].count(),'mean_spread':'count'}).rename(columns={'date_diff':'daysMissing','mean_spread':'total_days'})
        # Compute the proportion of days which can be classified as outliers
        df_grouped['proportion_outlier'] = df_grouped['daysMissing']/df_grouped['total_days']
        df_grouped.reset_index(inplace=True)
        df_grouped['month'] = df_grouped['valueDate'].dt.month
        df_grouped['year'] = df_grouped['valueDate'].dt.year  
        df_grouped.set_index(['family','year','month'],inplace=True)
       
        # Select dates and families which can be considered as outliers
        df_grouped = df_grouped.loc[df_grouped['proportion_outlier']>proportion,:]
        df_grouped.drop('valueDate',axis=1,inplace=True)
        
        # Merge the outliers to the original dataframe
        self.df.reset_index(inplace=True)
        self.df['month'] = self.df['valueDate'].dt.month
        self.df['year'] = self.df['valueDate'].dt.year  
        self.df.set_index(['family','year','month'],inplace=True)
        self.df = self.df.merge(df_grouped, how='left',left_index=True,right_index=True)

        # Fill non outliers with 0 value
        self.df['proportion_outlier'] = self.df['proportion_outlier'].fillna(value=0)
        # select non outliers values
        self.df= self.df.loc[self.df['proportion_outlier']==0,:]

        self.df.reset_index(inplace=True)
        return self.df
    

    def removeOutliers(self):
        """Remove Outliers based on the value of the spread
        Args:
            self.df (df): The dataframe with all the cds

        Returns:
            self.df (df): The dataframe with no outliers
        """
        self.df.reset_index(inplace=True)
        self.df.reset_index(inplace=True)
        self.df.rename(columns = {'index':'TransactionID'}, inplace=True)
        self.df.set_index('valueDate', inplace=True)
        self.df.sort_index(inplace=True)

        # mad score template
        mad1 = lambda x: np.fabs(x - x.median()).median()
        df_trades_mad = self.df.groupby('family').rolling(window=100).agg({'spread':[mad1,'median'], 'TransactionID': 'max'}).reset_index()
        df_trades_mad.columns = ['family','valueDate','MADRollingSpread','Median','TransactionID']
        df_trades_mad.dropna(subset=['Median'], inplace=True)
        df_trades_mad['TransactionID'] = df_trades_mad['TransactionID'].astype(int)

        # Add mad score
        self.df = self.df.merge(df_trades_mad, how='left', left_on = ['TransactionID'], right_on = ['TransactionID'])
        self.df.dropna(subset=['MADRollingSpread','Median'], inplace=True)
        self.df['ModifiedZScore'] = 0.6745* (self.df['spread']-self.df['Median'])/self.df['MADRollingSpread']
        
        # Select trades with z scores < 3.5
        self.df = self.df.loc[abs(self.df['ModifiedZScore'])<=3.5,:]

        # Cleanup data frame
        self.df.set_index('TransactionID',inplace=True)
        self.df.drop('family_y', axis=1,inplace=True)
        self.df = self.df.rename(columns={'family_x':'family'})
        return self.df

    def aggregateTimeSimilar(self):
        """Aggregates trades which have similar family, date and time
        Args:
            self.df (df): The dataframe with all the cds

        Returns:
            self.df (df): The dataframe with no duplicates
        """
        self.df = self.df.groupby(['family','valueDate']).agg({'spread':'median', 'size': 'median'}).reset_index().set_index('valueDate')
        return self.df

    
    
